"""
m03_create_risk_maps.py
Create risk heatmaps using signals from m02.

Uses pre-computed signals (SAM + CLIP + gaps) from m02_extract_signals.py
to create visual risk maps showing where the image deviates from baseline.

Visualization Logic:
  - v0_original: No overlay (clean reference image)
  - Variations: RED overlay ONLY on segments with issues:
    * Segments with gaps (wrong_color, missing object, wrong_object)
    * Segments with low prompt_score (< v0_score * threshold)
  - NO green/orange highlights for correct objects

Run:
    python -u src/m03_create_risk_maps.py 2>&1 | tee logs/m03_$(date +%Y%m%d_%H%M%S).log
"""

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from utils.mask_utils import decode_mask_rle


# ============================================================================
# Configuration
# ============================================================================
class Config:
    PROJECT_ROOT = Path(__file__).parent.parent
    DB_PATH = PROJECT_ROOT / "outputs" / "centralized.db"
    # Note: image_path comes from DB (model-specific dirs handled by m01)
    OUTPUT_DIR = PROJECT_ROOT / "outputs" / "m03_risk_maps"

    # Overlay config - OUTLINE mode (preserves original colors)
    OUTLINE_THICKNESS = 4  # Pixels for red border around bad segments
    OUTLINE_COLOR = (255, 0, 0)  # Pure red for maximum visibility
    SCORE_THRESHOLD_RATIO = 0.92  # Mark as "bad" if score < v0_score * ratio


# ============================================================================
# Database
# ============================================================================
def init_risk_maps_table(conn: sqlite3.Connection) -> None:
    """Create risk_maps table."""
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS risk_maps")
    cursor.execute("""
        CREATE TABLE risk_maps (
            image_id TEXT PRIMARY KEY,
            risk_map_path TEXT NOT NULL,
            num_bad_segments INTEGER,
            avg_object_score REAL,
            v0_ref_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()


def get_signals_with_images(conn: sqlite3.Connection) -> list:
    """Get signals data joined with image info."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT s.image_id, s.prompt, s.objects_json, s.gradcam_path,
               s.gaps_json, s.num_gaps, s.avg_object_score,
               i.image_path, i.variation, i.base_id, i.baseline_prompt
        FROM signals s
        JOIN images i ON s.image_id = i.image_id
        ORDER BY i.base_id, i.variation
    """)
    return cursor.fetchall()


def get_v0_scores_by_base(conn: sqlite3.Connection) -> dict:
    """Get v0_original avg_object_score for each base_id as reference."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT i.base_id, s.avg_object_score
        FROM signals s
        JOIN images i ON s.image_id = i.image_id
        WHERE i.variation = 'v0_original'
    """)
    return {row[0]: row[1] for row in cursor.fetchall()}


def get_v0_images_by_base(conn: sqlite3.Connection) -> dict:
    """Get v0_original image_path for each base_id."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT base_id, image_path
        FROM images
        WHERE variation = 'v0_original'
    """)
    return {row[0]: row[1] for row in cursor.fetchall()}


def save_risk_map_record(conn: sqlite3.Connection, image_id: str, risk_map_path: str,
                         num_bad: int, avg_score: float, v0_score: float) -> None:
    """Save risk map record to database."""
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO risk_maps
        (image_id, risk_map_path, num_bad_segments, avg_object_score, v0_ref_score)
        VALUES (?, ?, ?, ?, ?)
    """, (image_id, risk_map_path, num_bad, avg_score, v0_score))
    conn.commit()


# ============================================================================
# Risk Map Generation
# ============================================================================
def mask_to_outline(mask: np.ndarray, thickness: int = 4) -> np.ndarray:
    """
    Convert a filled mask to an outline mask.

    Uses morphological erosion: outline = mask - eroded_mask
    """
    if not mask.any():
        return mask

    # Create structure element for erosion
    struct = ndimage.generate_binary_structure(2, 1)

    # Erode the mask by thickness iterations
    eroded = ndimage.binary_erosion(mask, structure=struct, iterations=thickness)

    # Outline = original - eroded
    outline = mask & ~eroded

    return outline


def create_risk_overlay(
    image: Image.Image,
    objects: list,
    gaps: list,
    v0_score: float,
    config: Config,
    text_header_height: int = 80
) -> tuple[Image.Image, int]:
    """
    Create RED OUTLINE around bad segments (preserves original colors).

    Uses outline instead of fill so the underlying object colors remain visible.
    This is critical for v1_attribute (color changes) where we need to SEE
    the wrong color while marking it as incorrect.

    Bad segments:
    1. Segments with gaps (wrong_color, missing, wrong_object)

    Returns: (overlay_image, num_bad_segments)
    """
    img_array = np.array(image).astype(np.uint8)
    full_h, full_w = img_array.shape[:2]

    # Content area (excluding text header)
    content_h = full_h - text_header_height

    # Track bad pixels (filled mask first, then convert to outline)
    bad_mask = np.zeros((full_h, full_w), dtype=bool)
    num_bad = 0

    # Build set of segment_ids with gaps (wrong_color, missing, wrong_object)
    gap_segment_ids = set()
    for gap in gaps:
        seg_id = gap.get('segment_id', -1)
        if seg_id >= 0:
            gap_segment_ids.add(seg_id)

    # Check each detected object - ONLY mark segments with GAPS as bad
    for obj in objects:
        segment_id = obj.get('segment_id', -1)
        mask_rle = obj.get('mask_rle')

        # ONLY mark as bad if segment has a gap (wrong color, missing, wrong object)
        is_bad = segment_id in gap_segment_ids

        if not is_bad:
            continue

        num_bad += 1

        # Use mask if available, otherwise fall back to bbox
        if mask_rle:
            try:
                # Decode mask (this is for the cropped content area)
                segment_mask = decode_mask_rle(mask_rle)

                # If mask covers >30% of image, it's likely inverted (background)
                mask_coverage = segment_mask.sum() / segment_mask.size
                if mask_coverage > 0.30:
                    segment_mask = ~segment_mask

                # Place mask in the correct position (after text header)
                mask_h, mask_w = segment_mask.shape
                if mask_h == content_h and mask_w == full_w:
                    bad_mask[text_header_height:, :] |= segment_mask
                else:
                    end_y = min(text_header_height + mask_h, full_h)
                    end_x = min(mask_w, full_w)
                    bad_mask[text_header_height:end_y, :end_x] |= segment_mask[:end_y - text_header_height, :end_x]
            except Exception:
                # Fall back to bbox if mask decoding fails
                bbox = obj.get('bbox')
                if bbox:
                    x, y, bw, bh = bbox
                    x1, y1 = max(0, int(x)), max(0, int(y))
                    x2, y2 = min(full_w, int(x + bw)), min(full_h, int(y + bh))
                    if x2 > x1 and y2 > y1:
                        bad_mask[y1:y2, x1:x2] = True
        else:
            # No mask - use bbox outline
            bbox = obj.get('bbox')
            if bbox:
                x, y, bw, bh = bbox
                x1, y1 = max(0, int(x)), max(0, int(y))
                x2, y2 = min(full_w, int(x + bw)), min(full_h, int(y + bh))
                if x2 > x1 and y2 > y1:
                    bad_mask[y1:y2, x1:x2] = True

    # Convert filled mask to OUTLINE
    result = img_array.copy()
    if bad_mask.any():
        outline_mask = mask_to_outline(bad_mask, thickness=config.OUTLINE_THICKNESS)

        # Apply pure red color to outline pixels only
        result[outline_mask] = config.OUTLINE_COLOR

    return Image.fromarray(result), num_bad


def create_side_by_side(
    v0_image: Image.Image,
    variation_image_with_overlay: Image.Image,
    baseline_prompt: str,
    generation_prompt: str,
    variation: str,
    num_bad: int
) -> Image.Image:
    """
    Create side-by-side comparison: v0 (baseline) vs variation with RED outline.

    LHS: v0 image (clean reference)
    RHS: variation image with RED OUTLINE around different objects
    """
    w, h = v0_image.size
    combined = Image.new('RGB', (w * 2 + 20, h + 80), color=(255, 255, 255))

    # Paste images
    combined.paste(v0_image, (0, 70))
    combined.paste(variation_image_with_overlay, (w + 20, 70))

    # Add text using matplotlib
    fig, ax = plt.subplots(figsize=(combined.width / 100, combined.height / 100), dpi=100)
    ax.imshow(combined)
    ax.axis('off')

    # Title - show baseline prompt
    prompt_display = baseline_prompt[:80] + '...' if len(baseline_prompt) > 80 else baseline_prompt
    ax.text(combined.width / 2, 15, f'Baseline: "{prompt_display}"',
            ha='center', va='center', fontsize=9, fontweight='bold')

    # Subtitle - variation info
    bad_text = f"{num_bad} different object(s) outlined in RED" if num_bad > 0 else "No differences"
    ax.text(combined.width / 2, 35, f'{variation}: {bad_text}',
            ha='center', va='center', fontsize=9, color='red' if num_bad > 0 else 'green')

    # Labels
    ax.text(w / 2, 60, 'v0 (Baseline)', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(w + 20 + w / 2, 60, f'{variation} (RED outline = wrong)', ha='center', va='center', fontsize=9, fontweight='bold')

    # Convert to PIL
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    result = Image.frombuffer('RGBA', fig.canvas.get_width_height(), buf, 'raw', 'RGBA', 0, 1)
    result = result.convert('RGB')
    plt.close(fig)

    return result


# ============================================================================
# Main Pipeline
# ============================================================================
def generate_all_risk_maps(config: Config) -> None:
    """
    Generate risk maps for VARIATION images only.

    Format: LHS = v0 (baseline), RHS = variation with RED overlay on different objects.
    """
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(config.DB_PATH))

    # Check if signals table exists
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='signals'")
    if not cursor.fetchone():
        print("ERROR: No signals table found. Run m02_extract_signals.py first.")
        conn.close()
        return

    init_risk_maps_table(conn)

    signals_data = get_signals_with_images(conn)
    if not signals_data:
        print("No signals data found. Run m02_extract_signals.py first.")
        conn.close()
        return

    # Get v0 reference scores and images
    v0_scores = get_v0_scores_by_base(conn)
    v0_images = get_v0_images_by_base(conn)
    print(f"Loaded {len(v0_scores)} v0 reference scores")
    print(f"Loaded {len(v0_images)} v0 reference images")

    # Filter to only variations (skip v0)
    variation_data = [row for row in signals_data if row[8] != "v0_original"]
    print(f"Generating risk maps for {len(variation_data)} variations...")

    for row in tqdm(variation_data, desc="Creating risk maps"):
        (image_id, prompt, objects_json, gradcam_path, gaps_json,
         num_gaps, avg_score, image_path, variation, base_id, baseline_prompt) = row

        # Get v0 image path for this base_id
        v0_image_path = v0_images.get(base_id)
        if not v0_image_path:
            print(f"No v0 image found for base_id: {base_id}")
            continue

        # Load v0 image (LHS - baseline reference)
        full_v0_path = config.PROJECT_ROOT / v0_image_path
        if not full_v0_path.exists():
            print(f"v0 image not found: {full_v0_path}")
            continue

        # Load variation image (RHS - will have RED overlay)
        full_variation_path = config.PROJECT_ROOT / image_path
        if not full_variation_path.exists():
            print(f"Variation image not found: {full_variation_path}")
            continue

        try:
            v0_image = Image.open(full_v0_path).convert("RGB")
            variation_image = Image.open(full_variation_path).convert("RGB")

            # Parse JSON data
            objects = json.loads(objects_json) if objects_json else []
            gaps = json.loads(gaps_json) if gaps_json else []

            # Get v0 reference score for this base_id
            v0_score = v0_scores.get(base_id, avg_score or 0.5)

            # Create RED overlay on variation image (segments with gaps = different from baseline)
            variation_with_overlay, num_bad = create_risk_overlay(
                variation_image, objects, gaps, v0_score, config
            )

            # Create side-by-side: LHS=v0, RHS=variation with overlay
            combined = create_side_by_side(
                v0_image,
                variation_with_overlay,
                baseline_prompt or prompt,
                prompt,  # generation_prompt
                variation,
                num_bad
            )

            # Save
            output_filename = f"{image_id}_risk_map.png"
            output_path = config.OUTPUT_DIR / output_filename
            combined.save(str(output_path))

            # Save to DB
            save_risk_map_record(
                conn, image_id, str(output_path.relative_to(config.PROJECT_ROOT)),
                num_bad, avg_score or 0.0, v0_score
            )

        except Exception as e:
            print(f"Error processing {image_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    conn.close()
    print(f"Done! Risk maps saved to {config.OUTPUT_DIR}")
    print_summary(config)


def print_summary(config: Config) -> None:
    """Print summary of generated risk maps."""
    conn = sqlite3.connect(str(config.DB_PATH))
    cursor = conn.cursor()

    cursor.execute("""
        SELECT COUNT(*), AVG(num_bad_segments), AVG(avg_object_score)
        FROM risk_maps
    """)
    total, avg_bad, avg_score = cursor.fetchone()

    cursor.execute("""
        SELECT image_id, num_bad_segments, avg_object_score
        FROM risk_maps
        WHERE num_bad_segments > 0
        ORDER BY num_bad_segments DESC
        LIMIT 5
    """)
    top_bad = cursor.fetchall()

    conn.close()

    print(f"\n{'='*60}")
    print(f"RISK MAP SUMMARY")
    print(f"{'='*60}")
    print(f"Total risk maps: {total}")
    print(f"Avg bad segments per image: {avg_bad:.2f}" if avg_bad else "Avg bad: N/A")
    print(f"Avg object score: {avg_score:.3f}" if avg_score else "Avg score: N/A")

    if top_bad:
        print(f"\nTop 5 images with most bad segments:")
        for img_id, num_bad, score in top_bad:
            print(f"  {img_id}: {num_bad} bad segments, score={score:.3f}")


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    config = Config()
    generate_all_risk_maps(config)
