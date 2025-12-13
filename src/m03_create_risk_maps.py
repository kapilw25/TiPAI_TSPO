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
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================================================================
# Configuration
# ============================================================================
class Config:
    PROJECT_ROOT = Path(__file__).parent.parent
    DB_PATH = PROJECT_ROOT / "outputs" / "centralized.db"
    # Note: image_path comes from DB (model-specific dirs handled by m01)
    OUTPUT_DIR = PROJECT_ROOT / "outputs" / "m03_risk_maps"

    # Overlay config
    OVERLAY_ALPHA = 0.7
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
def create_risk_overlay(
    image: Image.Image,
    objects: list,
    gaps: list,
    v0_score: float,
    config: Config
) -> tuple[Image.Image, int]:
    """
    Create RED-only overlay on bad segments.

    Bad segments:
    1. Segments with gaps (wrong_color, missing, wrong_object)
    2. Segments with prompt_score < v0_score * SCORE_THRESHOLD_RATIO

    Returns: (overlay_image, num_bad_segments)
    """
    img_array = np.array(image).astype(np.float32)
    h, w = img_array.shape[:2]

    # Track bad pixels
    bad_mask = np.zeros((h, w), dtype=bool)
    num_bad = 0

    # Score threshold
    threshold = v0_score * config.SCORE_THRESHOLD_RATIO if v0_score else 0.5

    # Build set of segment_ids with gaps
    gap_segment_ids = set()
    for gap in gaps:
        seg_id = gap.get('segment_id', -1)
        if seg_id >= 0:
            gap_segment_ids.add(seg_id)

    # Check each detected object
    for obj in objects:
        segment_id = obj.get('segment_id', -1)
        prompt_score = obj.get('prompt_score', obj.get('label_score', 0.5))
        bbox = obj.get('bbox')

        if bbox is None:
            continue

        x, y, bw, bh = bbox

        # Clamp bbox to image bounds
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(w, int(x + bw)), min(h, int(y + bh))

        if x2 <= x1 or y2 <= y1:
            continue

        # Mark as bad if:
        # 1. Has a gap (wrong color, missing, wrong object)
        # 2. Score below threshold
        is_bad = (segment_id in gap_segment_ids) or (prompt_score < threshold)

        if is_bad:
            bad_mask[y1:y2, x1:x2] = True
            num_bad += 1

    # Apply RED overlay only on bad segments
    result = img_array.copy()
    if bad_mask.any():
        alpha = config.OVERLAY_ALPHA
        red_overlay = np.array([255, 0, 0], dtype=np.float32)
        result[bad_mask] = (1 - alpha) * result[bad_mask] + alpha * red_overlay

    return Image.fromarray(result.astype(np.uint8)), num_bad


def create_side_by_side(
    image: Image.Image,
    risk_map: Image.Image,
    baseline_prompt: str,
    variation: str,
    num_bad: int,
    avg_score: float
) -> Image.Image:
    """Create side-by-side comparison with labels."""
    w, h = image.size
    combined = Image.new('RGB', (w * 2 + 20, h + 60), color=(255, 255, 255))

    # Paste images
    combined.paste(image, (0, 50))
    combined.paste(risk_map, (w + 20, 50))

    # Add text using matplotlib
    fig, ax = plt.subplots(figsize=(combined.width / 100, combined.height / 100), dpi=100)
    ax.imshow(combined)
    ax.axis('off')

    # Title
    bad_text = f" | {num_bad} bad segments" if num_bad > 0 else " | No issues"
    prompt_display = baseline_prompt[:60] + '...' if len(baseline_prompt) > 60 else baseline_prompt
    ax.text(combined.width / 2, 20, f'"{prompt_display}" (score: {avg_score:.3f}{bad_text})',
            ha='center', va='center', fontsize=10, fontweight='bold')

    # Labels
    ax.text(w / 2, 45, f'Original ({variation})', ha='center', va='center', fontsize=9)
    ax.text(w + 20 + w / 2, 45, 'Risk Map', ha='center', va='center', fontsize=9)

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
    """Generate risk maps for all images with signals data."""
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

    # Get v0 reference scores
    v0_scores = get_v0_scores_by_base(conn)
    print(f"Loaded {len(v0_scores)} v0 reference scores")
    print(f"Generating risk maps for {len(signals_data)} images...")

    for row in tqdm(signals_data, desc="Creating risk maps"):
        (image_id, prompt, objects_json, gradcam_path, gaps_json,
         num_gaps, avg_score, image_path, variation, base_id, baseline_prompt) = row

        # Load image
        full_image_path = config.PROJECT_ROOT / image_path
        if not full_image_path.exists():
            print(f"Image not found: {full_image_path}")
            continue

        try:
            image = Image.open(full_image_path).convert("RGB")

            # Parse JSON data
            objects = json.loads(objects_json) if objects_json else []
            gaps = json.loads(gaps_json) if gaps_json else []

            # Get v0 reference score for this base_id
            v0_score = v0_scores.get(base_id, avg_score or 0.5)

            # For v0_original: no overlay (reference image)
            # For variations: create risk overlay
            if variation == "v0_original":
                risk_map = image  # No overlay for reference
                num_bad = 0
            else:
                risk_map, num_bad = create_risk_overlay(
                    image, objects, gaps, v0_score, config
                )

            # Create side-by-side visualization
            combined = create_side_by_side(
                image, risk_map, baseline_prompt or prompt,
                variation, num_bad, avg_score or 0.0
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
