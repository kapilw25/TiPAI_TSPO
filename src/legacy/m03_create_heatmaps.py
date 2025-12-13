"""
m03_create_heatmaps.py
Create heatmap visualizations using SAM segmentation + CLIP scoring.

SAM identifies objects, CLIP scores each object against baseline prompt.
Objects that don't match the baseline prompt get red overlay.

REQUIRES: CUDA GPU (SAM + CLIP)

Run:
    python -u src/m03_create_heatmaps.py 2>&1 | tee logs/m03_$(date +%Y%m%d_%H%M%S).log
"""

import json
import sqlite3
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import open_clip

# SAM imports
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


# ============================================================================
# Configuration
# ============================================================================
class Config:
    PROJECT_ROOT = Path(__file__).parent.parent
    DB_PATH = PROJECT_ROOT / "outputs" / "centralized.db"
    IMAGES_DIR = PROJECT_ROOT / "outputs" / "m01_images"
    OUTPUT_DIR = PROJECT_ROOT / "outputs" / "m03_heatmaps"

    # SAM config
    SAM_CHECKPOINT = PROJECT_ROOT / "models" / "sam_vit_b_01ec64.pth"
    SAM_MODEL_TYPE = "vit_b"

    # Overlay config
    OVERLAY_ALPHA = 0.7
    SCORE_THRESHOLD_RATIO = 0.92  # Mark object as "bad" if score < v0_score * this ratio
    MIN_OBJECT_AREA_RATIO = 0.05  # Only consider segments > 5% of image (main objects)
    MAX_OBJECT_AREA_RATIO = 0.50  # Only consider segments < 50% of image (not background)
    MAX_SEGMENTS_TO_CHECK = 3     # Only check top N largest segments (main subjects)


# ============================================================================
# Global Models (loaded once)
# ============================================================================
SAM_MODEL = None
MASK_GENERATOR = None
CLIP_MODEL = None
CLIP_PREPROCESS = None
CLIP_TOKENIZER = None


def load_models(config: Config):
    """Load SAM and CLIP models on GPU."""
    global SAM_MODEL, MASK_GENERATOR, CLIP_MODEL, CLIP_PREPROCESS, CLIP_TOKENIZER

    assert torch.cuda.is_available(), "CUDA required!"

    # Load SAM
    print(f"Loading SAM ({config.SAM_MODEL_TYPE})...")
    SAM_MODEL = sam_model_registry[config.SAM_MODEL_TYPE](checkpoint=str(config.SAM_CHECKPOINT))
    SAM_MODEL = SAM_MODEL.to("cuda")
    MASK_GENERATOR = SamAutomaticMaskGenerator(
        SAM_MODEL,
        points_per_side=16,  # Fewer points for faster inference
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        min_mask_region_area=1000,  # Ignore tiny regions
    )

    # Load CLIP
    print("Loading CLIP (ViT-L-14)...")
    CLIP_MODEL, _, CLIP_PREPROCESS = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    CLIP_MODEL = CLIP_MODEL.to("cuda").eval()
    CLIP_TOKENIZER = open_clip.get_tokenizer("ViT-L-14")

    print("Models loaded!")


# ============================================================================
# Database
# ============================================================================
def get_all_scores(conn: sqlite3.Connection) -> list:
    """Get all scores from database."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT s.image_id, s.baseline_prompt, s.baseline_score, s.baseline_patch_scores, i.image_path, i.base_id, i.variation
        FROM scores s
        JOIN images i ON s.image_id = i.image_id
    """)
    return cursor.fetchall()


def get_v0_scores_by_base(conn: sqlite3.Connection) -> dict:
    """Get v0_original global scores for each base_id as reference threshold."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT i.base_id, s.baseline_score
        FROM scores s
        JOIN images i ON s.image_id = i.image_id
        WHERE i.variation = 'v0_original'
    """)
    return {row[0]: row[1] for row in cursor.fetchall()}


def init_heatmaps_table(conn: sqlite3.Connection) -> None:
    """Create heatmaps table (drop and recreate to update schema)."""
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS heatmaps")
    cursor.execute("""
        CREATE TABLE heatmaps (
            image_id TEXT PRIMARY KEY,
            heatmap_path TEXT NOT NULL,
            min_patch_score REAL,
            max_patch_score REAL,
            mean_patch_score REAL,
            num_bad_segments INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()


def save_heatmap_record(conn: sqlite3.Connection, image_id: str, heatmap_path: str,
                        patch_scores: list, num_bad_segments: int = 0) -> None:
    """Save heatmap record to database."""
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO heatmaps (image_id, heatmap_path, min_patch_score, max_patch_score, mean_patch_score, num_bad_segments)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (image_id, heatmap_path, min(patch_scores), max(patch_scores), np.mean(patch_scores), num_bad_segments))
    conn.commit()


# ============================================================================
# SAM + CLIP Scoring
# ============================================================================

# Background words to filter out (segments matching these are ignored)
BACKGROUND_WORDS = {
    "sky", "ground", "floor", "wall", "ceiling", "background", "grass", "sand",
    "water", "air", "space", "room", "area", "scene", "land", "surface", "sunset",
    "beach", "park", "street", "garden", "kitchen", "bathroom", "bedroom", "gym",
    "cafe", "library", "farm", "studio", "forest", "mountain", "ocean", "sea"
}


def extract_object_nouns(prompt: str) -> list:
    """Extract likely object nouns from prompt for CLIP-filtering.

    Simple heuristic: words after 'a/an/the' or colors are likely objects.
    """
    # Common colors that precede objects
    colors = {"red", "blue", "green", "yellow", "orange", "purple", "pink", "black",
              "white", "brown", "gray", "silver", "gold", "golden", "bronze"}

    words = prompt.lower().replace(",", " ").replace(".", " ").split()
    nouns = []

    for i, word in enumerate(words):
        # Skip background words
        if word in BACKGROUND_WORDS:
            continue
        # Skip colors and small words
        if word in colors or len(word) <= 2:
            continue
        # Word after 'a', 'an', 'the', or a color is likely a noun
        if i > 0 and words[i-1] in {"a", "an", "the"} | colors:
            nouns.append(word)

    # Deduplicate while preserving order
    seen = set()
    unique_nouns = []
    for n in nouns:
        if n not in seen:
            seen.add(n)
            unique_nouns.append(n)

    return unique_nouns


def get_clip_score(image: Image.Image, prompt: str) -> float:
    """Compute CLIP score for an image region against prompt."""
    # Preprocess image
    img_tensor = CLIP_PREPROCESS(image).unsqueeze(0).to("cuda")

    # Get text embedding
    tokens = CLIP_TOKENIZER([prompt]).to("cuda")

    with torch.no_grad():
        img_emb = CLIP_MODEL.encode_image(img_tensor)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

        text_emb = CLIP_MODEL.encode_text(tokens)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

        similarity = (text_emb @ img_emb.T).item()

    # Scale from [-1, 1] to [0, 1]
    return (similarity + 1) / 2


def segment_is_object(segment_img: Image.Image, object_nouns: list,
                      min_match_score: float = 0.54) -> tuple:
    """Check if segment matches any object noun from prompt (filters out background).

    Returns (is_object, best_matching_noun, best_score)
    """
    if not object_nouns:
        return True, None, 0.0  # If no nouns extracted, don't filter

    best_score = 0.0
    best_noun = None

    for noun in object_nouns:
        # Score segment against individual object noun
        score = get_clip_score(segment_img, f"a photo of a {noun}")
        if score > best_score:
            best_score = score
            best_noun = noun

    # Segment is considered an "object" if it matches any noun above threshold
    return best_score >= min_match_score, best_noun, best_score


def extract_masked_region(image: np.ndarray, mask: np.ndarray) -> Image.Image:
    """Extract the masked region as a cropped image for CLIP scoring."""
    # Get bounding box of mask
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # Add padding
    pad = 10
    y_min = max(0, y_min - pad)
    y_max = min(image.shape[0], y_max + pad)
    x_min = max(0, x_min - pad)
    x_max = min(image.shape[1], x_max + pad)

    # Crop region
    cropped = image[y_min:y_max, x_min:x_max].copy()

    return Image.fromarray(cropped)


def create_sam_heatmap(image: Image.Image, baseline_prompt: str,
                       v0_score: float, config: Config) -> tuple:
    """
    Use SAM to segment image, CLIP to score each segment.
    CLIP-filtering: Only consider segments that match objects in the prompt.
    Returns (heatmap_image, num_bad_segments)
    """
    img_array = np.array(image)

    # Generate SAM masks
    masks = MASK_GENERATOR.generate(img_array)

    if not masks:
        # No masks found, return original
        return image, 0

    # Extract object nouns from prompt for CLIP-filtering
    object_nouns = extract_object_nouns(baseline_prompt)

    # Sort masks by area (largest first)
    masks = sorted(masks, key=lambda x: x['area'], reverse=True)

    # Score threshold: object is "bad" if score < v0_score * ratio
    threshold = v0_score * config.SCORE_THRESHOLD_RATIO

    # Track which pixels should be highlighted
    bad_mask = np.zeros(img_array.shape[:2], dtype=bool)
    num_bad = 0

    total_pixels = img_array.shape[0] * img_array.shape[1]

    # Filter to only medium-sized segments (main objects)
    valid_masks = []
    for mask_data in masks:
        area_ratio = mask_data['area'] / total_pixels
        if config.MIN_OBJECT_AREA_RATIO <= area_ratio <= config.MAX_OBJECT_AREA_RATIO:
            valid_masks.append(mask_data)

    # Check segments (increased limit since we filter by object matching)
    segments_checked = 0
    max_to_check = config.MAX_SEGMENTS_TO_CHECK * 2  # Check more, filter later

    for mask_data in valid_masks:
        if segments_checked >= max_to_check:
            break

        mask = mask_data['segmentation']

        try:
            region_img = extract_masked_region(img_array, mask)

            # CLIP-filter: Skip segments that don't match any object in prompt
            is_object, matched_noun, object_score = segment_is_object(region_img, object_nouns)
            if not is_object:
                continue  # Skip background segments

            segments_checked += 1

            # Score against full baseline prompt
            score = get_clip_score(region_img, baseline_prompt)

            # If score below threshold, mark as bad
            if score < threshold:
                bad_mask |= mask
                num_bad += 1

        except Exception:
            continue

    # Apply red overlay only on bad segments
    result = img_array.astype(np.float32)
    if bad_mask.any():
        alpha = config.OVERLAY_ALPHA
        red_overlay = np.array([255, 0, 0], dtype=np.float32)

        # Apply overlay where bad_mask is True
        result[bad_mask] = (1 - alpha) * result[bad_mask] + alpha * red_overlay

    return Image.fromarray(result.astype(np.uint8)), num_bad


# ============================================================================
# Visualization
# ============================================================================
def create_side_by_side(image: Image.Image, heatmap: Image.Image,
                       prompt: str, global_score: float, num_bad: int = 0) -> Image.Image:
    """Create side-by-side comparison with labels."""
    w, h = image.size
    combined = Image.new('RGB', (w * 2 + 20, h + 60), color=(255, 255, 255))

    # Paste images
    combined.paste(image, (0, 50))
    combined.paste(heatmap, (w + 20, 50))

    # Add text using matplotlib
    fig, ax = plt.subplots(figsize=(combined.width / 100, combined.height / 100), dpi=100)
    ax.imshow(combined)
    ax.axis('off')

    # Title
    bad_text = f" | {num_bad} bad segments" if num_bad > 0 else ""
    ax.text(combined.width / 2, 20, f'"{prompt}" (score: {global_score:.3f}{bad_text})',
            ha='center', va='center', fontsize=10, fontweight='bold')

    # Labels
    ax.text(w / 2, 45, 'Original', ha='center', va='center', fontsize=9)
    ax.text(w + 20 + w / 2, 45, 'SAM + CLIP Heatmap', ha='center', va='center', fontsize=9)

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
def generate_all_heatmaps(config: Config) -> None:
    """Generate SAM-based heatmaps for all scored images."""
    # Ensure dataset is available
    from utils.hf_utils import ensure_dataset_available
    if not ensure_dataset_available(config.IMAGES_DIR):
        print("ERROR: Could not get dataset locally or from HuggingFace.")
        return

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(config.DB_PATH))
    init_heatmaps_table(conn)

    scores_data = get_all_scores(conn)
    if not scores_data:
        print("No scores found. Run m02_clip_scoring.py first.")
        conn.close()
        return

    # Load models
    load_models(config)

    # Get v0 reference scores
    v0_scores = get_v0_scores_by_base(conn)
    print(f"Loaded {len(v0_scores)} v0 reference scores")
    print(f"Generating SAM+CLIP heatmaps for {len(scores_data)} images...")

    for image_id, baseline_prompt, global_score, patch_scores_json, image_path, base_id, variation in tqdm(scores_data, desc="Heatmaps"):
        patch_scores = json.loads(patch_scores_json)
        full_image_path = config.PROJECT_ROOT / image_path

        if not full_image_path.exists():
            print(f"Image not found: {full_image_path}")
            continue

        image = Image.open(full_image_path).convert("RGB")

        # For v0: no overlay (reference image)
        # For variations: SAM segment + CLIP score each segment
        if variation == "v0_original":
            heatmap = image  # No overlay for reference
            num_bad = 0
        else:
            v0_score = v0_scores.get(base_id, global_score)
            heatmap, num_bad = create_sam_heatmap(image, baseline_prompt, v0_score, config)

        # Create side-by-side
        combined = create_side_by_side(image, heatmap, baseline_prompt, global_score, num_bad)

        # Save
        heatmap_filename = f"{image_id}_heatmap.png"
        heatmap_path = config.OUTPUT_DIR / heatmap_filename
        combined.save(str(heatmap_path))

        save_heatmap_record(conn, image_id, str(heatmap_path.relative_to(config.PROJECT_ROOT)), patch_scores, num_bad)

    conn.close()
    print(f"Done! Heatmaps saved to {config.OUTPUT_DIR}")


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    config = Config()
    generate_all_heatmaps(config)
