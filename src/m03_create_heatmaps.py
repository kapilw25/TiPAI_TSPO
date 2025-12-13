"""
m03_create_heatmaps.py
Create heatmap visualizations from patch scores.
Reads from: centralized.db (scores table)
Writes to: outputs/m03_heatmaps/

Note: CPU is fine for this (just visualization, no heavy compute)

Run:
    python src/m03_create_heatmaps.py
"""

import json
import sqlite3
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import zoom
from tqdm import tqdm


# ============================================================================
# Configuration
# ============================================================================
class Config:
    PROJECT_ROOT = Path(__file__).parent.parent
    DB_PATH = PROJECT_ROOT / "outputs" / "centralized.db"
    IMAGES_DIR = PROJECT_ROOT / "outputs" / "m01_images"
    OUTPUT_DIR = PROJECT_ROOT / "outputs" / "m03_heatmaps"
    PATCH_GRID = 7
    OVERLAY_ALPHA = 0.5


# ============================================================================
# Database
# ============================================================================
def get_all_scores(conn: sqlite3.Connection) -> list:
    """Get all scores from database."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT s.image_id, s.prompt_text, s.global_score, s.patch_scores, i.image_path
        FROM scores s
        JOIN images i ON s.image_id = i.image_id
    """)
    return cursor.fetchall()


def init_heatmaps_table(conn: sqlite3.Connection) -> None:
    """Create heatmaps table if not exists."""
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS heatmaps (
            image_id TEXT PRIMARY KEY,
            heatmap_path TEXT NOT NULL,
            min_patch_score REAL,
            max_patch_score REAL,
            mean_patch_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()


def save_heatmap_record(conn: sqlite3.Connection, image_id: str, heatmap_path: str,
                        patch_scores: list) -> None:
    """Save heatmap record to database."""
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO heatmaps (image_id, heatmap_path, min_patch_score, max_patch_score, mean_patch_score)
        VALUES (?, ?, ?, ?, ?)
    """, (image_id, heatmap_path, min(patch_scores), max(patch_scores), np.mean(patch_scores)))
    conn.commit()


# ============================================================================
# Heatmap Generation
# ============================================================================
def create_heatmap_overlay(image: Image.Image, patch_scores: list,
                          grid_size: int, alpha: float) -> Image.Image:
    """Create heatmap overlay on image."""
    # Reshape scores to grid
    scores_array = np.array(patch_scores).reshape(grid_size, grid_size)

    # Upscale to image size
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    scale_h, scale_w = h / grid_size, w / grid_size
    heatmap = zoom(scores_array, (scale_h, scale_w), order=1)

    # Create colormap (red=low/bad, green=high/good)
    cmap = plt.cm.RdYlGn  # Red-Yellow-Green
    norm = mcolors.Normalize(vmin=0.3, vmax=0.8)  # Adjust range for visibility
    heatmap_colored = cmap(norm(heatmap))[:, :, :3]  # RGB only
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    # Blend with original image
    blended = (1 - alpha) * img_array + alpha * heatmap_colored
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return Image.fromarray(blended)


def create_side_by_side(image: Image.Image, heatmap: Image.Image,
                       prompt: str, global_score: float) -> Image.Image:
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
    ax.text(combined.width / 2, 20, f'"{prompt}" (score: {global_score:.3f})',
            ha='center', va='center', fontsize=10, fontweight='bold')

    # Labels
    ax.text(w / 2, 45, 'Original', ha='center', va='center', fontsize=9)
    ax.text(w + 20 + w / 2, 45, 'Faithfulness Heatmap', ha='center', va='center', fontsize=9)

    # Convert to PIL
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    result = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    plt.close(fig)

    return result


def generate_all_heatmaps(config: Config) -> None:
    """Generate heatmaps for all scored images."""
    # Ensure dataset is available (download from HF if not local)
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

    print(f"Generating heatmaps for {len(scores_data)} images...")

    for image_id, prompt_text, global_score, patch_scores_json, image_path in tqdm(scores_data, desc="Heatmaps"):
        patch_scores = json.loads(patch_scores_json)
        full_image_path = config.PROJECT_ROOT / image_path

        if not full_image_path.exists():
            print(f"Image not found: {full_image_path}")
            continue

        image = Image.open(full_image_path).convert("RGB")

        # Create heatmap overlay
        heatmap = create_heatmap_overlay(image, patch_scores, config.PATCH_GRID, config.OVERLAY_ALPHA)

        # Create side-by-side
        combined = create_side_by_side(image, heatmap, prompt_text, global_score)

        # Save
        heatmap_filename = f"{image_id}_heatmap.png"
        heatmap_path = config.OUTPUT_DIR / heatmap_filename
        combined.save(str(heatmap_path))

        save_heatmap_record(conn, image_id, str(heatmap_path.relative_to(config.PROJECT_ROOT)), patch_scores)

    conn.close()
    print(f"Done! Heatmaps saved to {config.OUTPUT_DIR}")


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    config = Config()
    generate_all_heatmaps(config)
