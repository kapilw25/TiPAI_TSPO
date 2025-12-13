"""
m04_create_pairs.py
Create chosen/rejected pairs from variation-based images.

Pairing Logic:
  - v0_original (chosen) vs v1_attribute (rejected) → attribute failure
  - v0_original (chosen) vs v2_object (rejected) → object failure
  - v0_original (chosen) vs v3_spatial (rejected) → spatial failure

Output: 20 base prompts × 3 pairs = 60 pairs

Run:
    python src/m04_create_pairs.py
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
    IMAGES_DIR = PROJECT_ROOT / "outputs" / "m01_images"
    HEATMAPS_DIR = PROJECT_ROOT / "outputs" / "m03_heatmaps"
    OUTPUT_DIR = PROJECT_ROOT / "outputs" / "m04_pairs"


# ============================================================================
# Database
# ============================================================================
def init_pairs_table(conn: sqlite3.Connection) -> None:
    """Create pairs table if not exists."""
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pairs (
            pair_id TEXT PRIMARY KEY,
            base_id TEXT NOT NULL,
            failure_type TEXT NOT NULL,
            chosen_prompt TEXT NOT NULL,
            rejected_prompt TEXT NOT NULL,
            chosen_image_id TEXT NOT NULL,
            rejected_image_id TEXT NOT NULL,
            chosen_score REAL NOT NULL,
            rejected_score REAL NOT NULL,
            score_gap REAL NOT NULL,
            comparison_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()


def get_scores_by_base_id(conn: sqlite3.Connection) -> dict:
    """Get all scores grouped by base_id with variation info."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT i.base_id, i.variation, i.prompt_text, s.image_id, s.global_score, i.image_path
        FROM scores s
        JOIN images i ON s.image_id = i.image_id
        ORDER BY i.base_id, i.variation
    """)

    results = {}
    for base_id, variation, prompt_text, image_id, score, image_path in cursor.fetchall():
        if base_id not in results:
            results[base_id] = {}
        results[base_id][variation] = {
            "prompt_text": prompt_text,
            "image_id": image_id,
            "score": score,
            "path": image_path
        }
    return results


def save_pair(conn: sqlite3.Connection, pair_id: str, base_id: str, failure_type: str,
              chosen_prompt: str, rejected_prompt: str,
              chosen_id: str, rejected_id: str, chosen_score: float,
              rejected_score: float, comparison_path: str) -> None:
    """Save pair to database."""
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO pairs
        (pair_id, base_id, failure_type, chosen_prompt, rejected_prompt,
         chosen_image_id, rejected_image_id, chosen_score, rejected_score, score_gap, comparison_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (pair_id, base_id, failure_type, chosen_prompt, rejected_prompt,
          chosen_id, rejected_id, chosen_score, rejected_score,
          chosen_score - rejected_score, comparison_path))
    conn.commit()


# ============================================================================
# Pair Creation
# ============================================================================
def create_comparison_image(chosen_path: Path, rejected_path: Path,
                           chosen_heatmap: Path, rejected_heatmap: Path,
                           chosen_prompt: str, rejected_prompt: str,
                           failure_type: str,
                           chosen_score: float, rejected_score: float) -> Image.Image:
    """Create 2x2 comparison grid with failure type info."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Load images
    chosen_img = Image.open(chosen_path) if chosen_path.exists() else None
    rejected_img = Image.open(rejected_path) if rejected_path.exists() else None
    chosen_heat = Image.open(chosen_heatmap) if chosen_heatmap.exists() else None
    rejected_heat = Image.open(rejected_heatmap) if rejected_heatmap.exists() else None

    # Row 1: Original images
    if chosen_img:
        axes[0, 0].imshow(chosen_img)
    axes[0, 0].set_title(f'CHOSEN (score: {chosen_score:.3f})\n"{chosen_prompt[:50]}..."',
                         fontsize=10, color='green', fontweight='bold')
    axes[0, 0].axis('off')

    if rejected_img:
        axes[0, 1].imshow(rejected_img)
    axes[0, 1].set_title(f'REJECTED (score: {rejected_score:.3f})\n"{rejected_prompt[:50]}..."',
                         fontsize=10, color='red', fontweight='bold')
    axes[0, 1].axis('off')

    # Row 2: Heatmaps
    if chosen_heat:
        axes[1, 0].imshow(chosen_heat)
    axes[1, 0].set_title('Chosen Heatmap', fontsize=10)
    axes[1, 0].axis('off')

    if rejected_heat:
        axes[1, 1].imshow(rejected_heat)
    axes[1, 1].set_title('Rejected Heatmap', fontsize=10)
    axes[1, 1].axis('off')

    # Main title with failure type
    failure_labels = {
        "attribute": "ATTRIBUTE FAILURE (color/size/material)",
        "object": "OBJECT FAILURE (wrong object)",
        "spatial": "SPATIAL/COUNT FAILURE"
    }
    fig.suptitle(f'{failure_labels.get(failure_type, failure_type)}\nScore Gap: {chosen_score - rejected_score:.3f}',
                 fontsize=12, fontweight='bold')

    plt.tight_layout()

    # Convert to PIL
    fig.canvas.draw()
    result = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    plt.close(fig)

    return result


def create_all_pairs(config: Config) -> None:
    """Create variation-based pairs: v0 (chosen) vs v1/v2/v3 (rejected)."""
    # Ensure dataset is available (download from HF if not local)
    from utils.hf_utils import ensure_dataset_available
    if not ensure_dataset_available(config.IMAGES_DIR):
        print("ERROR: Could not get dataset locally or from HuggingFace.")
        return

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(config.DB_PATH))
    init_pairs_table(conn)

    scores_by_base = get_scores_by_base_id(conn)
    if not scores_by_base:
        print("No scores found. Run m02_clip_scoring.py first.")
        conn.close()
        return

    # Variation pairs: v0 is always chosen
    variation_pairs = [
        ("v1_attribute", "attribute"),
        ("v2_object", "object"),
        ("v3_spatial", "spatial"),
    ]

    total_pairs = len(scores_by_base) * len(variation_pairs)
    print(f"Creating {total_pairs} pairs for {len(scores_by_base)} base prompts...")

    for base_id, variations in tqdm(scores_by_base.items(), desc="Creating pairs"):
        # v0_original is always the chosen image
        if "v0_original" not in variations:
            print(f"Skipping {base_id}: no v0_original found")
            continue

        chosen = variations["v0_original"]

        for var_name, failure_type in variation_pairs:
            if var_name not in variations:
                continue

            rejected = variations[var_name]
            pair_id = f"pair_{base_id}_{failure_type}"

            # Paths
            chosen_img_path = config.PROJECT_ROOT / chosen["path"]
            rejected_img_path = config.PROJECT_ROOT / rejected["path"]
            chosen_heatmap = config.HEATMAPS_DIR / f"{chosen['image_id']}_heatmap.png"
            rejected_heatmap = config.HEATMAPS_DIR / f"{rejected['image_id']}_heatmap.png"

            # Create comparison
            comparison = create_comparison_image(
                chosen_img_path, rejected_img_path,
                chosen_heatmap, rejected_heatmap,
                chosen["prompt_text"], rejected["prompt_text"],
                failure_type,
                chosen["score"], rejected["score"]
            )

            comparison_filename = f"{pair_id}_comparison.png"
            comparison_path = config.OUTPUT_DIR / comparison_filename
            comparison.save(str(comparison_path))

            # Save to DB
            save_pair(conn, pair_id, base_id, failure_type,
                     chosen["prompt_text"], rejected["prompt_text"],
                     chosen["image_id"], rejected["image_id"],
                     chosen["score"], rejected["score"],
                     str(comparison_path.relative_to(config.PROJECT_ROOT)))

    conn.close()
    print(f"Done! Pairs saved to {config.OUTPUT_DIR}")


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    config = Config()
    create_all_pairs(config)
