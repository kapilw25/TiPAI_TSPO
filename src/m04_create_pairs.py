"""
m04_create_pairs.py
Create chosen/rejected pairs from variation-based images.

Uses signals from m02 and risk_maps from m03.

Pairing Logic:
  - v0_original (chosen) vs v1_attribute (rejected) → attribute failure
  - v0_original (chosen) vs v2_object (rejected) → object failure
  - v0_original (chosen) vs v3_spatial (rejected) → spatial failure

Output: 20 base prompts × 3 pairs = 60 pairs

Run:
    python -u src/m04_create_pairs.py 2>&1 | tee logs/m04_$(date +%Y%m%d_%H%M%S).log
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
    RISK_MAPS_DIR = PROJECT_ROOT / "outputs" / "m03_risk_maps"
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


def get_signals_by_base_id(conn: sqlite3.Connection) -> dict:
    """Get all signals grouped by base_id with variation info."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT i.base_id, i.variation, i.generation_prompt, s.image_id,
               s.avg_object_score, i.image_path, s.num_gaps
        FROM signals s
        JOIN images i ON s.image_id = i.image_id
        ORDER BY i.base_id, i.variation
    """)

    results = {}
    for base_id, variation, prompt_text, image_id, score, image_path, num_gaps in cursor.fetchall():
        if base_id not in results:
            results[base_id] = {}
        results[base_id][variation] = {
            "prompt_text": prompt_text,
            "image_id": image_id,
            "score": score or 0.0,
            "path": image_path,
            "num_gaps": num_gaps or 0
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
                           chosen_risk_map: Path, rejected_risk_map: Path,
                           chosen_prompt: str, rejected_prompt: str,
                           failure_type: str,
                           chosen_score: float, rejected_score: float,
                           rejected_gaps: int = 0) -> Image.Image:
    """Create 2x2 comparison grid with failure type info."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Load images
    chosen_img = Image.open(chosen_path) if chosen_path.exists() else None
    rejected_img = Image.open(rejected_path) if rejected_path.exists() else None
    chosen_risk = Image.open(chosen_risk_map) if chosen_risk_map.exists() else None
    rejected_risk = Image.open(rejected_risk_map) if rejected_risk_map.exists() else None

    # Row 1: Original images
    if chosen_img:
        axes[0, 0].imshow(chosen_img)
    axes[0, 0].set_title(f'CHOSEN (score: {chosen_score:.3f})\n"{chosen_prompt[:50]}..."',
                         fontsize=10, color='green', fontweight='bold')
    axes[0, 0].axis('off')

    if rejected_img:
        axes[0, 1].imshow(rejected_img)
    axes[0, 1].set_title(f'REJECTED (score: {rejected_score:.3f}, gaps: {rejected_gaps})\n"{rejected_prompt[:50]}..."',
                         fontsize=10, color='red', fontweight='bold')
    axes[0, 1].axis('off')

    # Row 2: Risk Maps
    if chosen_risk:
        axes[1, 0].imshow(chosen_risk)
    axes[1, 0].set_title('Chosen Risk Map', fontsize=10)
    axes[1, 0].axis('off')

    if rejected_risk:
        axes[1, 1].imshow(rejected_risk)
    axes[1, 1].set_title('Rejected Risk Map (RED = issues)', fontsize=10)
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
    buf = fig.canvas.buffer_rgba()
    result = Image.frombuffer('RGBA', fig.canvas.get_width_height(), buf, 'raw', 'RGBA', 0, 1)
    result = result.convert('RGB')
    plt.close(fig)

    return result


def create_all_pairs(config: Config) -> None:
    """Create variation-based pairs: v0 (chosen) vs v1/v2/v3 (rejected)."""
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(config.DB_PATH))

    # Check if signals table exists
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='signals'")
    if not cursor.fetchone():
        print("ERROR: No signals table found. Run m02_extract_signals.py first.")
        conn.close()
        return

    init_pairs_table(conn)

    signals_by_base = get_signals_by_base_id(conn)
    if not signals_by_base:
        print("No signals found. Run m02_extract_signals.py first.")
        conn.close()
        return

    # Variation pairs: v0 is always chosen
    variation_pairs = [
        ("v1_attribute", "attribute"),
        ("v2_object", "object"),
        ("v3_spatial", "spatial"),
    ]

    total_pairs = len(signals_by_base) * len(variation_pairs)
    print(f"Creating {total_pairs} pairs for {len(signals_by_base)} base prompts...")

    for base_id, variations in tqdm(signals_by_base.items(), desc="Creating pairs"):
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
            chosen_risk_map = config.RISK_MAPS_DIR / f"{chosen['image_id']}_risk_map.png"
            rejected_risk_map = config.RISK_MAPS_DIR / f"{rejected['image_id']}_risk_map.png"

            # Create comparison
            comparison = create_comparison_image(
                chosen_img_path, rejected_img_path,
                chosen_risk_map, rejected_risk_map,
                chosen["prompt_text"], rejected["prompt_text"],
                failure_type,
                chosen["score"], rejected["score"],
                rejected["num_gaps"]
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
