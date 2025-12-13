"""
hf_utils.py
HuggingFace utilities for TiPAI-TSPO dataset.
- Push dataset to HuggingFace
- Download dataset from HuggingFace
- Generate dataset README card
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime

from datasets import Dataset, load_dataset, Image as HFImage
from huggingface_hub import HfApi


# ============================================================================
# Configuration
# ============================================================================
HF_REPO = "kapilw25/TiPAI-POC-Faithfulness"


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent.parent


def load_hf_token() -> str:
    """Load HuggingFace token from .env file."""
    env_path = get_project_root() / ".env"
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                if line.startswith('HF_TOKEN='):
                    return line.strip().split('=', 1)[1]
    raise ValueError("HF_TOKEN not found in .env file. Add HF_TOKEN=your_token to .env")


# ============================================================================
# Dataset Statistics (Dynamic)
# ============================================================================
def get_dataset_stats(db_path: Path) -> dict:
    """Get dynamic statistics from database."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    stats = {}

    # Image counts
    cursor.execute("SELECT COUNT(*) FROM images")
    stats["total_images"] = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT prompt_id) FROM images")
    stats["total_prompts"] = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT prompt_category) FROM images")
    stats["total_categories"] = cursor.fetchone()[0]

    # Category breakdown
    cursor.execute("""
        SELECT prompt_category, COUNT(*) as count
        FROM images GROUP BY prompt_category ORDER BY count DESC
    """)
    stats["categories"] = {row[0]: row[1] for row in cursor.fetchall()}

    # Score stats
    cursor.execute("SELECT COUNT(*) FROM scores")
    stats["scored_images"] = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(global_score), MIN(global_score), MAX(global_score) FROM scores")
    row = cursor.fetchone()
    if row[0]:
        stats["avg_score"] = round(row[0], 4)
        stats["min_score"] = round(row[1], 4)
        stats["max_score"] = round(row[2], 4)

    # Pair stats
    cursor.execute("SELECT COUNT(*) FROM pairs")
    stats["total_pairs"] = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(score_gap) FROM pairs")
    row = cursor.fetchone()
    if row[0]:
        stats["avg_score_gap"] = round(row[0], 4)

    conn.close()
    return stats


# ============================================================================
# README Card Generator
# ============================================================================
def generate_readme_card(stats: dict) -> str:
    """Generate HuggingFace dataset README card."""
    categories_md = "\n".join([f"| {cat} | {count} |" for cat, count in stats.get("categories", {}).items()])

    readme = f"""---
license: mit
task_categories:
  - text-to-image
  - image-classification
language:
  - en
tags:
  - text-to-image
  - faithfulness
  - CLIP
  - patch-level
  - preference-learning
  - TiPAI
  - TSPO
  - TIFA
size_categories:
  - n<1K
---

# TiPAI-POC: Patch-Level Faithfulness Dataset

## Dataset Description

This dataset contains text-to-image generation samples with **patch-level faithfulness scores** and **systematic failure variations**.

Each base prompt has 4 variations:
- **v0_original**: Correct prompt (chosen baseline)
- **v1_attribute**: Wrong color/size/material
- **v2_object**: Swapped/wrong main object
- **v3_spatial**: Wrong spatial relation or count

**Purpose**: Training patch-level preference models for text-to-image alignment (TiPAI-TSPO research).

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Images | {stats.get('total_images', 'N/A')} |
| Base Prompts | {stats.get('total_prompts', 'N/A')} |
| Categories | {stats.get('total_categories', 'N/A')} |
| Scored Images | {stats.get('scored_images', 'N/A')} |
| Chosen/Rejected Pairs | {stats.get('total_pairs', 'N/A')} |

### Score Distribution

| Metric | Value |
|--------|-------|
| Average Score | {stats.get('avg_score', 'N/A')} |
| Min Score | {stats.get('min_score', 'N/A')} |
| Max Score | {stats.get('max_score', 'N/A')} |
| Avg Pair Gap | {stats.get('avg_score_gap', 'N/A')} |

### Categories

| Category | Images |
|----------|--------|
{categories_md}

## Variation Types (Based on TIFA Benchmark)

| Variation | Description | Example |
|-----------|-------------|---------|
| v0_original | Correct prompt | "a red car on the beach" |
| v1_attribute | Wrong attribute | "a blue car on the beach" |
| v2_object | Wrong object | "a red bicycle on the beach" |
| v3_spatial | Wrong spatial/count | "a red car in the ocean" |

## Dataset Structure

```python
{{
    "image_id": "attr_01_v0_seed42",
    "prompt_id": "attr_01_v0",
    "base_id": "attr_01",
    "variation": "v0_original",
    "prompt": "a shiny red sports car parked on a sandy beach at sunset",
    "category": "attribute_binding",
    "seed": 42,
    "image": <PIL.Image>,
    "global_score": 0.7234,
    "patch_scores": "[0.65, 0.72, ...]",  # 49 values (7x7 grid)
    "pair_role": "chosen",  # or "rejected"
    "failure_type": null  # or "attribute", "object", "spatial"
}}
```

## Columns

| Column | Type | Description |
|--------|------|-------------|
| `image_id` | string | Unique identifier |
| `prompt_id` | string | Prompt identifier with variation |
| `base_id` | string | Base prompt identifier |
| `variation` | string | v0_original, v1_attribute, v2_object, v3_spatial |
| `prompt` | string | Text prompt used for generation |
| `category` | string | Faithfulness category |
| `seed` | int | Random seed (fixed at 42) |
| `image` | Image | Generated image |
| `global_score` | float | CLIP alignment score (0-1) |
| `patch_scores` | string | JSON array of 49 patch scores |
| `pair_role` | string | "chosen" (v0) or "rejected" (v1/v2/v3) |
| `failure_type` | string | Type of intentional failure |

## Pairing Logic

For each base prompt, 3 pairs are created:
1. v0_original (chosen) vs v1_attribute (rejected) - **Attribute Failure**
2. v0_original (chosen) vs v2_object (rejected) - **Object Failure**
3. v0_original (chosen) vs v3_spatial (rejected) - **Spatial/Count Failure**

## Categories

- **object_presence**: Are all mentioned objects present?
- **attribute_binding**: Are attributes (color, size) correct?
- **counting**: Is the count of objects correct?
- **spatial_relations**: Are spatial relationships correct?
- **compositional**: Complex multi-object scenes

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("{HF_REPO}")

# Get chosen (v0) vs rejected pairs
chosen = dataset.filter(lambda x: x["variation"] == "v0_original")
attribute_failures = dataset.filter(lambda x: x["variation"] == "v1_attribute")
object_failures = dataset.filter(lambda x: x["variation"] == "v2_object")
spatial_failures = dataset.filter(lambda x: x["variation"] == "v3_spatial")

# Group by base_id for training
import json
for sample in dataset:
    patch_scores = json.loads(sample["patch_scores"])  # List of 49 floats
```

## Patch Grid Layout

```
+-----+-----+-----+-----+-----+-----+-----+
|  0  |  1  |  2  |  3  |  4  |  5  |  6  |
+-----+-----+-----+-----+-----+-----+-----+
|  7  |  8  |  9  | 10  | 11  | 12  | 13  |
+-----+-----+-----+-----+-----+-----+-----+
| ... |     |     |     |     |     | ... |
+-----+-----+-----+-----+-----+-----+-----+
| 42  | 43  | 44  | 45  | 46  | 47  | 48  |
+-----+-----+-----+-----+-----+-----+-----+
```

Low patch score = region fails to match prompt (potential issue).

## Citation

```bibtex
@misc{{tipai-poc-{datetime.now().year}}},
  title={{TiPAI-POC: Patch-Level Faithfulness Dataset with Systematic Failure Variations}},
  author={{Kapil Wanaskar}},
  year={{{datetime.now().year}}},
  publisher={{HuggingFace}},
  url={{https://huggingface.co/datasets/{HF_REPO}}}
}}
```

## References

- [TIFA Benchmark](https://tifa-benchmark.github.io/) - Failure category taxonomy
- [CLIP](https://openai.com/research/clip) - Patch-level scoring

## License

MIT License
"""
    return readme


# ============================================================================
# Push to HuggingFace
# ============================================================================
def push_dataset_to_hf(db_path: Path = None, images_dir: Path = None, push_readme: bool = True):
    """Push dataset to HuggingFace with auto-generated README."""
    project_root = get_project_root()
    db_path = db_path or project_root / "outputs" / "centralized.db"
    images_dir = images_dir or project_root / "outputs" / "m01_images"

    if not db_path.exists():
        print(f"ERROR: Database not found at {db_path}")
        return False

    print("Loading data from database...")
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            i.image_id, i.prompt_id, i.base_id, i.variation,
            i.prompt_text, i.prompt_category, i.seed, i.image_path,
            COALESCE(s.global_score, 0.0) as global_score,
            COALESCE(s.patch_scores, '[]') as patch_scores
        FROM images i
        LEFT JOIN scores s ON i.image_id = s.image_id
        ORDER BY i.base_id, i.variation
    """)
    rows = cursor.fetchall()

    # Load pairs for failure type info
    cursor.execute("SELECT chosen_image_id, rejected_image_id, failure_type, score_gap FROM pairs")
    pairs_info = {}
    for chosen_id, rejected_id, failure_type, gap in cursor.fetchall():
        pairs_info[chosen_id] = ("chosen", None, gap)
        pairs_info[rejected_id] = ("rejected", failure_type, gap)

    conn.close()

    if not rows:
        print("ERROR: No images found in database.")
        return False

    # Build dataset
    data = {
        "image_id": [], "prompt_id": [], "base_id": [], "variation": [],
        "prompt": [], "category": [], "seed": [], "image": [],
        "global_score": [], "patch_scores": [],
        "pair_role": [], "failure_type": [], "score_gap": []
    }

    for row in rows:
        image_id, prompt_id, base_id, variation, prompt, category, seed, image_path, score, patches = row
        full_path = project_root / image_path

        if not full_path.exists():
            print(f"Skipping {image_id}: not found")
            continue

        pair_role, failure_type, score_gap = pairs_info.get(image_id, ("chosen" if variation == "v0_original" else "rejected", None, 0.0))

        data["image_id"].append(image_id)
        data["prompt_id"].append(prompt_id)
        data["base_id"].append(base_id)
        data["variation"].append(variation)
        data["prompt"].append(prompt)
        data["category"].append(category)
        data["seed"].append(seed)
        data["image"].append(str(full_path))
        data["global_score"].append(score)
        data["patch_scores"].append(patches)
        data["pair_role"].append(pair_role)
        data["failure_type"].append(failure_type)
        data["score_gap"].append(score_gap)

    dataset = Dataset.from_dict(data)
    dataset = dataset.cast_column("image", HFImage())

    print(f"Dataset: {len(dataset)} samples")

    # Push
    token = load_hf_token()
    dataset.push_to_hub(HF_REPO, token=token, private=False)

    # Push README
    if push_readme:
        stats = get_dataset_stats(db_path)
        readme = generate_readme_card(stats)

        api = HfApi()
        api.upload_file(
            path_or_fileobj=readme.encode(),
            path_in_repo="README.md",
            repo_id=HF_REPO,
            repo_type="dataset",
            token=token
        )
        print("README card uploaded.")

    print(f"\n✅ https://huggingface.co/datasets/{HF_REPO}")
    return True


# ============================================================================
# Download from HuggingFace
# ============================================================================
def download_dataset_from_hf(output_dir: Path = None) -> bool:
    """Download dataset from HuggingFace and save locally."""
    project_root = get_project_root()
    output_dir = output_dir or project_root / "outputs" / "m01_images"
    db_path = project_root / "outputs" / "centralized.db"

    print(f"Downloading from {HF_REPO}...")

    try:
        dataset = load_dataset(HF_REPO, split="train")
    except Exception as e:
        print(f"ERROR: Could not download dataset: {e}")
        return False

    print(f"Downloaded {len(dataset)} samples")

    # Create directories
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize database
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS images (
            image_id TEXT PRIMARY KEY,
            prompt_id TEXT NOT NULL,
            prompt_text TEXT NOT NULL,
            prompt_category TEXT NOT NULL,
            seed INTEGER NOT NULL,
            image_path TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scores (
            image_id TEXT PRIMARY KEY,
            prompt_text TEXT NOT NULL,
            global_score REAL NOT NULL,
            patch_scores TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Save images and data
    for sample in dataset:
        image_id = sample["image_id"]
        image_path = output_dir / f"{image_id}.png"

        # Save image
        sample["image"].save(str(image_path))

        # Save to DB
        cursor.execute("""
            INSERT OR REPLACE INTO images
            (image_id, prompt_id, prompt_text, prompt_category, seed, image_path)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (image_id, sample["prompt_id"], sample["prompt"],
              sample["category"], sample["seed"],
              str(image_path.relative_to(project_root))))

        if sample["global_score"] > 0:
            cursor.execute("""
                INSERT OR REPLACE INTO scores
                (image_id, prompt_text, global_score, patch_scores)
                VALUES (?, ?, ?, ?)
            """, (image_id, sample["prompt"], sample["global_score"], sample["patch_scores"]))

    conn.commit()
    conn.close()

    print(f"✅ Saved to {output_dir}")
    return True


def is_dataset_available_locally(images_dir: Path = None, min_images: int = 1) -> bool:
    """Check if dataset exists locally."""
    project_root = get_project_root()
    images_dir = images_dir or project_root / "outputs" / "m01_images"

    if not images_dir.exists():
        return False

    image_count = len(list(images_dir.glob("*.png")))
    return image_count >= min_images


def ensure_dataset_available(images_dir: Path = None) -> bool:
    """Ensure dataset is available locally, download if not."""
    if is_dataset_available_locally(images_dir):
        print("Dataset available locally.")
        return True

    print("Dataset not found locally. Downloading from HuggingFace...")
    return download_dataset_from_hf(images_dir)


# ============================================================================
# Main (standalone test)
# ============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true", help="Push to HuggingFace")
    parser.add_argument("--download", action="store_true", help="Download from HuggingFace")
    parser.add_argument("--check", action="store_true", help="Check local availability")
    args = parser.parse_args()

    if args.push:
        push_dataset_to_hf()
    elif args.download:
        download_dataset_from_hf()
    elif args.check:
        available = is_dataset_available_locally()
        print(f"Dataset available locally: {available}")
    else:
        print("Use --push, --download, or --check")
