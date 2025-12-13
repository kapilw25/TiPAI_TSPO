"""
m02_clip_scoring.py
Score images using CLIP (global + patch-level).
Reads from: centralized.db (images table)
Writes to: centralized.db (scores table)

KEY DESIGN:
  - self_score: Image scored against its OWN prompt (generation quality)
  - baseline_score: Image scored against v0_original prompt (faithfulness to baseline)

  For POC comparison, baseline_score is what matters:
    v0 image vs v0 prompt → high baseline_score
    v1 image vs v0 prompt → low baseline_score (different attribute)

REQUIRES: CUDA GPU

Run:
    python -u src/m02_clip_scoring.py 2>&1 | tee logs/m02_$(date +%Y%m%d_%H%M%S).log
"""

import json
import sqlite3
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import open_clip


# ============================================================================
# Configuration
# ============================================================================
class Config:
    PROJECT_ROOT = Path(__file__).parent.parent
    DB_PATH = PROJECT_ROOT / "outputs" / "centralized.db"
    IMAGES_DIR = PROJECT_ROOT / "outputs" / "m01_images"

    CLIP_MODEL = "ViT-L-14"
    CLIP_PRETRAINED = "openai"
    PATCH_GRID = 7  # 7x7 = 49 patches
    IMAGE_SIZE = 224  # CLIP input size


# ============================================================================
# Database
# ============================================================================
def init_scores_table(conn: sqlite3.Connection) -> None:
    """Create scores table if not exists."""
    cursor = conn.cursor()
    # Drop old table to recreate with new schema
    cursor.execute("DROP TABLE IF EXISTS scores")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scores (
            image_id TEXT PRIMARY KEY,
            prompt_text TEXT NOT NULL,
            baseline_prompt TEXT NOT NULL,
            self_score REAL NOT NULL,
            self_patch_scores TEXT NOT NULL,
            baseline_score REAL NOT NULL,
            baseline_patch_scores TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()


def get_images_to_score(conn: sqlite3.Connection) -> list:
    """Get all images that need scoring with their baseline (v0) prompt."""
    cursor = conn.cursor()
    # Get all images with their v0_original baseline prompt
    cursor.execute("""
        SELECT
            i.image_id,
            i.prompt_text,
            i.image_path,
            i.base_id,
            i.variation,
            v0.prompt_text as baseline_prompt
        FROM images i
        LEFT JOIN images v0 ON i.base_id = v0.base_id AND v0.variation = 'v0_original'
        LEFT JOIN scores s ON i.image_id = s.image_id
        WHERE s.image_id IS NULL
    """)
    return cursor.fetchall()


def save_score(conn: sqlite3.Connection, image_id: str, prompt_text: str,
               baseline_prompt: str, self_score: float, self_patch_scores: list,
               baseline_score: float, baseline_patch_scores: list) -> None:
    """Save score to database."""
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO scores
        (image_id, prompt_text, baseline_prompt, self_score, self_patch_scores,
         baseline_score, baseline_patch_scores)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (image_id, prompt_text, baseline_prompt, self_score, json.dumps(self_patch_scores),
          baseline_score, json.dumps(baseline_patch_scores)))
    conn.commit()


# ============================================================================
# CLIP Scoring (CUDA ONLY)
# ============================================================================
def load_clip_model(config: Config):
    """Load CLIP model on CUDA."""
    assert torch.cuda.is_available(), "CUDA required!"

    print(f"Loading CLIP {config.CLIP_MODEL} on CUDA...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        config.CLIP_MODEL, pretrained=config.CLIP_PRETRAINED
    )
    model = model.to("cuda").eval()
    tokenizer = open_clip.get_tokenizer(config.CLIP_MODEL)
    return model, preprocess, tokenizer


def get_text_embedding(model, tokenizer, text: str) -> torch.Tensor:
    """Get CLIP text embedding."""
    tokens = tokenizer([text]).to("cuda")
    with torch.no_grad():
        text_emb = model.encode_text(tokens)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    return text_emb


def get_image_embedding(model, preprocess, image: Image.Image) -> torch.Tensor:
    """Get CLIP image embedding."""
    img_tensor = preprocess(image).unsqueeze(0).to("cuda")
    with torch.no_grad():
        img_emb = model.encode_image(img_tensor)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
    return img_emb


def compute_global_score(text_emb: torch.Tensor, img_emb: torch.Tensor) -> float:
    """Compute cosine similarity between text and image."""
    similarity = (text_emb @ img_emb.T).item()
    # Scale from [-1,1] to [0,1]
    score = (similarity + 1) / 2
    return score


def extract_patches(image: Image.Image, grid_size: int) -> list:
    """Split image into grid_size x grid_size patches."""
    image = image.convert("RGB")
    w, h = image.size
    patch_w, patch_h = w // grid_size, h // grid_size

    patches = []
    for row in range(grid_size):
        for col in range(grid_size):
            left = col * patch_w
            top = row * patch_h
            right = left + patch_w
            bottom = top + patch_h
            patch = image.crop((left, top, right, bottom))
            patches.append(patch)
    return patches


def compute_patch_scores(model, preprocess, tokenizer, image: Image.Image,
                        prompt: str, grid_size: int) -> list:
    """Compute CLIP score for each patch."""
    text_emb = get_text_embedding(model, tokenizer, prompt)
    patches = extract_patches(image, grid_size)

    scores = []
    for patch in patches:
        patch_emb = get_image_embedding(model, preprocess, patch)
        score = compute_global_score(text_emb, patch_emb)
        scores.append(round(score, 4))
    return scores


def score_all_images(config: Config) -> None:
    """Score all images in database."""
    # Ensure dataset is available (download from HF if not local)
    from utils.hf_utils import ensure_dataset_available
    if not ensure_dataset_available(config.IMAGES_DIR):
        print("ERROR: Could not get dataset locally or from HuggingFace.")
        return

    conn = sqlite3.connect(str(config.DB_PATH))
    init_scores_table(conn)

    images = get_images_to_score(conn)
    if not images:
        print("No images to score.")
        conn.close()
        return

    print(f"Images to score: {len(images)}")
    model, preprocess, tokenizer = load_clip_model(config)

    for image_id, prompt_text, image_path, base_id, variation, baseline_prompt in tqdm(images, desc="Scoring"):
        full_path = config.PROJECT_ROOT / image_path
        if not full_path.exists():
            print(f"Image not found: {full_path}")
            continue

        image = Image.open(full_path).convert("RGB")
        img_emb = get_image_embedding(model, preprocess, image)

        # Self score (image vs its own prompt) - measures generation quality
        self_text_emb = get_text_embedding(model, tokenizer, prompt_text)
        self_score = compute_global_score(self_text_emb, img_emb)
        self_patch_scores = compute_patch_scores(model, preprocess, tokenizer,
                                                  image, prompt_text, config.PATCH_GRID)

        # Baseline score (image vs v0_original prompt) - measures faithfulness to baseline
        baseline_text_emb = get_text_embedding(model, tokenizer, baseline_prompt)
        baseline_score = compute_global_score(baseline_text_emb, img_emb)
        baseline_patch_scores = compute_patch_scores(model, preprocess, tokenizer,
                                                      image, baseline_prompt, config.PATCH_GRID)

        save_score(conn, image_id, prompt_text, baseline_prompt,
                   self_score, self_patch_scores, baseline_score, baseline_patch_scores)

    conn.close()
    print("Done!")


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    config = Config()
    score_all_images(config)
