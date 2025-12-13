"""
m02_clip_scoring.py
Score images using CLIP (global + patch-level).
Reads from: centralized.db (images table)
Writes to: centralized.db (scores table)

REQUIRES: CUDA GPU

Run:
    python src/m02_clip_scoring.py
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
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scores (
            image_id TEXT PRIMARY KEY,
            prompt_text TEXT NOT NULL,
            global_score REAL NOT NULL,
            patch_scores TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()


def get_images_to_score(conn: sqlite3.Connection) -> list:
    """Get all images that need scoring."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT i.image_id, i.prompt_text, i.image_path
        FROM images i
        LEFT JOIN scores s ON i.image_id = s.image_id
        WHERE s.image_id IS NULL
    """)
    return cursor.fetchall()


def save_score(conn: sqlite3.Connection, image_id: str, prompt_text: str,
               global_score: float, patch_scores: list) -> None:
    """Save score to database."""
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO scores (image_id, prompt_text, global_score, patch_scores)
        VALUES (?, ?, ?, ?)
    """, (image_id, prompt_text, global_score, json.dumps(patch_scores)))
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

    for image_id, prompt_text, image_path in tqdm(images, desc="Scoring"):
        full_path = config.PROJECT_ROOT / image_path
        if not full_path.exists():
            print(f"Image not found: {full_path}")
            continue

        image = Image.open(full_path).convert("RGB")

        # Global score
        text_emb = get_text_embedding(model, tokenizer, prompt_text)
        img_emb = get_image_embedding(model, preprocess, image)
        global_score = compute_global_score(text_emb, img_emb)

        # Patch scores
        patch_scores = compute_patch_scores(model, preprocess, tokenizer,
                                           image, prompt_text, config.PATCH_GRID)

        save_score(conn, image_id, prompt_text, global_score, patch_scores)

    conn.close()
    print("Done!")


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    config = Config()
    score_all_images(config)
