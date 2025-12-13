"""
m01_generate_images.py
Generate images using Stable Diffusion for each prompt variation.
Output: 20 base prompts × 4 variations × 1 seed = 80 images

Structure:
  - v0_original: Correct prompt (chosen baseline)
  - v1_attribute: Wrong color/size/material
  - v2_object: Swapped/wrong main object
  - v3_spatial: Wrong spatial relation or count

REQUIRES: CUDA GPU (no CPU/MPS fallback)

Run:
    python src/m01_generate_images.py --push-hf
"""

import json
import sqlite3
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from diffusers import DiffusionPipeline


# ============================================================================
# Configuration
# ============================================================================
class Config:
    PROJECT_ROOT = Path(__file__).parent.parent
    PROMPTS_PATH = PROJECT_ROOT / "data" / "prompts.json"
    OUTPUT_DIR = PROJECT_ROOT / "outputs" / "m01_images"
    DB_PATH = PROJECT_ROOT / "outputs" / "centralized.db"

    MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
    IMAGE_SIZE = 512
    NUM_INFERENCE_STEPS = 30
    GUIDANCE_SCALE = 7.5
    SEED = 42  # Single seed for all variations


# ============================================================================
# Database
# ============================================================================
def init_database(db_path: Path) -> sqlite3.Connection:
    """Initialize SQLite database with images table."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS images (
            image_id TEXT PRIMARY KEY,
            prompt_id TEXT NOT NULL,
            base_id TEXT NOT NULL,
            variation TEXT NOT NULL,
            prompt_text TEXT NOT NULL,
            prompt_category TEXT NOT NULL,
            seed INTEGER NOT NULL,
            image_path TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


def save_image_record(conn: sqlite3.Connection, image_id: str, prompt_id: str,
                      base_id: str, variation: str, prompt_text: str,
                      prompt_category: str, seed: int, image_path: str) -> None:
    """Save image record to database."""
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO images
        (image_id, prompt_id, base_id, variation, prompt_text, prompt_category, seed, image_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (image_id, prompt_id, base_id, variation, prompt_text, prompt_category, seed, image_path))
    conn.commit()


# ============================================================================
# Image Generation (CUDA ONLY)
# ============================================================================
def load_prompts(prompts_path: Path) -> list:
    """Load prompts from JSON file."""
    with open(prompts_path, 'r') as f:
        data = json.load(f)
    return data['prompts']


def load_pipeline(model_id: str) -> DiffusionPipeline:
    """Load Stable Diffusion pipeline on CUDA (official SDXL usage)."""
    assert torch.cuda.is_available(), "CUDA required! No CPU/MPS fallback."

    print(f"Loading {model_id} on CUDA...")
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    pipe = pipe.to("cuda")

    # Optional: 20-30% speedup with torch.compile (torch >= 2.0)
    # Uncomment for faster generation (adds ~60s compile time on first run)
    # pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

    return pipe


def generate_image(pipe: DiffusionPipeline, prompt: str, seed: int, config: Config) -> Image.Image:
    """Generate a single image on CUDA."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    image = pipe(
        prompt=prompt,
        num_inference_steps=config.NUM_INFERENCE_STEPS,
        guidance_scale=config.GUIDANCE_SCALE,
        height=config.IMAGE_SIZE,
        width=config.IMAGE_SIZE,
        generator=generator
    ).images[0]
    return image


def generate_all_images(config: Config) -> None:
    """Generate all images for all prompt variations."""
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    conn = init_database(config.DB_PATH)
    prompts = load_prompts(config.PROMPTS_PATH)

    print(f"Total prompts (with variations): {len(prompts)}")
    print(f"Seed: {config.SEED}")

    pipe = load_pipeline(config.MODEL_ID)
    pbar = tqdm(total=len(prompts), desc="Generating")

    for prompt_data in prompts:
        prompt_id = prompt_data['id']
        base_id = prompt_data['base_id']
        variation = prompt_data['variation']
        prompt_text = prompt_data['text']
        prompt_category = prompt_data['category']

        image_id = f"{prompt_id}_seed{config.SEED}"
        image_path = config.OUTPUT_DIR / f"{image_id}.png"

        if image_path.exists():
            pbar.update(1)
            continue

        image = generate_image(pipe, prompt_text, config.SEED, config)
        image.save(str(image_path))
        save_image_record(conn, image_id, prompt_id, base_id, variation,
                          prompt_text, prompt_category, config.SEED,
                          str(image_path.relative_to(config.PROJECT_ROOT)))
        pbar.update(1)

    pbar.close()
    conn.close()
    print(f"Done! Saved to {config.OUTPUT_DIR}")


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--push-hf", action="store_true", help="Push to HuggingFace after generation")
    args = parser.parse_args()

    config = Config()
    generate_all_images(config)

    if args.push_hf:
        from utils.hf_utils import push_dataset_to_hf
        push_dataset_to_hf(db_path=config.DB_PATH, images_dir=config.OUTPUT_DIR)
