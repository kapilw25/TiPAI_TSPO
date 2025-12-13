"""
m01_generate_images.py
Generate images using Stable Diffusion or FLUX for each prompt variation.
Output: 20 base prompts × 4 variations × 1 seed = 80 images

Run:
    mkdir -p logs
    # SDXL (default)
    python -u src/m01_generate_images.py 2>&1 | tee logs/m01_sdxl_$(date +%Y%m%d_%H%M%S).log
    # FLUX.1-dev (better multi-object)
    python -u src/m01_generate_images.py --model flux 2>&1 | tee logs/m01_flux_$(date +%Y%m%d_%H%M%S).log
    # With HuggingFace push
    python -u src/m01_generate_images.py --model flux --push-hf 2>&1 | tee logs/m01_flux_$(date +%Y%m%d_%H%M%S).log

Models:
  - SDXL (default): stabilityai/stable-diffusion-xl-base-1.0
  - FLUX: black-forest-labs/FLUX.1-dev (better prompt following)

Structure:
  - v0_original: Correct prompt (chosen baseline)
  - v1_attribute: Wrong color/size/material
  - v2_object: Swapped/wrong main object
  - v3_spatial: Wrong spatial relation or count

Each image has text overlay showing:
  - baseline_prompt: The v0_original prompt (reference)
  - generation_prompt: The prompt used to generate this image

REQUIRES: CUDA GPU (no CPU/MPS fallback)
"""

import json
import sqlite3
import textwrap
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from diffusers import DiffusionPipeline, FluxPipeline


# ============================================================================
# Configuration
# ============================================================================
class Config:
    PROJECT_ROOT = Path(__file__).parent.parent
    PROMPTS_PATH = PROJECT_ROOT / "data" / "prompts.json"
    OUTPUT_DIR = PROJECT_ROOT / "outputs" / "m01_images"
    DB_PATH = PROJECT_ROOT / "outputs" / "centralized.db"

    SEED = 42  # Single seed for all variations

    # Text overlay settings
    TEXT_HEADER_HEIGHT = 80  # Pixels for text above image
    FONT_SIZE = 12

    # Model configurations
    MODELS = {
        "sdxl": {
            "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
            "pipeline_class": "DiffusionPipeline",
            "torch_dtype": torch.float16,
            "image_size": 512,
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "use_cpu_offload": False,
            "generator_device": "cuda",
        },
        "flux": {
            "model_id": "black-forest-labs/FLUX.1-dev",
            "pipeline_class": "FluxPipeline",
            "torch_dtype": torch.bfloat16,
            "image_size": 512,  # Can be 1024 for higher quality
            "num_inference_steps": 50,
            "guidance_scale": 3.5,
            "use_cpu_offload": True,  # FLUX needs more VRAM
            "generator_device": "cpu",  # FLUX uses CPU generator
            "max_sequence_length": 512,
        },
    }

    # Active model (set via command line)
    active_model = "sdxl"


# ============================================================================
# Database
# ============================================================================
def init_database(db_path: Path) -> sqlite3.Connection:
    """Initialize SQLite database with images table."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Drop old table to update schema
    cursor.execute("DROP TABLE IF EXISTS images")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS images (
            image_id TEXT PRIMARY KEY,
            prompt_id TEXT NOT NULL,
            base_id TEXT NOT NULL,
            variation TEXT NOT NULL,
            baseline_prompt TEXT NOT NULL,
            generation_prompt TEXT NOT NULL,
            prompt_category TEXT NOT NULL,
            seed INTEGER NOT NULL,
            image_path TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


def save_image_record(conn: sqlite3.Connection, image_id: str, prompt_id: str,
                      base_id: str, variation: str, baseline_prompt: str,
                      generation_prompt: str, prompt_category: str,
                      seed: int, image_path: str) -> None:
    """Save image record to database."""
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO images
        (image_id, prompt_id, base_id, variation, baseline_prompt, generation_prompt,
         prompt_category, seed, image_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (image_id, prompt_id, base_id, variation, baseline_prompt, generation_prompt,
          prompt_category, seed, image_path))
    conn.commit()


# ============================================================================
# Image Generation (CUDA ONLY)
# ============================================================================
def load_prompts(prompts_path: Path) -> tuple[list, dict]:
    """
    Load prompts from JSON file.

    Returns:
        prompts: List of all prompt dicts
        baseline_prompts: Dict mapping base_id -> v0_original prompt text
    """
    with open(prompts_path, 'r') as f:
        data = json.load(f)

    prompts = data['prompts']

    # Build baseline prompt lookup (v0_original for each base_id)
    baseline_prompts = {}
    for p in prompts:
        if p['variation'] == 'v0_original':
            baseline_prompts[p['base_id']] = p['text']

    return prompts, baseline_prompts


def load_hf_token() -> str:
    """Load HuggingFace token from .env file."""
    env_path = Config.PROJECT_ROOT / ".env"
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                if line.startswith('HF_TOKEN='):
                    return line.strip().split('=', 1)[1]
    return None


def load_pipeline(config: Config):
    """Load diffusion pipeline based on active model config."""
    assert torch.cuda.is_available(), "CUDA required! No CPU/MPS fallback."

    model_config = config.MODELS[config.active_model]
    model_id = model_config["model_id"]

    # Load HF token for gated models
    hf_token = load_hf_token()

    print(f"Loading {model_id} ({config.active_model.upper()}) on CUDA...")

    if config.active_model == "flux":
        # FLUX.1-dev pipeline (gated model - requires token)
        if not hf_token:
            raise ValueError("FLUX.1-dev requires HF_TOKEN in .env file")

        pipe = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=model_config["torch_dtype"],
            token=hf_token,
        )
        if model_config["use_cpu_offload"]:
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to("cuda")
    else:
        # SDXL pipeline
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=model_config["torch_dtype"],
            use_safetensors=True,
            variant="fp16",
            token=hf_token,  # Optional for SDXL but good to have
        )
        pipe = pipe.to("cuda")

    return pipe


def generate_image(pipe, prompt: str, seed: int, config: Config) -> Image.Image:
    """Generate a single image based on active model config."""
    model_config = config.MODELS[config.active_model]

    # Create generator on appropriate device
    generator = torch.Generator(device=model_config["generator_device"]).manual_seed(seed)

    if config.active_model == "flux":
        # FLUX generation
        image = pipe(
            prompt=prompt,
            height=model_config["image_size"],
            width=model_config["image_size"],
            guidance_scale=model_config["guidance_scale"],
            num_inference_steps=model_config["num_inference_steps"],
            max_sequence_length=model_config["max_sequence_length"],
            generator=generator,
        ).images[0]
    else:
        # SDXL generation
        image = pipe(
            prompt=prompt,
            num_inference_steps=model_config["num_inference_steps"],
            guidance_scale=model_config["guidance_scale"],
            height=model_config["image_size"],
            width=model_config["image_size"],
            generator=generator,
        ).images[0]

    return image


def add_text_overlay(image: Image.Image, baseline_prompt: str, generation_prompt: str,
                     variation: str, config: Config) -> Image.Image:
    """
    Add text header above image showing baseline and generation prompts.

    For v0_original: baseline == generation (no difference highlighted)
    For variations: shows both prompts for comparison
    """
    img_w, img_h = image.size
    header_h = config.TEXT_HEADER_HEIGHT

    # Create new image with header space
    new_img = Image.new('RGB', (img_w, img_h + header_h), color=(255, 255, 255))
    new_img.paste(image, (0, header_h))

    # Draw text
    draw = ImageDraw.Draw(new_img)

    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", config.FONT_SIZE)
        font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", config.FONT_SIZE)
    except:
        font = ImageFont.load_default()
        font_bold = font

    # Wrap text to fit image width
    max_chars = img_w // 6  # Approximate chars that fit

    # Format prompts
    baseline_wrapped = textwrap.fill(f"BASELINE: {baseline_prompt}", width=max_chars)
    gen_wrapped = textwrap.fill(f"GENERATED ({variation}): {generation_prompt}", width=max_chars)

    # Colors
    if variation == "v0_original":
        gen_color = (0, 128, 0)  # Green for correct
    else:
        gen_color = (200, 0, 0)  # Red for variations

    # Draw baseline prompt
    draw.text((5, 2), baseline_wrapped, fill=(0, 0, 0), font=font)

    # Draw generation prompt (below baseline)
    y_offset = 2 + (baseline_wrapped.count('\n') + 1) * (config.FONT_SIZE + 2)
    draw.text((5, y_offset), gen_wrapped, fill=gen_color, font=font)

    return new_img


def check_existing_images(config: Config, prompts: list) -> list:
    """Check for existing images and return list of existing paths."""
    existing = []
    for prompt_data in prompts:
        prompt_id = prompt_data['id']
        image_id = f"{prompt_id}_seed{config.SEED}"
        image_path = config.OUTPUT_DIR / f"{image_id}.png"
        if image_path.exists():
            existing.append(image_path)
    return existing


def prompt_user_for_action(existing_count: int, total_count: int) -> str:
    """Ask user whether to resume or restart."""
    print(f"\n{'='*60}")
    print(f"Found {existing_count}/{total_count} images already generated.")
    print(f"{'='*60}")
    print("Options:")
    print("  [1] RESUME - Skip existing, generate remaining images")
    print("  [2] RESTART - Delete all existing images, start fresh")
    print(f"{'='*60}")

    while True:
        choice = input("Enter choice [1/2]: ").strip()
        if choice in ['1', '2']:
            return choice
        print("Invalid choice. Enter 1 or 2.")


def delete_existing_images(config: Config, prompts: list) -> None:
    """Delete all existing images and clear DB records."""
    # Delete image files
    if config.OUTPUT_DIR.exists():
        for png in config.OUTPUT_DIR.glob("*.png"):
            png.unlink()
        print(f"Deleted all images from {config.OUTPUT_DIR}")

    # Clear DB records
    if config.DB_PATH.exists():
        conn = sqlite3.connect(str(config.DB_PATH))
        cursor = conn.cursor()
        cursor.execute("DELETE FROM images")
        conn.commit()
        conn.close()
        print("Cleared images table in database")


def generate_all_images(config: Config) -> None:
    """Generate all images for all prompt variations."""
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prompts, baseline_prompts = load_prompts(config.PROMPTS_PATH)

    model_config = config.MODELS[config.active_model]
    print(f"Model: {config.active_model.upper()} ({model_config['model_id']})")
    print(f"Total prompts (with variations): {len(prompts)}")
    print(f"Base prompts: {len(baseline_prompts)}")
    print(f"Seed: {config.SEED}")
    print(f"Image size: {model_config['image_size']}x{model_config['image_size']}")

    # Check for existing images
    existing = check_existing_images(config, prompts)
    if existing:
        choice = prompt_user_for_action(len(existing), len(prompts))
        if choice == '2':
            delete_existing_images(config, prompts)
            print("Starting fresh generation...")
        else:
            print(f"Resuming... Will skip {len(existing)} existing images.")

    conn = init_database(config.DB_PATH)
    pipe = load_pipeline(config)
    pbar = tqdm(total=len(prompts), desc=f"Generating ({config.active_model})")

    for prompt_data in prompts:
        prompt_id = prompt_data['id']
        base_id = prompt_data['base_id']
        variation = prompt_data['variation']
        generation_prompt = prompt_data['text']
        prompt_category = prompt_data['category']

        # Get baseline prompt for this base_id
        baseline_prompt = baseline_prompts.get(base_id, generation_prompt)

        image_id = f"{prompt_id}_seed{config.SEED}"
        image_path = config.OUTPUT_DIR / f"{image_id}.png"

        if image_path.exists():
            # Still save to DB if not already there
            save_image_record(conn, image_id, prompt_id, base_id, variation,
                              baseline_prompt, generation_prompt, prompt_category,
                              config.SEED, str(image_path.relative_to(config.PROJECT_ROOT)))
            pbar.update(1)
            continue

        # Generate image
        image = generate_image(pipe, generation_prompt, config.SEED, config)

        # Add text overlay
        image_with_text = add_text_overlay(image, baseline_prompt, generation_prompt,
                                           variation, config)

        # Save
        image_with_text.save(str(image_path))
        save_image_record(conn, image_id, prompt_id, base_id, variation,
                          baseline_prompt, generation_prompt, prompt_category,
                          config.SEED, str(image_path.relative_to(config.PROJECT_ROOT)))
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
    parser.add_argument("--model", type=str, default="sdxl", choices=["sdxl", "flux"],
                        help="Model to use: sdxl (default) or flux")
    parser.add_argument("--push-hf", action="store_true", help="Push to HuggingFace after generation")
    args = parser.parse_args()

    config = Config()
    config.active_model = args.model

    print(f"\n{'='*60}")
    print(f"TiPAI Image Generation")
    print(f"{'='*60}")

    generate_all_images(config)

    if args.push_hf:
        from utils.hf_utils import push_dataset_to_hf
        push_dataset_to_hf(db_path=config.DB_PATH, images_dir=config.OUTPUT_DIR)
