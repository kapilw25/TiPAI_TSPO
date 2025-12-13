"""
m05_demo_app.py
Gradio demo for TiPAI POC - Patch-Level Faithfulness Heatmap.

Two modes:
1. Browse: View pre-generated pairs from database
2. Live: Generate new images on-the-fly (requires GPU)

REQUIRES: CUDA GPU for live mode

Run:
    python src/m05_demo_app.py --port 7860
    python src/m05_demo_app.py --live --port 7860
"""

import json
import sqlite3
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import zoom
import gradio as gr


# ============================================================================
# Configuration
# ============================================================================
class Config:
    PROJECT_ROOT = Path(__file__).parent.parent
    DB_PATH = PROJECT_ROOT / "outputs" / "centralized.db"
    IMAGES_DIR = PROJECT_ROOT / "outputs" / "m01_images"
    HEATMAPS_DIR = PROJECT_ROOT / "outputs" / "m03_heatmaps"
    PAIRS_DIR = PROJECT_ROOT / "outputs" / "m04_pairs"

    CLIP_MODEL = "ViT-L-14"
    CLIP_PRETRAINED = "openai"
    PATCH_GRID = 7
    OVERLAY_ALPHA = 0.5


# ============================================================================
# Database Functions
# ============================================================================
def get_all_pairs(config: Config) -> list:
    """Get all pairs from database."""
    conn = sqlite3.connect(str(config.DB_PATH))
    cursor = conn.cursor()
    cursor.execute("""
        SELECT pair_id, prompt_text, chosen_image_id, rejected_image_id,
               chosen_score, rejected_score, comparison_path
        FROM pairs
        ORDER BY score_gap DESC
    """)
    results = cursor.fetchall()
    conn.close()
    return results


def get_pair_details(config: Config, pair_id: str) -> dict:
    """Get detailed info for a specific pair."""
    conn = sqlite3.connect(str(config.DB_PATH))
    cursor = conn.cursor()

    # Get pair info
    cursor.execute("""
        SELECT prompt_text, chosen_image_id, rejected_image_id, chosen_score, rejected_score
        FROM pairs WHERE pair_id = ?
    """, (pair_id,))
    pair = cursor.fetchone()

    if not pair:
        conn.close()
        return None

    prompt, chosen_id, rejected_id, chosen_score, rejected_score = pair

    # Get patch scores
    cursor.execute("SELECT patch_scores FROM scores WHERE image_id = ?", (chosen_id,))
    chosen_patches = json.loads(cursor.fetchone()[0])

    cursor.execute("SELECT patch_scores FROM scores WHERE image_id = ?", (rejected_id,))
    rejected_patches = json.loads(cursor.fetchone()[0])

    # Get image paths
    cursor.execute("SELECT image_path FROM images WHERE image_id = ?", (chosen_id,))
    chosen_path = cursor.fetchone()[0]

    cursor.execute("SELECT image_path FROM images WHERE image_id = ?", (rejected_id,))
    rejected_path = cursor.fetchone()[0]

    conn.close()

    return {
        "prompt": prompt,
        "chosen_id": chosen_id,
        "rejected_id": rejected_id,
        "chosen_score": chosen_score,
        "rejected_score": rejected_score,
        "chosen_patches": chosen_patches,
        "rejected_patches": rejected_patches,
        "chosen_path": config.PROJECT_ROOT / chosen_path,
        "rejected_path": config.PROJECT_ROOT / rejected_path,
    }


# ============================================================================
# Heatmap Functions
# ============================================================================
def create_heatmap(image: Image.Image, patch_scores: list, grid_size: int = 7, alpha: float = 0.5) -> Image.Image:
    """Create heatmap overlay."""
    scores_array = np.array(patch_scores).reshape(grid_size, grid_size)
    img_array = np.array(image.convert("RGB"))
    h, w = img_array.shape[:2]

    heatmap = zoom(scores_array, (h / grid_size, w / grid_size), order=1)
    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=0.3, vmax=0.8)
    heatmap_colored = (cmap(norm(heatmap))[:, :, :3] * 255).astype(np.uint8)

    blended = ((1 - alpha) * img_array + alpha * heatmap_colored).astype(np.uint8)
    return Image.fromarray(blended)


# ============================================================================
# Gradio Interface
# ============================================================================
def browse_pairs(pair_selection: str, config: Config):
    """Browse pre-generated pairs."""
    if not pair_selection:
        return None, None, None, None, "Select a pair"

    pair_id = pair_selection.split(" | ")[0]
    details = get_pair_details(config, pair_id)

    if not details:
        return None, None, None, None, "Pair not found"

    # Load images
    chosen_img = Image.open(details["chosen_path"]) if details["chosen_path"].exists() else None
    rejected_img = Image.open(details["rejected_path"]) if details["rejected_path"].exists() else None

    # Create heatmaps
    chosen_heat = create_heatmap(chosen_img, details["chosen_patches"]) if chosen_img else None
    rejected_heat = create_heatmap(rejected_img, details["rejected_patches"]) if rejected_img else None

    info = f"""**Prompt:** {details['prompt']}

**Chosen Score:** {details['chosen_score']:.4f}
**Rejected Score:** {details['rejected_score']:.4f}
**Score Gap:** {details['chosen_score'] - details['rejected_score']:.4f}

**Interpretation:**
- Green regions = high faithfulness to prompt
- Red regions = low faithfulness (potential issues)
"""
    return chosen_img, rejected_img, chosen_heat, rejected_heat, info


def create_browse_interface(config: Config):
    """Create browse interface for pre-generated pairs."""
    pairs = get_all_pairs(config)

    if not pairs:
        return gr.Markdown("No pairs found. Run the pipeline first (m01-m04).")

    pair_choices = [f"{p[0]} | {p[1][:50]}... (gap: {p[5]-p[4]:.3f})" for p in pairs]

    with gr.Blocks() as interface:
        gr.Markdown("# TiPAI POC: Patch-Level Faithfulness Demo")
        gr.Markdown("Select a prompt to see chosen vs rejected images with faithfulness heatmaps.")

        pair_dropdown = gr.Dropdown(choices=pair_choices, label="Select Prompt", value=pair_choices[0] if pair_choices else None)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### CHOSEN (Best)")
                chosen_img = gr.Image(label="Chosen Image", type="pil")
                chosen_heat = gr.Image(label="Chosen Heatmap", type="pil")

            with gr.Column():
                gr.Markdown("### REJECTED (Worst)")
                rejected_img = gr.Image(label="Rejected Image", type="pil")
                rejected_heat = gr.Image(label="Rejected Heatmap", type="pil")

        info_box = gr.Markdown("")

        pair_dropdown.change(
            fn=lambda x: browse_pairs(x, config),
            inputs=[pair_dropdown],
            outputs=[chosen_img, rejected_img, chosen_heat, rejected_heat, info_box]
        )

        # Load first pair on start
        if pair_choices:
            interface.load(
                fn=lambda: browse_pairs(pair_choices[0], config),
                outputs=[chosen_img, rejected_img, chosen_heat, rejected_heat, info_box]
            )

    return interface


# ============================================================================
# Live Generation Mode (requires GPU)
# ============================================================================
CLIP_MODEL = None
CLIP_PREPROCESS = None
CLIP_TOKENIZER = None


def load_clip_for_demo():
    """Load CLIP model for live scoring."""
    global CLIP_MODEL, CLIP_PREPROCESS, CLIP_TOKENIZER

    if CLIP_MODEL is not None:
        return

    import open_clip
    assert torch.cuda.is_available(), "CUDA required for live mode!"

    print("Loading CLIP for live demo...")
    CLIP_MODEL, _, CLIP_PREPROCESS = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    CLIP_MODEL = CLIP_MODEL.to("cuda").eval()
    CLIP_TOKENIZER = open_clip.get_tokenizer("ViT-L-14")


def score_image_live(image: Image.Image, prompt: str) -> tuple:
    """Score a single image live."""
    load_clip_for_demo()

    image = image.convert("RGB")

    # Text embedding
    tokens = CLIP_TOKENIZER([prompt]).to("cuda")
    with torch.no_grad():
        text_emb = CLIP_MODEL.encode_text(tokens)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    # Global score
    img_tensor = CLIP_PREPROCESS(image).unsqueeze(0).to("cuda")
    with torch.no_grad():
        img_emb = CLIP_MODEL.encode_image(img_tensor)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
    global_score = ((text_emb @ img_emb.T).item() + 1) / 2

    # Patch scores
    w, h = image.size
    grid = 7
    patch_w, patch_h = w // grid, h // grid
    patch_scores = []

    for row in range(grid):
        for col in range(grid):
            patch = image.crop((col * patch_w, row * patch_h, (col + 1) * patch_w, (row + 1) * patch_h))
            patch_tensor = CLIP_PREPROCESS(patch).unsqueeze(0).to("cuda")
            with torch.no_grad():
                patch_emb = CLIP_MODEL.encode_image(patch_tensor)
                patch_emb = patch_emb / patch_emb.norm(dim=-1, keepdim=True)
            score = ((text_emb @ patch_emb.T).item() + 1) / 2
            patch_scores.append(score)

    heatmap = create_heatmap(image, patch_scores)
    return global_score, patch_scores, heatmap


def live_score(image, prompt):
    """Gradio callback for live scoring."""
    if image is None or not prompt:
        return None, "Upload an image and enter a prompt"

    try:
        global_score, patch_scores, heatmap = score_image_live(image, prompt)

        min_patch = min(patch_scores)
        max_patch = max(patch_scores)
        min_idx = patch_scores.index(min_patch)

        info = f"""**Global Score:** {global_score:.4f}

**Patch Score Range:** {min_patch:.4f} - {max_patch:.4f}
**Weakest Patch:** #{min_idx} (row {min_idx // 7}, col {min_idx % 7})

**Interpretation:**
- Score > 0.6: Good alignment
- Score < 0.5: Potential issue
"""
        return heatmap, info
    except Exception as e:
        return None, f"Error: {str(e)}"


def create_live_interface():
    """Create live scoring interface (requires GPU)."""
    with gr.Blocks() as interface:
        gr.Markdown("# TiPAI Live Scoring")
        gr.Markdown("Upload an image and enter a prompt to see faithfulness heatmap.")

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Upload Image", type="pil")
                prompt_input = gr.Textbox(label="Prompt", placeholder="a red car on the beach")
                score_btn = gr.Button("Score Image", variant="primary")

            with gr.Column():
                heatmap_output = gr.Image(label="Faithfulness Heatmap", type="pil")
                info_output = gr.Markdown("")

        score_btn.click(fn=live_score, inputs=[input_image, prompt_input], outputs=[heatmap_output, info_output])

    return interface


# ============================================================================
# Main
# ============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Enable live scoring (requires GPU)")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    config = Config()

    # Ensure dataset is available (download from HF if not local)
    from utils.hf_utils import ensure_dataset_available
    if not ensure_dataset_available(config.IMAGES_DIR):
        print("ERROR: Could not get dataset locally or from HuggingFace.")
        return

    if args.live:
        print("Starting LIVE mode (requires CUDA)...")
        interface = create_live_interface()
    else:
        print("Starting BROWSE mode...")
        interface = create_browse_interface(config)

    interface.launch(server_port=args.port, share=False)


if __name__ == "__main__":
    main()
