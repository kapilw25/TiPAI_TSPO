"""
m08_demo_app.py
Gradio demo for TiPAI - Object-Level Faithfulness with Risk Maps.

Two modes:
1. Browse: View pre-generated pairs with risk maps from database
2. Live: Score new images on-the-fly (requires GPU)

Uses signals from m02 and risk_maps from m03.

REQUIRES: CUDA GPU for live mode

Run:
    python -u src/m08_demo_app.py --port 7860 2>&1 | tee logs/m08_$(date +%Y%m%d_%H%M%S).log
    python -u src/m08_demo_app.py --live --port 7860 2>&1 | tee logs/m08_$(date +%Y%m%d_%H%M%S).log
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
    # Note: image_path comes from DB (model-specific dirs handled by m01)
    RISK_MAPS_DIR = PROJECT_ROOT / "outputs" / "m03_risk_maps"
    PAIRS_DIR = PROJECT_ROOT / "outputs" / "m04_pairs"

    CLIP_MODEL = "ViT-L-14"
    CLIP_PRETRAINED = "openai"
    OVERLAY_ALPHA = 0.7


# ============================================================================
# Database Functions
# ============================================================================
def get_all_pairs(config: Config) -> list:
    """Get all pairs from database."""
    conn = sqlite3.connect(str(config.DB_PATH))
    cursor = conn.cursor()
    cursor.execute("""
        SELECT pair_id, chosen_prompt, chosen_image_id, rejected_image_id,
               chosen_score, rejected_score, comparison_path
        FROM pairs
        ORDER BY score_gap DESC
    """)
    results = cursor.fetchall()
    conn.close()
    return results


def get_pair_details(config: Config, pair_id: str) -> dict:
    """Get detailed info for a specific pair using signals and risk_maps."""
    conn = sqlite3.connect(str(config.DB_PATH))
    cursor = conn.cursor()

    # Get pair info
    cursor.execute("""
        SELECT chosen_prompt, chosen_image_id, rejected_image_id, chosen_score, rejected_score, failure_type
        FROM pairs WHERE pair_id = ?
    """, (pair_id,))
    pair = cursor.fetchone()

    if not pair:
        conn.close()
        return None

    prompt, chosen_id, rejected_id, chosen_score, rejected_score, failure_type = pair

    # Get signals data (num_gaps, objects)
    cursor.execute("SELECT num_gaps, objects_json FROM signals WHERE image_id = ?", (chosen_id,))
    chosen_sig = cursor.fetchone()
    chosen_gaps = chosen_sig[0] if chosen_sig else 0

    cursor.execute("SELECT num_gaps, objects_json, gaps_json FROM signals WHERE image_id = ?", (rejected_id,))
    rejected_sig = cursor.fetchone()
    rejected_gaps = rejected_sig[0] if rejected_sig else 0
    rejected_gaps_json = rejected_sig[2] if rejected_sig else "[]"

    # Get image paths
    cursor.execute("SELECT image_path FROM images WHERE image_id = ?", (chosen_id,))
    chosen_path = cursor.fetchone()[0]

    cursor.execute("SELECT image_path FROM images WHERE image_id = ?", (rejected_id,))
    rejected_path = cursor.fetchone()[0]

    # Get risk map paths
    cursor.execute("SELECT risk_map_path FROM risk_maps WHERE image_id = ?", (chosen_id,))
    chosen_risk_row = cursor.fetchone()
    chosen_risk_path = chosen_risk_row[0] if chosen_risk_row else None

    cursor.execute("SELECT risk_map_path FROM risk_maps WHERE image_id = ?", (rejected_id,))
    rejected_risk_row = cursor.fetchone()
    rejected_risk_path = rejected_risk_row[0] if rejected_risk_row else None

    conn.close()

    return {
        "prompt": prompt,
        "chosen_id": chosen_id,
        "rejected_id": rejected_id,
        "chosen_score": chosen_score,
        "rejected_score": rejected_score,
        "failure_type": failure_type,
        "chosen_gaps": chosen_gaps,
        "rejected_gaps": rejected_gaps,
        "rejected_gaps_detail": json.loads(rejected_gaps_json) if rejected_gaps_json else [],
        "chosen_path": config.PROJECT_ROOT / chosen_path,
        "rejected_path": config.PROJECT_ROOT / rejected_path,
        "chosen_risk_path": config.PROJECT_ROOT / chosen_risk_path if chosen_risk_path else None,
        "rejected_risk_path": config.PROJECT_ROOT / rejected_risk_path if rejected_risk_path else None,
    }


# ============================================================================
# Helper Functions
# ============================================================================
def format_gaps_detail(gaps: list) -> str:
    """Format gaps list for display."""
    if not gaps:
        return "None"
    details = []
    for gap in gaps:
        issue = gap.get('issue', 'unknown')
        obj = gap.get('object_noun', 'object')
        expected = gap.get('expected', '')
        found = gap.get('found', '')
        if issue == 'wrong_color':
            details.append(f"{obj}: expected {expected}, found {found}")
        elif issue == 'missing':
            details.append(f"{obj}: missing")
        else:
            details.append(f"{obj}: {issue}")
    return "; ".join(details)


def create_live_heatmap(image: Image.Image, patch_scores: list, grid_size: int = 7, alpha: float = 0.5) -> Image.Image:
    """Create heatmap overlay for live mode (patch-based CLIP scoring)."""
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
    """Browse pre-generated pairs with risk maps."""
    if not pair_selection:
        return None, None, None, None, "Select a pair"

    pair_id = pair_selection.split(" | ")[0]
    details = get_pair_details(config, pair_id)

    if not details:
        return None, None, None, None, "Pair not found"

    # Load images
    chosen_img = Image.open(details["chosen_path"]) if details["chosen_path"].exists() else None
    rejected_img = Image.open(details["rejected_path"]) if details["rejected_path"].exists() else None

    # Load risk maps
    chosen_risk = None
    rejected_risk = None
    if details["chosen_risk_path"] and details["chosen_risk_path"].exists():
        chosen_risk = Image.open(details["chosen_risk_path"])
    if details["rejected_risk_path"] and details["rejected_risk_path"].exists():
        rejected_risk = Image.open(details["rejected_risk_path"])

    # Format gaps detail
    gaps_detail = format_gaps_detail(details["rejected_gaps_detail"])

    failure_labels = {
        "attribute": "ATTRIBUTE (color/size/material)",
        "object": "OBJECT (wrong object)",
        "spatial": "SPATIAL/COUNT"
    }

    info = f"""**Prompt:** {details['prompt']}

**Failure Type:** {failure_labels.get(details['failure_type'], details['failure_type'])}

**Chosen Score:** {details['chosen_score']:.4f} (gaps: {details['chosen_gaps']})
**Rejected Score:** {details['rejected_score']:.4f} (gaps: {details['rejected_gaps']})
**Score Gap:** {details['chosen_score'] - details['rejected_score']:.4f}

**Detected Issues:** {gaps_detail}

**Interpretation:**
- RED regions in risk map = issues detected (wrong color, missing object, low score)
- No overlay = correct objects matching baseline prompt
"""
    return chosen_img, rejected_img, chosen_risk, rejected_risk, info


def create_browse_interface(config: Config):
    """Create browse interface for pre-generated pairs with risk maps."""
    pairs = get_all_pairs(config)

    if not pairs:
        return gr.Markdown("No pairs found. Run the pipeline first (m01-m04).")

    pair_choices = [f"{p[0]} | {p[1][:50]}... (gap: {p[5]-p[4]:.3f})" for p in pairs]

    with gr.Blocks() as interface:
        gr.Markdown("# TiPAI: Object-Level Faithfulness Demo")
        gr.Markdown("Select a pair to see chosen vs rejected images with risk maps (RED = issues).")

        pair_dropdown = gr.Dropdown(choices=pair_choices, label="Select Pair", value=pair_choices[0] if pair_choices else None)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### CHOSEN (v0_original)")
                chosen_img = gr.Image(label="Chosen Image", type="pil")
                chosen_risk = gr.Image(label="Chosen Risk Map", type="pil")

            with gr.Column():
                gr.Markdown("### REJECTED (variation)")
                rejected_img = gr.Image(label="Rejected Image", type="pil")
                rejected_risk = gr.Image(label="Rejected Risk Map", type="pil")

        info_box = gr.Markdown("")

        pair_dropdown.change(
            fn=lambda x: browse_pairs(x, config),
            inputs=[pair_dropdown],
            outputs=[chosen_img, rejected_img, chosen_risk, rejected_risk, info_box]
        )

        # Load first pair on start
        if pair_choices:
            interface.load(
                fn=lambda: browse_pairs(pair_choices[0], config),
                outputs=[chosen_img, rejected_img, chosen_risk, rejected_risk, info_box]
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

    heatmap = create_live_heatmap(image, patch_scores)
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

    # Check if database exists
    if not config.DB_PATH.exists():
        print(f"ERROR: Database not found at {config.DB_PATH}")
        print("Run the pipeline first: m01 -> m02 -> m03 -> m04")
        return

    if args.live:
        print("Starting LIVE mode (requires CUDA)...")
        print("Note: Live mode uses patch-based CLIP scoring for quick testing")
        interface = create_live_interface()
    else:
        print("Starting BROWSE mode...")
        interface = create_browse_interface(config)

    interface.launch(server_port=args.port, share=False)


if __name__ == "__main__":
    main()
