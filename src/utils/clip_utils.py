"""
clip_utils.py
CLIP utilities for scoring and Grad-CAM attention visualization.

Used for:
- Signal 1: Per-object CLIP scores
- Signal 2: CLIP Grad-CAM attention maps
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Optional

import open_clip


# ============================================================================
# Global Model Cache
# ============================================================================
_CLIP_CACHE = {
    "model": None,
    "preprocess": None,
    "tokenizer": None,
    "device": None
}


def load_clip(model_name: str = "ViT-L-14", pretrained: str = "openai", device: str = "cuda"):
    """Load CLIP model (cached)."""
    global _CLIP_CACHE

    if _CLIP_CACHE["model"] is not None and _CLIP_CACHE["device"] == device:
        return _CLIP_CACHE["model"], _CLIP_CACHE["preprocess"], _CLIP_CACHE["tokenizer"]

    assert torch.cuda.is_available(), "CUDA required!"

    print(f"Loading CLIP {model_name} on {device}...")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    _CLIP_CACHE["model"] = model
    _CLIP_CACHE["preprocess"] = preprocess
    _CLIP_CACHE["tokenizer"] = tokenizer
    _CLIP_CACHE["device"] = device

    return model, preprocess, tokenizer


def get_text_embedding(text: str, model=None, tokenizer=None, device: str = "cuda") -> torch.Tensor:
    """Get normalized CLIP text embedding."""
    if model is None:
        model, _, tokenizer = load_clip(device=device)

    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        text_emb = model.encode_text(tokens)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    return text_emb


def get_image_embedding(image: Image.Image, model=None, preprocess=None, device: str = "cuda") -> torch.Tensor:
    """Get normalized CLIP image embedding."""
    if model is None:
        model, preprocess, _ = load_clip(device=device)

    img_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        img_emb = model.encode_image(img_tensor)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
    return img_emb


def compute_clip_score(image: Image.Image, text: str, device: str = "cuda") -> float:
    """
    Compute CLIP similarity score between image and text.

    Returns: Score in [0, 1] range (scaled from cosine similarity)
    """
    model, preprocess, tokenizer = load_clip(device=device)

    img_emb = get_image_embedding(image, model, preprocess, device)
    text_emb = get_text_embedding(text, model, tokenizer, device)

    similarity = (text_emb @ img_emb.T).item()
    # Scale from [-1, 1] to [0, 1]
    return (similarity + 1) / 2


def compute_clip_scores_batch(images: list[Image.Image], text: str, device: str = "cuda") -> list[float]:
    """Compute CLIP scores for multiple images against same text."""
    model, preprocess, tokenizer = load_clip(device=device)

    # Batch process images
    img_tensors = torch.stack([preprocess(img) for img in images]).to(device)

    with torch.no_grad():
        img_embs = model.encode_image(img_tensors)
        img_embs = img_embs / img_embs.norm(dim=-1, keepdim=True)

        text_emb = get_text_embedding(text, model, tokenizer, device)
        similarities = (text_emb @ img_embs.T).squeeze()

    # Scale to [0, 1]
    scores = ((similarities + 1) / 2).cpu().tolist()
    return scores if isinstance(scores, list) else [scores]


# ============================================================================
# CLIP Grad-CAM
# ============================================================================
class CLIPGradCAM:
    """
    Compute Grad-CAM attention maps for CLIP ViT models.

    Shows which image regions CLIP focuses on when computing text-image similarity.
    """

    def __init__(self, model, preprocess, device: str = "cuda"):
        self.model = model
        self.preprocess = preprocess
        self.device = device

        # Storage for hooks
        self.activations = None
        self.gradients = None

        # Register hooks on the last transformer block's output
        self._register_hooks()

    def _register_hooks(self):
        """Register forward/backward hooks on visual encoder."""
        # For ViT models, hook into the last transformer block
        visual = self.model.visual

        # Find the transformer blocks
        if hasattr(visual, 'transformer'):
            # OpenCLIP ViT structure
            target_layer = visual.transformer.resblocks[-1]
        elif hasattr(visual, 'blocks'):
            # Alternative structure
            target_layer = visual.blocks[-1]
        else:
            raise ValueError("Cannot find transformer blocks in CLIP visual encoder")

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def compute_gradcam(self, image: Image.Image, text: str) -> np.ndarray:
        """
        Compute Grad-CAM attention map for image-text pair.

        Returns:
            np.ndarray: HxW attention map (same size as input image)
        """
        # Prepare inputs
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        img_tensor.requires_grad = True

        tokens = open_clip.get_tokenizer("ViT-L-14")([text]).to(self.device)

        # Forward pass
        img_emb = self.model.encode_image(img_tensor)
        text_emb = self.model.encode_text(tokens)

        # Normalize
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

        # Compute similarity
        similarity = (img_emb @ text_emb.T).squeeze()

        # Backward pass
        self.model.zero_grad()
        similarity.backward()

        # Compute Grad-CAM
        if self.gradients is None or self.activations is None:
            print("Warning: Gradients or activations not captured")
            return np.ones((image.height, image.width)) * 0.5

        # Activations shape: [1, num_patches+1, dim] (includes CLS token)
        # Gradients shape: same

        # Remove CLS token
        activations = self.activations[:, 1:, :]  # [1, num_patches, dim]
        gradients = self.gradients[:, 1:, :]

        # Global average pooling of gradients
        weights = gradients.mean(dim=-1, keepdim=True)  # [1, num_patches, 1]

        # Weighted combination
        cam = (weights * activations).sum(dim=-1)  # [1, num_patches]

        # Reshape to 2D grid
        # For ViT-L/14 with 224x224 input: 16x16 patches
        num_patches = cam.shape[1]
        grid_size = int(np.sqrt(num_patches))

        cam = cam.reshape(1, grid_size, grid_size)

        # ReLU (only positive contributions)
        cam = F.relu(cam)

        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to image size
        cam = F.interpolate(
            cam.unsqueeze(0),
            size=(image.height, image.width),
            mode='bilinear',
            align_corners=False
        ).squeeze().cpu().numpy()

        return cam


def compute_gradcam(image: Image.Image, text: str, device: str = "cuda") -> np.ndarray:
    """
    Compute CLIP Grad-CAM attention map.

    Args:
        image: PIL Image
        text: Text prompt

    Returns:
        np.ndarray: HxW attention map in [0, 1] range
    """
    model, preprocess, _ = load_clip(device=device)
    gradcam = CLIPGradCAM(model, preprocess, device)
    return gradcam.compute_gradcam(image, text)


# ============================================================================
# Object Classification with CLIP
# ============================================================================
def classify_object(
    image: Image.Image,
    candidate_labels: list[str],
    device: str = "cuda"
) -> tuple[str, float]:
    """
    Classify an image (or cropped object) into one of the candidate labels.

    Returns: (best_label, confidence_score)
    """
    model, preprocess, tokenizer = load_clip(device=device)

    # Prepare prompts for each label
    prompts = [f"a photo of a {label}" for label in candidate_labels]

    # Get embeddings
    img_emb = get_image_embedding(image, model, preprocess, device)

    tokens = tokenizer(prompts).to(device)
    with torch.no_grad():
        text_embs = model.encode_text(tokens)
        text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)

    # Compute similarities
    similarities = (img_emb @ text_embs.T).squeeze()

    # Softmax for probabilities
    probs = F.softmax(similarities * 100, dim=-1)  # Temperature scaling

    best_idx = probs.argmax().item()
    return candidate_labels[best_idx], probs[best_idx].item()


def detect_color(image: Image.Image, device: str = "cuda") -> tuple[str, float]:
    """Detect the dominant color of an object in an image."""
    colors = ["red", "blue", "green", "yellow", "orange", "purple", "pink",
              "black", "white", "brown", "gray", "silver", "gold"]
    return classify_object(image, colors, device)


# ============================================================================
# Test
# ============================================================================
if __name__ == "__main__":
    # Test basic scoring
    print("Testing CLIP utilities...")

    # Create a simple test image (red square)
    test_img = Image.new("RGB", (224, 224), color="red")

    score_red = compute_clip_score(test_img, "a red square")
    score_blue = compute_clip_score(test_img, "a blue square")

    print(f"Red square vs 'a red square': {score_red:.4f}")
    print(f"Red square vs 'a blue square': {score_blue:.4f}")

    # Test Grad-CAM
    print("\nTesting Grad-CAM...")
    gradcam = compute_gradcam(test_img, "a red square")
    print(f"Grad-CAM shape: {gradcam.shape}")
    print(f"Grad-CAM range: [{gradcam.min():.4f}, {gradcam.max():.4f}]")
