# lrp_image_explainer.py

import torch
import numpy as np
from PIL import Image
import cv2
import json
from pathlib import Path
from transformers import BeitImageProcessor, BeitForImageClassification

NUM_CLASSES = 75  # Fallback if label_encoder.json is unavailable


def _resolve_path(path_like: str) -> Path:
    candidate = Path(path_like)
    if candidate.exists():
        return candidate

    module_dir = Path(__file__).resolve().parent
    search_roots = [
        module_dir,
        module_dir.parent,
        module_dir.parent.parent,
        Path.cwd(),
    ]
    for base in search_roots:
        resolved = base / candidate
        if resolved.exists():
            return resolved
    raise FileNotFoundError(f"Could not locate artifact '{path_like}'.")


def _infer_num_classes() -> int:
    label_encoder_candidates = [
        "checkpoints/label_encoder.json",
        "LeafWise_optimised/checkpoints/label_encoder.json",
    ]
    for candidate in label_encoder_candidates:
        try:
            with open(_resolve_path(candidate), "r", encoding="utf-8") as f:
                label_mapping = json.load(f)
            if label_mapping:
                return len(label_mapping)
        except FileNotFoundError:
            continue
    return NUM_CLASSES

def generate_lrp_image(pil_image, model=None, device="cpu"):
    if model is None:
        num_classes = _infer_num_classes()
        model = BeitForImageClassification.from_pretrained(
            "microsoft/beit-base-patch16-224",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        checkpoint = torch.load(_resolve_path("checkpoints/beit_best_checkpoint.pth"), map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        model.to(device)

    model.eval()
    processor = BeitImageProcessor.from_pretrained("microsoft/beit-base-patch16-224")
    inputs = processor(images=pil_image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    pixel_values.requires_grad_(True)

    outputs = model(pixel_values)

    # FIX: argmax returns a 1-D tensor of shape (1,); use .item() to get a Python int
    # so that outputs.logits[0, pred_class] is a scalar — required for .backward()
    pred_class = outputs.logits.argmax(dim=-1).item()
    score = outputs.logits[0, pred_class]   # scalar ✅

    model.zero_grad()
    score.backward()

    # Aggregate gradient across colour channels → saliency map
    relevance = pixel_values.grad[0].abs().sum(dim=0).cpu().numpy()
    relevance = (relevance - relevance.min()) / (relevance.max() - relevance.min() + 1e-6)
    relevance = np.uint8(255 * relevance)

    heatmap = cv2.applyColorMap(relevance, cv2.COLORMAP_JET)
    # FIX: cv2 returns BGR; convert to RGB before handing to PIL
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap_resized = cv2.resize(heatmap_rgb, pil_image.size)

    overlay = Image.blend(
        pil_image.convert("RGBA"),
        Image.fromarray(heatmap_resized).convert("RGBA"),
        alpha=0.5
    )
    return overlay
