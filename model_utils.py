# model_utils.py

import torch
import torch.nn.functional as F
import joblib
import numpy as np
import json
from pathlib import Path
from transformers import BeitForImageClassification, BeitImageProcessor, BeitConfig

# === CONFIGURATION ===
NUM_CLASSES = 75  # Fallback if label_encoder.json is unavailable

MODULE_DIR = Path(__file__).resolve().parent


def _resolve_path(path_like: str) -> Path:
    candidate = Path(path_like)
    if candidate.exists():
        return candidate

    search_roots = [
        MODULE_DIR,
        MODULE_DIR.parent,
        MODULE_DIR.parent.parent,
        Path.cwd(),
    ]

    for base in search_roots:
        resolved = base / candidate
        if resolved.exists():
            return resolved

    searched = [str(base / candidate) for base in search_roots]
    raise FileNotFoundError(f"Could not locate artifact '{path_like}'. Searched: {searched}")

# === LOAD BEiT MODEL FOR IMAGE CLASSIFICATION ===
def load_beit_model():
    processor = BeitImageProcessor.from_pretrained("microsoft/beit-base-patch16-224")

    label_encoder_path = _resolve_path("checkpoints/label_encoder.json")
    checkpoint_path = _resolve_path("checkpoints/beit_best_checkpoint.pth")

    with open(label_encoder_path, 'r', encoding='utf-8') as f:
        class_to_idx = json.load(f)

    num_classes = len(class_to_idx) or NUM_CLASSES

    config = BeitConfig.from_pretrained("microsoft/beit-base-patch16-224")
    config.num_labels = num_classes

    model = BeitForImageClassification(config)

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        print("✅ Loaded model with strict=True")
    except Exception as e:
        print("⚠️ Falling back to strict=False due to mismatch:", e)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    model.eval()

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(num_classes)]

    return model, processor, class_names

# === LOAD TEXT MODEL ===
def load_text_model(tfidf_path="checkpoints/tfidf_vectorizer.pkl",
                    model_path="checkpoints/logistic_regression_metadata_model.pkl",
                    label_encoder_txt_path="checkpoints/label_encoder_txt.pkl"):
    tfidf_vectorizer = joblib.load(_resolve_path(tfidf_path))
    model = joblib.load(_resolve_path(model_path))
    label_encoder_txt = joblib.load(_resolve_path(label_encoder_txt_path))
    return model, tfidf_vectorizer, label_encoder_txt

# === IMAGE PREDICTION ===
def predict_image(image_pil, model, processor, class_names, device):
    inputs = processor(images=image_pil, return_tensors="pt").to(device)
    model.to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1).cpu().numpy().flatten()
    top_class = class_names[np.argmax(probs)]
    return top_class, probs.tolist(), inputs['pixel_values']

# === TEXT PREDICTION ===
def predict_text(text, model, tfidf_vectorizer, label_encoder_txt):
    X = tfidf_vectorizer.transform([text])
    probs = model.predict_proba(X).flatten()
    top_class = label_encoder_txt.inverse_transform([np.argmax(probs)])[0]
    return top_class, probs.tolist()

# === FUSED PREDICTION ===
def fuse_predictions(image_probs, text_probs, class_names, alpha=0.75):
    """
    Combines image and text prediction probabilities using weighted fusion.

    Args:
        image_probs (list): Softmaxed image model probabilities
        text_probs (list): Text model probabilities
        class_names (list): Ordered list of class names
        alpha (float): Weight for image (alpha) vs text (1 - alpha)

    Returns:
        top_class (str), fused_probabilities (list)
    """
    image_probs = np.array(image_probs)
    text_probs = np.array(text_probs)

    # Align length in case text model has different number of classes
    min_len = min(len(image_probs), len(text_probs))
    image_probs = image_probs[:min_len]
    text_probs = text_probs[:min_len]

    fused = alpha * image_probs + (1 - alpha) * text_probs
    top_class = class_names[np.argmax(fused)]
    return top_class, fused.tolist()
