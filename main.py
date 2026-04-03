# main.py — LeafWise FastAPI Backend

import os
import io
import json
import torch
import joblib
import numpy as np
from PIL import Image
from pathlib import Path
from huggingface_hub import hf_hub_download
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# ─── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(title="LeafWise API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your Vercel URL in production e.g. ["https://leafwise.vercel.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── HuggingFace Config ───────────────────────────────────────────────────────
HF_REPO_ID = os.environ.get("HF_REPO_ID", "Hru-Code/Leaf-Wise")
HF_TOKEN   = os.environ.get("HF_TOKEN", None)
MODELS_DIR = Path("./models_cache")
MODELS_DIR.mkdir(exist_ok=True)

NUM_CLASSES = 75

# ─── Global model holders ─────────────────────────────────────────────────────
beit_model      = None
beit_processor  = None
class_names     = None
text_model      = None
tfidf_vectorizer = None
label_encoder_txt = None
plant_knowledge = None


def download_from_hf(filename: str) -> str:
    """Download a file from HuggingFace Hub and return local path."""
    local_path = MODELS_DIR / filename
    if not local_path.exists():
        print(f"⬇️  Downloading {filename} from HuggingFace...")
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            local_dir=str(MODELS_DIR),
            token=HF_TOKEN,
        )
        print(f"✅ Downloaded {filename}")
    else:
        print(f"✅ Using cached {filename}")
    return str(local_path)


@app.on_event("startup")
async def load_all_models():
    """Download model files from HuggingFace and load into memory at startup."""
    global beit_model, beit_processor, class_names
    global text_model, tfidf_vectorizer, label_encoder_txt
    global plant_knowledge

    print("🚀 Starting model loading...")

    from transformers import BeitForImageClassification, BeitImageProcessor, BeitConfig

    # ── Download all model artefacts from HF ──────────────────────────────────
    checkpoint_path     = download_from_hf("beit_best_checkpoint.pth")
    classifier_head_path = download_from_hf("beit_classifier_head.pth")
    label_encoder_path  = download_from_hf("label_encoder.json")
    tfidf_path          = download_from_hf("tfidf_vectorizer.pkl")
    lr_model_path       = download_from_hf("logistic_regression_metadata_model.pkl")
    label_enc_txt_path  = download_from_hf("label_encoder_txt.pkl")

    # ── BEiT Image Model ───────────────────────────────────────────────────────
    beit_processor = BeitImageProcessor.from_pretrained("microsoft/beit-base-patch16-224")
    config = BeitConfig.from_pretrained("microsoft/beit-base-patch16-224")
    config.num_labels = NUM_CLASSES

    beit_model = BeitForImageClassification(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    try:
        beit_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except Exception as e:
        print(f"⚠️  Falling back to strict=False: {e}")
        beit_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    beit_model.eval()

    with open(label_encoder_path, "r") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(NUM_CLASSES)]
    print("✅ BEiT model loaded")

    # ── Text / Metadata Model ──────────────────────────────────────────────────
    text_model        = joblib.load(lr_model_path)
    tfidf_vectorizer  = joblib.load(tfidf_path)
    label_encoder_txt = joblib.load(label_enc_txt_path)
    print("✅ Text model loaded")

    # ── Plant Knowledge Base ───────────────────────────────────────────────────
    knowledge_path = Path("plant_knowledge.json")
    if knowledge_path.exists():
        with open(knowledge_path, "r") as f:
            plant_knowledge = json.load(f)
        print("✅ Plant knowledge base loaded")
    else:
        plant_knowledge = {}
        print("⚠️  plant_knowledge.json not found — advisory will be empty")

    print("🌿 All models ready!")


# ─── Helper: Image Prediction ─────────────────────────────────────────────────
def predict_image(image_pil):
    inputs = beit_processor(images=image_pil, return_tensors="pt")
    with torch.no_grad():
        outputs = beit_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy().flatten()
    top_class = class_names[np.argmax(probs)]
    return top_class, probs.tolist()


# ─── Helper: Text Prediction ──────────────────────────────────────────────────
def predict_text(text: str):
    X = tfidf_vectorizer.transform([text])
    probs = text_model.predict_proba(X).flatten()
    top_class = label_encoder_txt.inverse_transform([np.argmax(probs)])[0]
    return top_class, probs.tolist()


# ─── Helper: Fused Prediction ─────────────────────────────────────────────────
def fuse_predictions(image_probs, text_probs, alpha=0.75):
    ip = np.array(image_probs)
    tp = np.array(text_probs)
    min_len = min(len(ip), len(tp))
    fused = alpha * ip[:min_len] + (1 - alpha) * tp[:min_len]
    top_class = class_names[np.argmax(fused)]
    return top_class, fused.tolist()


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "LeafWise API is running 🌿"}


@app.get("/health")
def health():
    return {
        "status": "ready" if beit_model is not None else "loading",
        "beit_loaded": beit_model is not None,
        "text_model_loaded": text_model is not None,
    }


@app.post("/predict")
async def predict(
    image: UploadFile = File(None),
    metadata: str     = Form(None),
    language: str     = Form("en"),
):
    """
    Main prediction endpoint.
    - image    : leaf image file (jpg/png)
    - metadata : optional text description of the plant
    - language : target language code for translation (default 'en' = no translation)
    """
    if not image and not metadata:
        raise HTTPException(status_code=400, detail="Provide at least an image or metadata text.")

    image_pred   = None
    image_probs  = None
    text_pred    = None
    text_probs   = None
    final_pred   = None
    confidence   = None

    # ── Image Prediction ───────────────────────────────────────────────────────
    if image:
        try:
            contents  = await image.read()
            img_pil   = Image.open(io.BytesIO(contents)).convert("RGB")
            image_pred, image_probs = predict_image(img_pil)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Image prediction failed: {e}")

    # ── Text Prediction ────────────────────────────────────────────────────────
    if metadata:
        try:
            text_pred, text_probs = predict_text(metadata)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Text prediction failed: {e}")

    # ── Fusion or fallback ─────────────────────────────────────────────────────
    if image_probs and text_probs:
        final_pred, fused_probs = fuse_predictions(image_probs, text_probs)
        confidence = float(max(fused_probs))
    elif image_probs:
        final_pred = image_pred
        confidence = float(max(image_probs))
    else:
        final_pred = text_pred
        confidence = float(max(text_probs))

    # ── Advisory ──────────────────────────────────────────────────────────────
    advisory = plant_knowledge.get(final_pred, {})

    # ── Top 5 ─────────────────────────────────────────────────────────────────
    probs_for_top5 = np.array(image_probs or text_probs)
    top5_idx = probs_for_top5.argsort()[-5:][::-1]
    top5 = [
        {"plant": class_names[i], "confidence": round(float(probs_for_top5[i]) * 100, 2)}
        for i in top5_idx
    ]

    return JSONResponse({
        "prediction":    final_pred,
        "confidence":    round(confidence * 100, 2),
        "image_pred":    image_pred,
        "text_pred":     text_pred,
        "top5":          top5,
        "advisory":      advisory,
        "language":      language,
    })


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
