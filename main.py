# main.py — LeafWise FastAPI Backend (with XAI endpoint)

import os, io, json, base64, torch, joblib
import numpy as np
from PIL import Image
from pathlib import Path
from huggingface_hub import hf_hub_download
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="LeafWise API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your Vercel URL after deploy e.g. ["https://leafwise.vercel.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_REPO_ID  = os.environ.get("HF_REPO_ID", "Hru-Code/Leaf-Wise")
HF_TOKEN    = os.environ.get("HF_TOKEN", None)
MODELS_DIR  = Path("./models_cache")
MODELS_DIR.mkdir(exist_ok=True)
NUM_CLASSES = 75

# Global holders
beit_model       = None
beit_processor   = None
class_names      = None
text_model       = None
tfidf_vectorizer = None
label_encoder_txt = None
plant_knowledge  = {}


def download_from_hf(filename: str) -> str:
    local = MODELS_DIR / filename
    if not local.exists():
        print(f"⬇️  Downloading {filename} ...")
        hf_hub_download(repo_id=HF_REPO_ID, filename=filename,
                        local_dir=str(MODELS_DIR), token=HF_TOKEN)
    print(f"✅ {filename}")
    return str(local)


@app.on_event("startup")
async def load_all_models():
    global beit_model, beit_processor, class_names
    global text_model, tfidf_vectorizer, label_encoder_txt, plant_knowledge

    from transformers import BeitForImageClassification, BeitImageProcessor, BeitConfig

    checkpoint_path  = download_from_hf("beit_best_checkpoint.pth")
    download_from_hf("beit_classifier_head.pth")
    label_enc_path   = download_from_hf("label_encoder.json")
    tfidf_path       = download_from_hf("tfidf_vectorizer.pkl")
    lr_model_path    = download_from_hf("logistic_regression_metadata_model.pkl")
    label_enc_txt_path = download_from_hf("label_encoder_txt.pkl")

    # BEiT
    beit_processor = BeitImageProcessor.from_pretrained("microsoft/beit-base-patch16-224")
    config = BeitConfig.from_pretrained("microsoft/beit-base-patch16-224")
    config.num_labels = NUM_CLASSES
    beit_model = BeitForImageClassification(config)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    try:
        beit_model.load_state_dict(ckpt["model_state_dict"], strict=True)
    except Exception as e:
        print(f"⚠️  strict=False fallback: {e}")
        beit_model.load_state_dict(ckpt["model_state_dict"], strict=False)
    beit_model.eval()

    with open(label_enc_path) as f:
        c2i = json.load(f)
    idx_to_class = {v: k for k, v in c2i.items()}
    class_names = [idx_to_class[i] for i in range(NUM_CLASSES)]
    print("✅ BEiT loaded")

    text_model        = joblib.load(lr_model_path)
    tfidf_vectorizer  = joblib.load(tfidf_path)
    label_encoder_txt = joblib.load(label_enc_txt_path)
    print("✅ Text model loaded")

    kp = Path("plant_knowledge.json")
    if kp.exists():
        with open(kp) as f:
            plant_knowledge = json.load(f)
        print("✅ Knowledge base loaded")
    else:
        print("⚠️  plant_knowledge.json not found")

    print("🌿 All models ready!")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _predict_image(img_pil):
    inputs = beit_processor(images=img_pil, return_tensors="pt")
    with torch.no_grad():
        out = beit_model(**inputs)
        probs = torch.nn.functional.softmax(out.logits, dim=1).cpu().numpy().flatten()
    return class_names[np.argmax(probs)], probs.tolist()

def _predict_text(text):
    X = tfidf_vectorizer.transform([text])
    probs = text_model.predict_proba(X).flatten()
    label = label_encoder_txt.inverse_transform([np.argmax(probs)])[0]
    return label, probs.tolist()

def _fuse(ip, tp, alpha=0.75):
    ip, tp = np.array(ip), np.array(tp)
    n = min(len(ip), len(tp))
    fused = alpha * ip[:n] + (1 - alpha) * tp[:n]
    return class_names[np.argmax(fused)], fused.tolist()

def _pil_to_b64(img_pil) -> str:
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "LeafWise API 🌿"}

@app.get("/health")
def health():
    return {
        "status": "ready" if beit_model else "loading",
        "beit_loaded": beit_model is not None,
        "text_model_loaded": text_model is not None,
    }


@app.post("/predict")
async def predict(
    image:    UploadFile = File(None),
    metadata: str        = Form(None),
    language: str        = Form("English"),
):
    if not image and not metadata:
        raise HTTPException(400, "Provide at least an image or metadata.")

    image_pred = image_probs = text_pred = text_probs = None

    if image:
        raw = await image.read()
        img_pil = Image.open(io.BytesIO(raw)).convert("RGB")
        image_pred, image_probs = _predict_image(img_pil)

    if metadata:
        text_pred, text_probs = _predict_text(metadata)

    if image_probs and text_probs:
        final_pred, fused_probs = _fuse(image_probs, text_probs)
        confidence = float(max(fused_probs))
    elif image_probs:
        final_pred, confidence = image_pred, float(max(image_probs))
    else:
        final_pred, confidence = text_pred, float(max(text_probs))

    advisory = plant_knowledge.get(final_pred, {})

    probs_arr = np.array(image_probs or text_probs)
    top5_idx  = probs_arr.argsort()[-5:][::-1]
    top5 = [{"plant": class_names[i], "confidence": round(float(probs_arr[i]) * 100, 2)} for i in top5_idx]

    return JSONResponse({
        "prediction":  final_pred,
        "confidence":  round(confidence * 100, 2),
        "image_pred":  image_pred,
        "text_pred":   text_pred,
        "top5":        top5,
        "advisory":    advisory,
        "language":    language,
    })


@app.post("/xai")
async def xai_endpoint(image: UploadFile = File(...)):
    """Returns base64 encoded Grad-CAM, LIME, and LRP images."""
    raw     = await image.read()
    img_pil = Image.open(io.BytesIO(raw)).convert("RGB")
    result  = {}

    # Grad-CAM
    try:
        from camv2 import generate_gradcam
        gradcam_img, _ = generate_gradcam(img_pil, beit_model, device="cpu")
        result["gradcam"] = _pil_to_b64(gradcam_img.convert("RGB"))
    except Exception as e:
        print(f"Grad-CAM failed: {e}")
        result["gradcam"] = None

    # LIME Image
    try:
        from lime_image_explainer import explain_with_lime_image
        lime_img = explain_with_lime_image(img_pil, beit_model, beit_processor, "cpu")
        result["lime"] = _pil_to_b64(lime_img)
    except Exception as e:
        print(f"LIME failed: {e}")
        result["lime"] = None

    # LRP
    try:
        from lrp_image_explainer import generate_lrp_image
        lrp_img = generate_lrp_image(img_pil, beit_model, "cpu")
        result["lrp"] = _pil_to_b64(lrp_img)
    except Exception as e:
        print(f"LRP failed: {e}")
        result["lrp"] = None

    return JSONResponse(result)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
