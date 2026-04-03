import csv
import json
import os
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, roc_curve, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize

try:
    import optuna
except ImportError:
    optuna = None


# === Config ===

def _discover_workspace_root() -> Path:
    current = Path(__file__).resolve().parent
    for candidate in [current, *current.parents]:
        if (candidate / "MetaData_Cleaned.csv").exists() and (candidate / "Indian Medicinal Leaves").exists():
            return candidate
    return current


WORKSPACE_ROOT = _discover_workspace_root()
DATA_PATH = WORKSPACE_ROOT / "MetaData_Cleaned.csv"
MODEL_PATH = WORKSPACE_ROOT / "checkpoints"
MODEL_PATH.mkdir(parents=True, exist_ok=True)
KNOWLEDGE_BASE_PATH = WORKSPACE_ROOT / "plant_knowledge.json"
if not KNOWLEDGE_BASE_PATH.exists():
    KNOWLEDGE_BASE_PATH = WORKSPACE_ROOT / "indian_herbal_plants (1).json"
TEXT_MODEL_NAME = "logistic_regression_metadata_model.pkl"
TFIDF_NAME = "tfidf_vectorizer.pkl"
ENCODER_NAME = "label_encoder_txt.pkl"
OPTUNA_BEST_NAME = "metadata_optuna_best_params.json"

RANDOM_SEED = int(os.getenv("LEAFWISE_SEED", "42"))
ENABLE_OPTUNA = os.getenv("LEAFWISE_ENABLE_OPTUNA", "0") == "1"
OPTUNA_TRIALS = int(os.getenv("LEAFWISE_TEXT_OPTUNA_TRIALS", "30"))
OPTUNA_TIMEOUT = int(os.getenv("LEAFWISE_TEXT_OPTUNA_TIMEOUT", "0"))


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _build_label_map() -> dict:
    with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as handle:
        knowledge = json.load(handle)

    label_map = {}
    for canonical_name, record in knowledge.items():
        aliases = [
            canonical_name,
            record.get("canonical_name", ""),
            record.get("plant_name", ""),
            record.get("common_name", ""),
            record.get("scientific_name", ""),
        ]
        raw_aliases = record.get("aliases", [])
        if isinstance(raw_aliases, list):
            aliases.extend(raw_aliases)

        for alias in aliases:
            alias_text = str(alias).strip()
            if alias_text:
                label_map[_normalize_name(alias_text)] = canonical_name
    return label_map


LABEL_MAP = _build_label_map()


def _canonicalize_plant_name(value: str) -> str:
    value_text = str(value).strip()
    if not value_text:
        return value_text
    return LABEL_MAP.get(_normalize_name(value_text), value_text)


# === Load dataset ===
with open(DATA_PATH, newline="", encoding="utf-8") as csv_file:
    reader = csv.DictReader(csv_file)
    fieldnames = list(reader.fieldnames or [])
    rows = list(reader)

plant_name_col = "Plant name"
for row in rows:
    row[plant_name_col] = _canonicalize_plant_name(row.get(plant_name_col, ""))

text_columns = [col for col in fieldnames if col != plant_name_col]
texts = [" ".join(str(row.get(col, "")).strip() for col in text_columns) for row in rows]
labels = [row[plant_name_col] for row in rows]

# === Encode labels ===
label_encoder_txt = LabelEncoder()
y = label_encoder_txt.fit_transform(labels)
n_classes = len(label_encoder_txt.classes_)
top_k = min(5, n_classes)

texts_train, texts_val, y_train, y_val = train_test_split(
    texts,
    y,
    test_size=0.15,
    stratify=y,
    random_state=RANDOM_SEED,
)


def _build_vectorizer(params: dict | None = None) -> TfidfVectorizer:
    params = params or {}
    return TfidfVectorizer(
        ngram_range=(1, int(params.get("ngram_max", 3))),
        max_features=int(params.get("max_features", 4000)),
        min_df=int(params.get("min_df", 2)),
    )


def _build_classifier(params: dict | None = None) -> LogisticRegression:
    params = params or {}
    class_weight = params.get("class_weight")
    if class_weight == "none":
        class_weight = None
    return LogisticRegression(
        C=float(params.get("C", 1.0)),
        max_iter=1000,
        solver="lbfgs",
        verbose=1,
        class_weight=class_weight,
    )


def _train_and_eval(params: dict | None = None):
    tfidf_local = _build_vectorizer(params)
    X_train_local = tfidf_local.fit_transform(texts_train)
    X_val_local = tfidf_local.transform(texts_val)

    model_local = _build_classifier(params)
    model_local.fit(X_train_local, y_train)

    y_pred_local = model_local.predict(X_val_local)
    y_proba_local = model_local.predict_proba(X_val_local)

    acc_local = accuracy_score(y_val, y_pred_local)
    topk_local = top_k_accuracy_score(y_val, y_proba_local, k=top_k)

    return model_local, tfidf_local, y_pred_local, y_proba_local, acc_local, topk_local


best_params = None
if ENABLE_OPTUNA:
    if optuna is None:
        raise RuntimeError("Optuna mode requested but optuna is not installed. Run: pip install optuna")

    print("🔎 Starting Optuna tuning for metadata model...")

    def objective(trial):
        params = {
            "C": trial.suggest_float("C", 1e-2, 20.0, log=True),
            "ngram_max": trial.suggest_int("ngram_max", 1, 3),
            "max_features": trial.suggest_int("max_features", 2000, 8000, step=500),
            "min_df": trial.suggest_int("min_df", 1, 3),
            "class_weight": trial.suggest_categorical("class_weight", ["none", "balanced"]),
        }
        _, _, _, _, _, topk_score = _train_and_eval(params)
        return topk_score

    study = optuna.create_study(direction="maximize")
    optimize_kwargs = {"n_trials": OPTUNA_TRIALS}
    if OPTUNA_TIMEOUT > 0:
        optimize_kwargs["timeout"] = OPTUNA_TIMEOUT
    study.optimize(objective, **optimize_kwargs)

    best_params = study.best_params
    with open(MODEL_PATH / OPTUNA_BEST_NAME, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "best_trial": study.best_trial.number,
                "best_value": study.best_value,
                "best_params": best_params,
                "metric": f"top_{top_k}_accuracy",
                "trials": len(study.trials),
            },
            handle,
            indent=2,
        )
    print(f"✅ Optuna best params: {best_params}")


model, tfidf, y_pred, y_proba, acc, topk = _train_and_eval(best_params)

print(f"\n✅ Top-1 Accuracy: {acc * 100:.2f}%")
print(f"✅ Top-{top_k} Accuracy: {topk * 100:.2f}%")

# === ROC Curve Plot ===
y_val_bin = label_binarize(y_val, classes=np.arange(n_classes))
fpr, tpr, roc_auc = {}, {}, {}

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_val_bin[:, i], y_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(22, 12))
for i in range(n_classes):
    plt.plot(
        fpr[i],
        tpr[i],
        lw=1.5,
        label=f"{label_encoder_txt.classes_[i]} (AUC = {roc_auc[i]:.2f})",
    )

plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False Positive Rate", fontsize=14)
plt.ylabel("True Positive Rate", fontsize=14)
plt.title("ROC Curves for All Classes (Logistic Regression)", fontsize=18)
plt.legend(loc="upper right", fontsize="x-small", ncol=3)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{MODEL_PATH}/text_model_roc.png", dpi=300)
plt.close()

print(f"\n📈 ROC curve saved to: {MODEL_PATH}/text_model_roc.png")

# === Save Models and Encoders ===
joblib.dump(model, os.path.join(MODEL_PATH, TEXT_MODEL_NAME))
joblib.dump(tfidf, os.path.join(MODEL_PATH, TFIDF_NAME))
joblib.dump(label_encoder_txt, os.path.join(MODEL_PATH, ENCODER_NAME))

print("✅ All model components saved.")