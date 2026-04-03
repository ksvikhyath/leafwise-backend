# explainers.py

import numpy as np
import streamlit as st
from io import BytesIO
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import eli5
from lime.lime_text import LimeTextExplainer
import spacy

# ============================================================
# === ELI5 Text Explanation ===
# ============================================================
def explain_text_with_eli5(text, model, tfidf_vectorizer, label_encoder):
    """
    Returns HTML explanation from ELI5 for a given text.
    """
    feature_names = tfidf_vectorizer.get_feature_names_out().tolist()
    try:
        explanation = eli5.explain_prediction(
            model,
            text,
            vec=tfidf_vectorizer,
            feature_names=feature_names
        )
        return eli5.format_as_html(explanation)
    except Exception as e:
        return f"<p><strong>ELI5 explanation failed:</strong> {str(e)}</p>"


# ============================================================
# === LIME Text Explanation ===
# ============================================================
def explain_text_with_lime(text, model, vectorizer, label_encoder=None):
    """
    Generates a LIME explanation plot for a text input using TF-IDF.
    Returns an HTML <img> string to embed in Streamlit.
    """
    # Use human-readable class names if a label encoder is provided
    if label_encoder is not None and hasattr(label_encoder, 'classes_'):
        class_names = list(label_encoder.classes_)
    elif hasattr(model, 'classes_'):
        # model.classes_ holds integer indices when LabelEncoder was used;
        # fall back to string representation so LIME labels are readable
        class_names = [str(c) for c in model.classes_]
    else:
        class_names = None

    def predict_proba(texts):
        X = vectorizer.transform(texts)
        return model.predict_proba(X)

    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(text, predict_proba, num_features=10)

    fig = exp.as_pyplot_figure()
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)

    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"<img src='data:image/png;base64,{img_base64}' style='max-width:100%;'/>"


# ============================================================
# === SHAP Text Explanation ===
# ============================================================
def explain_text_with_shap(text, model, tfidf_vectorizer):
    try:
        import shap
        X = tfidf_vectorizer.transform([text]).toarray()
        masker = shap.maskers.Independent(X)
        explainer = shap.Explainer(model.predict_proba, masker)
        shap_values = explainer(X)

        fig = plt.figure(figsize=(10, 5))
        shap.plots.bar(shap_values[0], max_display=10, show=False)

        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        return f'<img src="data:image/png;base64,{encoded}" alt="SHAP Text Explanation" style="max-width:100%;"/>'
    except Exception as e:
        return f"<p><strong>SHAP explanation failed:</strong> {str(e)}</p>"


# ============================================================
# === Anchor Text Explanation ===
# ============================================================
def explain_text_with_anchor(text, model, tfidf_vectorizer):
    """
    Generate Anchor explanation HTML for the input text.
    Compatible with both alibi < 0.7 and alibi >= 0.9 APIs.
    """
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        return "<p><strong>Anchor failed:</strong> spaCy model 'en_core_web_sm' not found. Run: <code>python -m spacy download en_core_web_sm</code></p>"

    def predictor(texts):
        X = tfidf_vectorizer.transform(texts)
        return model.predict(X)

    try:
        from alibi.explainers import AnchorText

        # Try new alibi (>= 0.9) keyword-only constructor first
        try:
            explainer = AnchorText(predictor=predictor, nlp=nlp)
        except TypeError:
            # Fall back to old positional-arg constructor (alibi < 0.7)
            explainer = AnchorText(predictor, nlp=nlp)

        explanation = explainer.explain(text)

        # ── Extract fields — handle both old and new alibi response structure ──
        try:
            # alibi >= 0.9: all data lives under explanation.data
            data = explanation.data
            anchor      = data.get('anchor', [])
            raw         = data.get('raw', {})
            prediction  = raw.get('prediction', predictor([text])[0])

            prec = data.get('precision', [0.0])
            precision = float(prec[0]) if isinstance(prec, (list, np.ndarray)) else float(prec)

            cov = data.get('coverage', [0.0])
            coverage = float(cov[0]) if isinstance(cov, (list, np.ndarray)) else float(cov)

        except (AttributeError, KeyError):
            # alibi < 0.7: attributes are direct properties
            anchor     = getattr(explanation, 'anchor', [])
            precision  = float(getattr(explanation, 'precision', 0.0))
            coverage   = float(getattr(explanation, 'coverage', 0.0))
            raw        = getattr(explanation, 'raw', {})
            prediction = raw.get('prediction', predictor([text])[0])

        anchor_str = ', '.join(anchor) if anchor else 'No decisive anchor found'

        return f"""
        <div style="font-family: monospace; white-space: pre-wrap; padding: 8px;">
        <h4>Anchor Explanation</h4>
        <p><strong>Prediction:</strong> {prediction}</p>
        <p><strong>Anchor (decisive features):</strong> {anchor_str}</p>
        <p><strong>Precision:</strong> {precision:.2f} &nbsp;
           <em>(how often these features produce this prediction)</em></p>
        <p><strong>Coverage:</strong> {coverage:.2f} &nbsp;
           <em>(fraction of similar inputs matching this anchor)</em></p>
        <p><strong>Input text:</strong> {text}</p>
        </div>
        """

    except Exception as e:
        return f"<p><strong>Anchor explanation failed:</strong> {str(e)}</p>"
