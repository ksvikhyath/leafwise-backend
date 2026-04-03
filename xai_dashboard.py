import json

import numpy as np
import streamlit as st
from PIL import Image

from advisory import get_plant_advisory
from camv2 import generate_gradcam
from explainers import explain_text_with_anchor, explain_text_with_eli5, explain_text_with_lime
from lime_image_explainer import explain_with_lime_image
from lrp_image_explainer import generate_lrp_image
from model_utils import fuse_predictions, load_beit_model, load_text_model, predict_image, predict_text
from translator_claude import SUPPORTED_LANGUAGES, get_supported_languages, translate_text


st.set_page_config(
    page_title="LeafWise - Intelligent Plant Identifier and Advisory Model",
    layout="wide",
)


st.markdown(
    """
    <style>
        .stApp { background-color: #e0f2f1; }
        body, p, h1, h2, h3, h4, h5, h6, .stMarkdown, .stText {
            color: #263238 !important;
        }
        .stButton>button {
            background-color: #00796b !important;
            color: white !important;
        }
        .prediction-card {
            background-color: #b2dfdb;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }
        .advisory-box {
            background-color: #f1f8e9;
            border-left: 5px solid #2e7d32;
            border-radius: 8px;
            padding: 14px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def _translate_cached(text: str, language_name: str) -> str:
    if not text or language_name == "English":
        return text
    result = translate_text(text=text, target_language=language_name, source_language="en")
    if result.get("success"):
        return result.get("translated_text", text)
    return text


def tr(text: str, language_name: str) -> str:
    return _translate_cached(text, language_name)


@st.cache_resource
def load_models():
    with st.spinner("Loading models..."):
        beit_model, beit_processor, class_names = load_beit_model()
        text_model, tfidf_vectorizer, label_encoder = load_text_model()
    return beit_model, beit_processor, class_names, text_model, tfidf_vectorizer, label_encoder


def _render_top5(title: str, probs: np.ndarray, labels: list[str], language_name: str):
    st.markdown(f"### {tr(title, language_name)}")
    top_idx = probs.argsort()[-5:][::-1]
    for rank, idx in enumerate(top_idx, start=1):
        confidence = float(probs[idx] * 100)
        col1, col2 = st.columns([4, 1])
        with col1:
            st.progress(int(confidence))
            st.write(f"{rank}. {labels[idx]}")
        with col2:
            st.write(f"**{confidence:.2f}%**")


def _generate_image_xai(image, model, processor):
    gradcam_img = lime_img = lrp_img = None
    gradcam_img, _ = generate_gradcam(image, model, "cpu")
    lime_img = explain_with_lime_image(image, model, processor, "cpu")
    lrp_img = generate_lrp_image(image, model, "cpu")
    return gradcam_img, lime_img, lrp_img


def _render_speaker_button(text_to_speak: str, lang_code: str, language_name: str):
    escaped_text = json.dumps(text_to_speak)
    escaped_lang = json.dumps(lang_code)
    label = tr("Read advisory aloud", language_name)
    html = f"""
    <div style="margin-top:10px;">
      <button id="leafwiseSpeak" style="font-size:20px;padding:6px 12px;border-radius:8px;border:1px solid #00796b;cursor:pointer;">
        🔊 {label}
      </button>
    </div>
    <script>
      const text = {escaped_text};
      const lang = {escaped_lang};
      const btn = document.getElementById("leafwiseSpeak");
      if (btn) {{
        btn.onclick = function() {{
          window.speechSynthesis.cancel();
          const u = new SpeechSynthesisUtterance(text);
          u.lang = lang;
          u.rate = 1.0;
          u.pitch = 1.0;
          window.speechSynthesis.speak(u);
        }};
      }}
    </script>
    """
    st.components.v1.html(html, height=65)


def main():
    st.title("LeafWise - Intelligent Plant Identifier and Advisory Model")

    language_options = get_supported_languages()
    default_index = language_options.index("English") if "English" in language_options else 0
    selected_language = st.selectbox("Select language", language_options, index=default_index)

    beit_model, beit_processor, class_names, text_model, tfidf_vectorizer, label_encoder = load_models()

    with st.expander(tr("Input Options", selected_language), expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            uploaded_image = st.file_uploader(tr("Upload leaf image", selected_language), type=["jpg", "jpeg", "png"])
        with c2:
            metadata_input = st.text_area(
                tr("Enter plant metadata (optional)", selected_language),
                placeholder=tr("e.g., aroma, venation, arrangement, etc.", selected_language),
                height=100,
            )

    if not st.button(tr("Identify", selected_language), use_container_width=True):
        return

    if not (uploaded_image or metadata_input):
        st.error(tr("Please provide either an image or text metadata.", selected_language))
        return

    st.markdown(f"## {tr('Prediction Results', selected_language)}")

    img = None
    image_probs = None
    text_probs = None
    image_label = None
    text_label = None
    fused_probs = None
    fused_label = None

    if uploaded_image:
        img = Image.open(uploaded_image).convert("RGB")
        with st.spinner(tr("Running image model...", selected_language)):
            image_label, image_probs, _ = predict_image(img, beit_model, beit_processor, class_names, "cpu")
        st.markdown(f"<div class='prediction-card'><b>{tr('Image model top result', selected_language)}:</b> {image_label}</div>", unsafe_allow_html=True)

    if metadata_input:
        with st.spinner(tr("Running text model...", selected_language)):
            text_label, text_probs = predict_text(metadata_input, text_model, tfidf_vectorizer, label_encoder)
        st.markdown(f"<div class='prediction-card'><b>{tr('Text model top result', selected_language)}:</b> {text_label}</div>", unsafe_allow_html=True)

    if image_probs is not None and text_probs is not None:
        fused_label, fused_probs = fuse_predictions(image_probs, text_probs, class_names)
        st.markdown(f"<div class='prediction-card'><b>{tr('Late fused top result', selected_language)}:</b> {fused_label}</div>", unsafe_allow_html=True)

    identified_plant = fused_label or image_label or text_label
    if identified_plant:
        identified_msg = tr("The model identifies this plant as", selected_language)
        st.success(f"{identified_msg}: **{identified_plant}**")

    if image_probs is not None:
        _render_top5("Top 5 Image Predictions", np.asarray(image_probs), class_names, selected_language)

    if text_probs is not None:
        text_labels = [str(label) for label in label_encoder.classes_]
        _render_top5("Top 5 Text Predictions", np.asarray(text_probs), text_labels, selected_language)

    if fused_probs is not None:
        fused_labels = class_names[: len(fused_probs)]
        _render_top5("Top 5 Late Fused Predictions", np.asarray(fused_probs), fused_labels, selected_language)

    with st.expander(tr("Image XAI Explanations", selected_language), expanded=False):
        if img is None:
            st.info(tr("Upload an image to view image-based explanations.", selected_language))
        else:
            try:
                with st.spinner(tr("Generating image explanations...", selected_language)):
                    gradcam_img, lime_img, lrp_img = _generate_image_xai(img, beit_model, beit_processor)
                cols = st.columns(3)
                with cols[0]:
                    st.image(gradcam_img, caption=tr("Grad-CAM", selected_language), use_container_width=True)
                with cols[1]:
                    st.image(lime_img, caption=tr("LIME", selected_language), use_container_width=True)
                with cols[2]:
                    st.image(lrp_img, caption=tr("LRP", selected_language), use_container_width=True)
            except Exception as exc:
                st.error(f"{tr('Image XAI failed', selected_language)}: {exc}")

    with st.expander(tr("Text XAI Explanations", selected_language), expanded=False):
        if not metadata_input or text_probs is None:
            st.info(tr("Enter metadata text to view text-based explanations.", selected_language))
        else:
            try:
                st.markdown(f"#### {tr('ELI5 Explanation', selected_language)}")
                eli_html = explain_text_with_eli5(metadata_input, text_model, tfidf_vectorizer, label_encoder)
                st.components.v1.html(eli_html, height=320, scrolling=True)

                st.markdown(f"#### {tr('LIME Text Explanation', selected_language)}")
                lime_html = explain_text_with_lime(metadata_input, text_model, tfidf_vectorizer, label_encoder)
                st.components.v1.html(lime_html, height=360, scrolling=True)

                st.markdown(f"#### {tr('Anchor Explanation', selected_language)}")
                anchor_html = explain_text_with_anchor(metadata_input, text_model, tfidf_vectorizer)
                st.markdown(anchor_html, unsafe_allow_html=True)
            except Exception as exc:
                st.error(f"{tr('Text XAI failed', selected_language)}: {exc}")

    with st.expander(tr("Plant Advisory", selected_language), expanded=False):
        if not identified_plant:
            st.info(tr("Run identification first to view advisory output.", selected_language))
        else:
            advisory = get_plant_advisory(identified_plant)
            advisory_text = (
                f"Plant: {advisory.get('common_name', identified_plant)}\n"
                f"Scientific Name: {advisory.get('scientific_name', 'N/A')}\n\n"
                f"Medicinal Uses: {advisory.get('medicinal_uses', 'N/A')}\n\n"
                f"Cultivation: {advisory.get('cultivation', 'N/A')}\n\n"
                f"Income Potential: {advisory.get('income_potential', 'N/A')}\n\n"
                f"Care Instructions: {advisory.get('care_instructions', 'N/A')}"
            )
            translated_advisory = tr(advisory_text, selected_language)
            st.markdown(f"<div class='advisory-box'><pre style='white-space: pre-wrap;'>{translated_advisory}</pre></div>", unsafe_allow_html=True)

            lang_code = SUPPORTED_LANGUAGES.get(selected_language, "en")
            _render_speaker_button(translated_advisory, lang_code, selected_language)


if __name__ == "__main__":
    main()
