# lime_image_explainer.py

import torch
import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from PIL import Image
import io
from torchvision import transforms
from transformers import BeitImageProcessor

def explain_with_lime_image(pil_image, model=None, processor=None, device="cpu"):
    if model is None or processor is None:
        raise ValueError("Model and processor must be provided.")

    model.eval()
    model.to(device)

    def batch_predict(images):
        model.eval()
        batch = torch.stack([
            processor(images=Image.fromarray(img), return_tensors="pt")["pixel_values"].squeeze(0)
            for img in images
        ]).to(device)
        with torch.no_grad():
            logits = model(batch).logits
        probs = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy()
        return probs

    img_np = np.array(pil_image.resize((224, 224)))
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_np, 
        batch_predict, 
        top_labels=1, 
        hide_color=0, 
        num_samples=1000,
        segmentation_fn=None
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], 
        positive_only=False, 
        num_features=10, 
        hide_rest=False
    )
    lime_overlay = mark_boundaries(temp / 255.0, mask, color=(1, 0, 0))  # red boundaries

    # Convert overlay to PIL image
    fig, ax = plt.subplots()
    ax.imshow(lime_overlay)
    ax.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    lime_pil_img = Image.open(buf)

    return lime_pil_img
