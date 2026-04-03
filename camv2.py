# camv2.py

import torch
import numpy as np
import cv2
from PIL import Image
from transformers import BeitImageProcessor

def generate_gradcam(pil_image, model, device="cpu"):
    model.eval()
    processor = BeitImageProcessor.from_pretrained("microsoft/beit-base-patch16-224")
    inputs = processor(images=pil_image, return_tensors="pt").to(device)
    input_tensor = inputs["pixel_values"]

    gradients = []
    activations = []

    def save_grad(module, grad_input, grad_output):
        # grad_output is a tuple; first element is the gradient tensor
        gradients.append(grad_output[0].detach())

    def save_act(module, input, output):
        activations.append(output.detach())

    # Hook into the output of the last encoder block
    target_layer = model.beit.encoder.layer[-1].output
    handle_fwd = target_layer.register_forward_hook(save_act)
    handle_bwd = target_layer.register_full_backward_hook(save_grad)

    # Forward pass — keep graph for backward
    outputs = model(pixel_values=input_tensor)

    # FIX: use .item() so indexing gives a scalar score
    pred_class = outputs.logits.argmax(dim=-1).item()
    score = outputs.logits[0, pred_class]   # scalar ✅

    model.zero_grad()
    score.backward()

    handle_fwd.remove()
    handle_bwd.remove()

    if not gradients or not activations:
        raise RuntimeError("Grad-CAM hooks did not fire. Check target_layer path.")

    # gradients[0]: (1, num_tokens, hidden)  →  mean over hidden dim → per-token weights
    grad = gradients[0].squeeze(0)      # (num_tokens, hidden)
    act  = activations[0].squeeze(0)    # (num_tokens, hidden)

    # Weight each token's activation by its mean gradient (channel-wise)
    weights = grad.mean(dim=1, keepdim=True)        # (num_tokens, 1)
    cam = (weights * act).sum(dim=1)                # (num_tokens,)
    cam = cam[1:]                                   # drop CLS token → (196,)
    cam = cam.cpu().numpy()

    # BEiT 224×224 with patch size 16 → 14×14 spatial grid
    cam = cam.reshape(14, 14)
    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = np.uint8(255 * cam)

    # Resize to original image dimensions
    cam_resized = cv2.resize(cam, pil_image.size)
    heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)

    # FIX: cv2 returns BGR; convert to RGB before handing to PIL
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = Image.blend(
        pil_image.convert("RGBA"),
        Image.fromarray(heatmap_rgb).convert("RGBA"),
        alpha=0.5
    )
    return overlay, pred_class
