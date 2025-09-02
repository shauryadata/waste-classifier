# app.py — Waste Classifier (ResNet-50, TTA, Abstain w/ top-2)
import os
os.environ["MPLBACKEND"] = "Agg"  # headless Matplotlib for Streamlit

import io
import numpy as np
from pathlib import Path
from PIL import Image

import streamlit as st
import torch, timm
from torchvision import transforms

# --- MUST be first Streamlit call ---
st.set_page_config(page_title="Waste Classifier", page_icon="♻️", layout="centered")

# Optional Grad-CAM (will gracefully skip if not installed)
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    HAS_CAM = True
except Exception:
    HAS_CAM = False

CKPT_PATH = "waste_resnet50_large_ft.pt"

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = get_device()

# --------- ensure model is present (auto-download via download_model.py) ---------
def ensure_model(ckpt_path=CKPT_PATH):
    if not Path(ckpt_path).exists():
        st.info("Downloading model (~90 MB) from GitHub Release…")
        try:
            import download_model  # runs and saves to repo root
        except Exception as e:
            st.error(f"Auto-download failed: {e}")
            st.stop()
        if not Path(ckpt_path).exists():
            st.error("Model download did not complete. Check MODEL_URL in download_model.py.")
            st.stop()

ensure_model()

# ---------------------------- model load (cached) ----------------------------
@st.cache_resource
def load_model_and_classes(ckpt_path=CKPT_PATH):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    classes = ckpt["classes"]
    model = timm.create_model("resnet50", pretrained=False, num_classes=len(classes))
    model.load_state_dict(ckpt["model"])
    model.to(DEVICE).eval()
    return model, classes

model, classes = load_model_and_classes()
IMG_SIZE = 224

# ---------- Preprocess & TTA ----------
base_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
])

def tta_batch(pil_img: Image.Image):
    """Create a small batch of augmented views for test-time augmentation."""
    w, h = pil_img.size
    ims = [pil_img]
    # horizontal flip
    ims.append(pil_img.transpose(Image.FLIP_LEFT_RIGHT))
    # center crop (90%) then resize back
    s = int(min(w, h) * 0.9)
    left = (w - s)//2; top = (h - s)//2
    ims.append(pil_img.crop((left, top, left+s, top+s)).resize((w, h), Image.BICUBIC))
    # padded resize
    pad = int(min(w, h) * 0.05)
    padded = Image.new("RGB", (w+2*pad, h+2*pad), (255,255,255))
    padded.paste(pil_img, (pad, pad))
    ims.append(padded.resize((w, h), Image.BICUBIC))
    xs = [base_tf(im).unsqueeze(0) for im in ims]
    return torch.cat(xs, dim=0)  # [T, C, H, W]

def predict_with_tta(pil_img: Image.Image):
    xb = tta_batch(pil_img).to(DEVICE)
    with torch.no_grad():
        logits = model(xb)                      # [T, C]
        probs  = torch.softmax(logits, dim=1)   # [T, C]
        p = probs.mean(dim=0).cpu().numpy()     # [C]
    order = np.argsort(p)[::-1]
    return p, order  # probabilities + indices sorted desc

def get_target_layer(m):
    if hasattr(m, "layer4"):
        return m.layer4[-1]
    # fallback: last Conv2d
    last = None
    for mod in m.modules():
        if isinstance(mod, torch.nn.Conv2d):
            last = mod
    return last

# ------------------------------- UI -------------------------------
st.title("♻️ Waste Classifier — ResNet-50 (TTA)")
st.caption("Upload an image. The app averages multiple views (TTA) and abstains when confidence is low.")

col1, col2 = st.columns([2,1])
with col2:
    threshold = st.slider(
        "Abstain threshold", 0.30, 0.90, 0.55, 0.01,
        help="If top-1 probability is below this, the app abstains instead of forcing a label."
    )
    force_predict = st.checkbox("Force predict (ignore threshold)", value=False)
    show_cam = st.checkbox("Show Grad-CAM (if available)", value=False)

file = st.file_uploader("Upload a JPG/PNG image", type=["jpg","jpeg","png"])

if file is not None:
    pil = Image.open(io.BytesIO(file.read())).convert("RGB")
    with col1:
        st.image(pil, caption="Uploaded image", use_column_width=True)

    # --- inference ---
    probs, order = predict_with_tta(pil)
    top1, top2, top3 = order[0], order[1], order[2]
    p1, p2, p3 = probs[top1], probs[top2], probs[top3]

    # --- decision logic (shows top-2 when abstaining) ---
    st.subheader("Prediction")
    st.caption(f"Top-1 prob: {p1:.3f}  •  Threshold: {threshold:.2f}")

    if (p1 < threshold) and (not force_predict):
        st.warning(
            f"Unsure (top class below threshold). Most likely **{classes[top1]} — {p1*100:.1f}%**; "
            f"second **{classes[top2]} — {p2*100:.1f}%**."
        )
    else:
        st.success(f"**{classes[top1]} — {p1*100:.1f}%**")

    st.write("Top-3 probabilities:")
    st.write(f"- {classes[top1]} — {p1*100:.1f}%")
    st.write(f"- {classes[top2]} — {p2*100:.1f}%")
    st.write(f"- {classes[top3]} — {p3*100:.1f}%")

    # --- optional Grad-CAM on resized single view ---
    if show_cam:
        if not HAS_CAM:
            st.info("Grad-CAM not available (package not installed).")
        else:
            try:
                x = base_tf(pil.resize((IMG_SIZE, IMG_SIZE))).unsqueeze(0).to(DEVICE)
                target_layer = get_target_layer(model)
                cam = GradCAM(model=model, target_layers=[target_layer])
                grayscale = cam(input_tensor=x, targets=[ClassifierOutputTarget(int(top1))])[0]
                rgb = np.asarray(pil.resize((IMG_SIZE, IMG_SIZE))).astype("float32")/255.0
                overlay = show_cam_on_image(rgb, grayscale, use_rgb=True)
                st.subheader("Explanation (Grad-CAM)")
                st.image(overlay, use_column_width=True)
            except Exception as e:
                st.warning(f"Grad-CAM failed: {e}")
