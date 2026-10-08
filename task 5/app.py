import streamlit as st
import cv2
import numpy as np
from PIL import Image
import joblib
import json
from pathlib import Path
from skimage.feature import hog

st.set_page_config(page_title="Steel Surface Defect Detection", layout="centered")
st.title("Steel Surface Defect Detection")
st.write("Upload an image of a steel surface. The application will scan it using HOG features and an SVM model to highlight potential defects.")

@st.cache_resource
def load_models():
    models_dir = Path(__file__).parent / "models"
    config_path = models_dir / "inspection_config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    
    model_b = joblib.load(models_dir / config["model_track_b"])
    model_a = joblib.load(models_dir / config["model_track_a"])
    return config, model_a, model_b

try:
    config, model_a, model_b = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

uploaded_file = st.file_uploader("Upload Steel Surface Image", type=["jpg", "png", "jpeg", "bmp"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    img_np = np.array(image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # We want to scan the image with 64x64 windows
    win_size = config["win_size"]
    stride = config["scan_stride"]
    
    H, W = gray.shape
    if H < win_size or W < win_size:
        gray = cv2.resize(gray, (max(W, win_size), max(H, win_size)))
        H, W = gray.shape
        img_np = cv2.resize(img_np, (W, H))

    nx = int(np.ceil((W - win_size) / stride)) + 1
    ny = int(np.ceil((H - win_size) / stride)) + 1
    xs = np.unique(np.round(np.linspace(0, W - win_size, nx)).astype(int))
    ys = np.unique(np.round(np.linspace(0, H - win_size, ny)).astype(int))
    
    pos = [(int(x), int(y)) for y in ys for x in xs]
    
    wins = np.stack([gray[y:y+win_size, x:x+win_size] for x, y in pos])
    
    hog_b = config["hog_track_b"]
    Xw = []
    for w in wins:
        h = hog(w, orientations=hog_b["orient"], 
                pixels_per_cell=(hog_b["cell"], hog_b["cell"]),
                cells_per_block=(hog_b["block"], hog_b["block"]),
                block_norm=hog_b["block_norm"], feature_vector=True)
        Xw.append(h)
    Xw = np.stack(Xw)
    
    idx_1 = list(model_b.classes_).index(1)
    probs = model_b.predict_proba(Xw)[:, idx_1]
    
    threshold = config["reject_threshold"]
    defective_indices = np.where(probs >= threshold)[0]
    
    out_img = img_np.copy()
    for idx in defective_indices:
        x, y = pos[idx]
        cv2.rectangle(out_img, (x, y), (x + win_size, y + win_size), (255, 0, 0), 2)
        
    st.subheader("Inspection Result")
    
    if len(defective_indices) > 0:
        st.error("DEFECTIVE PRODUCT DETECTED")
        st.image(out_img, caption="Detected Cracks / Defects (Red Boxes)", use_container_width=True)
        
        # Track A: predict defect type on the whole image (resized to 128x128)
        img_size = config["img_size"]
        gray_resized = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
        hog_a = config["hog_track_a"]
        x6 = hog(gray_resized, orientations=hog_a["orient"], 
                 pixels_per_cell=(hog_a["cell"], hog_a["cell"]),
                 cells_per_block=(hog_a["block"], hog_a["block"]),
                 block_norm=hog_a["block_norm"], feature_vector=True)
        
        type_idx = int(model_a.predict([x6])[0])
        defect_name = config["classes"][type_idx]
        st.write(f"**Predicted Defect Type:** {defect_name}")
        
    else:
        st.success("NON-DEFECTIVE PRODUCT")
        st.image(out_img, caption="Clean Product", use_container_width=True)
