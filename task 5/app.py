import streamlit as st
import cv2
import numpy as np
from PIL import Image
import joblib
import json
from pathlib import Path
from skimage.feature import hog

# Configure page layout and style
st.set_page_config(page_title="Steel Defect Detection", layout="wide", page_icon="🔍")

# Custom CSS for a tidier UI
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1E3A8A; }
    .sub-header { font-size: 1.2rem; color: #4B5563; margin-bottom: 2rem; }
    .status-pass { padding: 1rem; border-radius: 0.5rem; background-color: #D1FAE5; color: #065F46; font-weight: bold; text-align: center; font-size: 1.5rem; }
    .status-fail { padding: 1rem; border-radius: 0.5rem; background-color: #FEE2E2; color: #991B1B; font-weight: bold; text-align: center; font-size: 1.5rem; }
</style>
""", unsafe_allow_html=True)

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

def process_frame(img_np, threshold, config, model_a, model_b):
    """Processes a single frame (RGB), extracts HOG features, predicts defects and returns the annotated frame."""
    # Resize to 200x200 to match the training domain scale. Use INTER_AREA to avoid aliasing artifacts.
    img_resized = cv2.resize(img_np, (200, 200), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    
    win_size = config["win_size"]
    stride = config["scan_stride"]
    
    H, W = gray.shape
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
    
    defective_indices = np.where(probs >= threshold)[0]
    
    out_img = img_resized.copy()
    for idx in defective_indices:
        x, y = pos[idx]
        cv2.rectangle(out_img, (x, y), (x + win_size, y + win_size), (255, 0, 0), 2)
        
    defect_name = None
    if len(defective_indices) > 0:
        img_size = config["img_size"]
        gray_scaled = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
        hog_a = config["hog_track_a"]
        x6 = hog(gray_scaled, orientations=hog_a["orient"], 
                 pixels_per_cell=(hog_a["cell"], hog_a["cell"]),
                 cells_per_block=(hog_a["block"], hog_a["block"]),
                 block_norm=hog_a["block_norm"], feature_vector=True)
        
        type_idx = int(model_a.predict([x6])[0])
        defect_name = config["classes"][type_idx]
        
    return out_img, len(defective_indices) > 0, defect_name

# --- Layout ---
st.markdown('<div class="main-header">🔍 Steel Surface Defect Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated industrial visual inspection pipeline powered by HOG features and SVM.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Configuration")
    input_mode = st.radio("Input Source", ["🖼️ Image Upload", "📹 Live Video Stream"])
    st.markdown("---")
    threshold = st.slider(
        "Sensitivity Threshold", 
        0.0, 1.0, float(config["reject_threshold"]), 0.05,
        help="Higher values reduce false positives but might miss faint defects."
    )
    st.info("The model expects grainy, flat steel surfaces. Smooth textures may cause false positives.")

if input_mode == "🖼️ Image Upload":
    uploaded_file = st.file_uploader("Upload Steel Surface Image", type=["jpg", "png", "jpeg", "bmp"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)
        
        with st.spinner("Analyzing surface..."):
            out_img, is_defective, defect_name = process_frame(img_np, threshold, config, model_a, model_b)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Capture")
            st.image(image, use_container_width=True)
        with col2:
            st.subheader("Inspection Result")
            st.image(out_img, use_container_width=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        if is_defective:
            st.markdown(f'<div class="status-fail">🔴 DEFECTIVE: {defect_name.replace("_", " ").title()}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-pass">🟢 PASS - NO DEFECTS DETECTED</div>', unsafe_allow_html=True)

elif input_mode == "📹 Live Video Stream":
    st.markdown("### Real-time Industrial Webcam Feed")
    st.write("Enable the live stream to continuously monitor the surface for defects.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        stframe = st.empty()
    
    with col2:
        st.write("Controls:")
        run_stream = st.checkbox("🟢 Start Live Stream", value=False)
        status_text = st.empty()
    
    if run_stream:
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Error: Could not access the webcam.")
        else:
            # Continuously read and process frames
            while run_stream:
                ret, frame = cap.read()
                if not ret:
                    st.error("Error: Failed to grab frame.")
                    break
                
                # Convert BGR to RGB for processing & display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process the frame
                out_img, is_defective, defect_name = process_frame(frame_rgb, threshold, config, model_a, model_b)
                
                # Update display placeholders
                stframe.image(out_img, caption="Live Inspection Stream (scaled to 200x200)", use_container_width=True)
                
                if is_defective:
                    status_text.markdown(f'<div class="status-fail">🔴 DEFECTIVE<br><small>{defect_name.replace("_", " ").title()}</small></div>', unsafe_allow_html=True)
                else:
                    status_text.markdown('<div class="status-pass">🟢 PASS</div>', unsafe_allow_html=True)
                    
            cap.release()
