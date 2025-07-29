import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINED_MODELS = {
    "Nano (nano_best.pt)": os.path.join(MODELS_DIR, "nano_best.pt"),
    "Medium (medium_best.pt)": os.path.join(MODELS_DIR, "medium_best.pt"),
    "Large (large_best.pt)": os.path.join(MODELS_DIR, "large_best.pt"),
}

@st.cache_resource
def load_model(weights_path: str):
    """Load YOLO model (cached). Supports local path or model name."""
    return YOLO(weights_path)

def main():
    st.set_page_config(page_title="YOLOv8 Demo", layout="centered")
    st.title("YOLOv8 — Plastic Detection")
    st.markdown("Upload an image and run detection using one of your trained models.")

    col1, col2 = st.columns([1, 3])
    with col1:
        model_choice = st.selectbox("Select Model", list(TRAINED_MODELS.keys()), index=0)
        weights = TRAINED_MODELS[model_choice]
        conf = st.slider("Confidence threshold", 0.0, 1.0, 0.25)
        imgsz = st.selectbox("Image size (px)", [320, 480, 640, 800], index=2)

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        st.image(uploaded, caption="Input image")

    if st.button("Run detection"):
        if uploaded is None:
            st.warning("Please upload an image first.")
            return

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        try:
            model = load_model(weights)
            results = model.predict(source=tmp_path, conf=conf, imgsz=imgsz)
            st.subheader("Result")
            st.image(results[0].plot())
        except Exception as e:
            st.error(f"Inference error: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

if __name__ == '__main__':
    main()
