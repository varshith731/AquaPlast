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

def annotate_results(results):
    """Return annotated image (numpy array) and class names detected."""
    r = results[0]
    ann = r.plot()
    class_counts = {}
    if hasattr(r, 'boxes') and r.boxes is not None:
        try:
            class_names = r.names
            for box in r.boxes:
                cls_id = int(box.cls)
                cls_name = class_names.get(cls_id, f"Class {cls_id}") if isinstance(class_names, dict) else class_names[cls_id]
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
        except Exception:
            class_counts = {}
    return ann, class_counts

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
            ann, class_counts = annotate_results(results)
            st.subheader("Result")
            st.image(ann)
            if class_counts:
                st.subheader("Detections Found")
                for cls_name, count in class_counts.items():
                    st.write(f"🔍 **{cls_name}**: {count} detected")
            else:
                st.info("No detections above the confidence threshold.")
        except Exception as e:
            st.error(f"Inference error: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

if __name__ == '__main__':
    main()
