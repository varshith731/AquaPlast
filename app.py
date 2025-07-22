import streamlit as st
from ultralytics import YOLO
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

if __name__ == '__main__':
    main()
