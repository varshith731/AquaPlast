# AquaPlast Model Weights Guide

The pre-trained YOLOv8 weights for AquaPlast identify **Masks**, **Bags**, and **Bottles**:

- `nano_best.pt` (~6.2 MB): Fastest inference, lightweight edge deployment
- `medium_best.pt` (~52 MB): Balanced precision and inference speed
- `large_best.pt` (~87.6 MB): Maximum accuracy and feature extraction capability

## Obtaining Weights

### Option 1: Re-train in Google Colab
Run the included `Project.ipynb` notebook on a GPU runtime (T4 or A100). The training cells will automatically generate:
- `/content/AI_Project/YOLOv8_Nano/yolov8-nano_plastics_balanced/weights/best.pt` -> `nano_best.pt`
- `/content/AI_Project/YOLOv8_Medium/yolov8-medium_plastics_balanced/weights/best.pt` -> `medium_best.pt`
- `/content/AI_Project/YOLOv8_Large/yolov8-large_plastics_balanced/weights/best.pt` -> `large_best.pt`

### Option 2: Pre-trained Model Checkpoints
Ensure `nano_best.pt`, `medium_best.pt`, and `large_best.pt` are placed in the project root directory alongside `app.py` before launching the Streamlit interface:
```bash
streamlit run app.py
```
