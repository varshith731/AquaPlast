# YOLOv8 Plastic Detection System

A deep learning-based object detection system that identifies and classifies three types of plastic waste: **Masks**, **Bags**, and **Bottles**. Built with YOLOv8 (Nano, Medium, Large variants) and deployed with a user-friendly Streamlit web interface.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Latest-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red)


---

## 🎯 Features

 **Three Model Variants**
   - Nano (lightweight, fast inference)
   - Medium (balanced accuracy & speed)
   - Large (highest accuracy, slower inference)

 **Real-Time Detection**
   - Fast inference on CPU and GPU
   - Adjustable confidence threshold
   - Configurable image resolution (320px, 480px, 640px, 800px)

 **User-Friendly Interface**
   - Web-based Streamlit dashboard
   - Simple image upload and detection
   - Visual bounding boxes on detected plastics
   - Detection summary with counts

 **Production-Ready**
   - Pre-trained models on custom plastic dataset
   - Balanced and augmented training data
   - Optimized for real-world scenarios

---

## 📋 Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Project Structure](#project-structure)
4. [Usage](#usage)
5. [Model Details](#model-details)
6. [Dataset](#dataset)
7. [Configuration](#configuration)
8. [Performance](#performance)
9. [Troubleshooting](#troubleshooting)
10. [Applications & Future Scope](#applications--future-scope)
11. [Contributing](#contributing)
12. [License](#license)

---

## 🛠️ Installation

### Prerequisites
- **Python** 3.8 or higher
- **pip** (Python package manager)
- **Windows/Linux/macOS** (tested on Windows)

### Step 1: Clone or Download the Project
```bash
cd AI_PROJECT
```

### Step 2: Create Virtual Environment (Recommended)
```powershell
# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Dependencies:**
- `ultralytics` - YOLOv8 framework
- `streamlit` - Web interface
- `opencv-python-headless` - Image processing
- `pillow` - Image manipulation
- `numpy` - Numerical computations
- `pandas` - Data handling
- `matplotlib` - Visualization
- `pyyaml` - Configuration files
- `tqdm` - Progress bars

### Step 4: Verify Installation
```bash
python -c "from ultralytics import YOLO; print(' YOLOv8 installed successfully!')"
```

---

## 🚀 Quick Start

### Run the Web Application

```powershell
streamlit run app.py
```

The app will open in your browser at **http://localhost:8501**

### Using the Interface

1. **Select Model**: Choose from Nano, Medium, or Large in the sidebar
2. **Upload Image**: Click "Upload an image" and select a JPG, JPEG, or PNG file
3. **Adjust Parameters**:
   - Confidence threshold (0.0 - 1.0, default: 0.25)
   - Image size (320px, 480px, 640px, 800px)
4. **Run Detection**: Click "Run detection" button
5. **View Results**:
   - Annotated image with bounding boxes
   - Detection summary showing count of each plastic type

### Example Output
```
Result
[Annotated image with bounding boxes]

Detections Found
🔍 Masks: 5 detected
🔍 Bags: 3 detected
🔍 Bottles: 7 detected
```

---

## 📁 Project Structure

```
AI_PROJECT/
├── app.py                           # Streamlit web application
├── Project.ipynb                    # Jupyter notebook (Colab version)
├── requirements.txt                 # Python dependencies
├── nano_best.pt                     # YOLOv8 Nano trained model
├── medium_best.pt                   # YOLOv8 Medium trained model
├── large_best.pt                    # YOLOv8 Large trained model
├── .venv/                           # Virtual environment (created during setup)
└── README.md                        # Environment setup guide
```

---

## 💻 Usage

### Command-Line Inference (Advanced)

```python
from ultralytics import YOLO

# Load model
model = YOLO('nano_best.pt')  # or 'medium_best.pt' or 'large_best.pt'

# Run inference
results = model.predict(source='image.jpg', conf=0.25, imgsz=640)

# Access detections
for result in results:
    for box in result.boxes:
        class_id = int(box.cls)
        confidence = float(box.conf)
        class_name = result.names[class_id]
        print(f"{class_name}: {confidence:.2f}")
```

### Batch Processing

```python
from ultralytics import YOLO

model = YOLO('medium_best.pt')

# Process entire folder
results = model.predict(
    source='path/to/images/',
    conf=0.25,
    imgsz=640,
    save=True  # Save annotated images
)
```

### Video Processing

```python
from ultralytics import YOLO

model = YOLO('medium_best.pt')

# Process video
results = model.predict(
    source='video.mp4',
    conf=0.25,
    imgsz=640,
    save=True  # Save output video
)
```

---

## 🧠 Model Details

### YOLOv8 Variants Comparison

| Aspect | Nano | Medium | Large |
|--------|------|--------|-------|
| **Model Size** | ~7 MB | ~50 MB | ~200 MB |
| **Inference Time** | ~15ms | ~40ms | ~100ms |
| **Memory Usage** | ~100 MB | ~500 MB | ~2 GB |
| **CPU Inference** | ✅ Fast | ⚠️ Moderate | ❌ Slow |
| **GPU Inference** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Accuracy** | Good | Better | Best |
| **Best For** | Edge devices, Real-time | Balanced use | High accuracy |

### Training Details

**Dataset:**
- Source: plastics_filtered dataset
- Train/Val/Test split: standard YOLO format
- Classes