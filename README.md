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
python -c "from ultralytics import YOLO; print(' YO