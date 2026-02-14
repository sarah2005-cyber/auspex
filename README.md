# Universal Forensic Steganalysis (AUSPEX)

A comprehensive forensic steganalysis system using deep learning to detect steganographic content in multimedia files. This project implements AUSPEX (a universal steganalysis model) with a dual-stream neural network architecture for robust detection and analysis.

## Project Overview

This repository contains the complete implementation of a universal steganalysis pipeline including:
- Pre-trained deep learning model for detecting steganographic content
- Jupyter notebooks for model inference and analysis
- Complete dataset collection and preprocessing scripts
- Backend implementation with Streamlit integration

## Project Structure

```
Model/
├── README.md                                    # This file
├── Universal Forensic Steganalysis.ipynb        # Main analysis notebook
├── backend/
│   ├── model.py                                 # AUSPEX model architecture and utilities
│   ├── requirements.txt                         # Python dependencies
│   ├── Auspex_Universal_v1_seed2026_best_epoch.pt    # Pre-trained model weights
│   └── temp/                                    # Temporary files directory
└── dataset script/
    └── Final_dataset_collection_10000.ipynb     # Dataset collection and preparation
```

## Prerequisites

Before setting up the project, ensure you have the following installed:
- **Python 3.8 or higher** - [Download Python](https://www.python.org/downloads/)
- **pip** - Usually comes with Python 3.4+
- **Git** (optional) - For version control

## Installation & Setup Guide

### Step 1: Clone or Download the Project

If using Git:
```bash
git clone <repository-url>
cd "IIT University of Westminster/4th Year/FYP/Model"
```

Or download the project files and navigate to the project directory.

### Step 2: Create and Activate Virtual Environment

Creating a virtual environment isolates project dependencies and prevents conflicts with other Python projects.

**Windows (PowerShell or Command Prompt):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If you get an execution policy error in PowerShell, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

After activation, your terminal prompt should show `(.venv)` at the beginning.

### Step 3: Install Required Dependencies

With your virtual environment activated, install all required packages:

```bash
pip install --upgrade pip
pip install -r backend/requirements.txt
```

**Project Dependencies:**
| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.2.0 | Deep learning framework for model inference |
| streamlit | 1.28.1 | Interactive web interface for visualization |
| numpy | 1.26.4 | Numerical computing and array operations |
| pandas | 2.2.2 | Data manipulation and analysis |
| matplotlib | 3.8.1 | 2D plotting and visualization |
| librosa | 0.11.0 | Audio processing and feature extraction |
| scipy | 1.11.3 | Scientific computing and signal processing |
| opencv-python | 4.8.1.78 | Computer vision and image processing |
| soundfile | 0.12.1 | Audio file I/O |
| pydub | 0.25.1 | Audio file handling and manipulation |

### Step 4: Verify Installation

Verify that all dependencies are correctly installed:

```bash
python -c "import torch; import librosa; import streamlit; print('✓ All dependencies installed successfully!')"
```

You should see the success message printed to the console.

## Usage

### Quick Start

Once setup is complete and dependencies are installed, you have multiple ways to use the project:

#### Option 1: Run the Interactive Web Application

Launch the Streamlit web interface for easy interaction:

```bash
# Navigate to the backend directory
cd backend

# Run the Streamlit app
streamlit run model.py
```

The application will open in your default browser at `http://localhost:8501`. Use the web interface to:
- Upload audio/media files
- Run steganalysis detection
- View confidence scores and predictions
- Visualize analysis results

#### Option 2: Use the Jupyter Notebook

For detailed analysis and exploration:

```bash
jupyter notebook "Universal Forensic Steganalysis.ipynb"
```

The notebook provides:
- Complete steganalysis pipeline demonstration
- Step-by-step processing of media files
- Visualization of detection results
- Model inference examples

#### Option 3: Prepare Custom Datasets

Prepare and organize your own datasets:

```bash
jupyter notebook "dataset script/Final_dataset_collection_10000.ipynb"
```

Use this notebook to:
- Collect and preprocess audio/media files
- Generate dataset samples (supports up to 10,000 samples)
- Prepare data for model training or evaluation
- Organize data in the project structure

### Using the Pre-trained Model Directly

Access the pre-trained model programmatically in your Python code:

```python
from backend.model import load_model, predict

# Load the pre-trained model
model = load_model('backend/Auspex_Universal_v1_seed2026_best_epoch.pt')

# Run inference on your file
results = predict(model, 'path/to/audio.wav')
print(f"Steganographic content detected: {results['prediction']}")
print(f"Confidence score: {results['confidence']:.4f}")
```

The pre-trained model `Auspex_Universal_v1_seed2026_best_epoch.pt` has been trained with seed 2026 and provides:

## Key Files

| File | Purpose |
|------|---------|
| `backend/model.py` | AUSPEX model architecture, data preprocessing, and inference utilities |
| `backend/Auspex_Universal_v1_seed2026_best_epoch.pt` | Pre-trained model weights (trained with seed 2026) |
| `Universal Forensic Steganalysis.ipynb` | Complete pipeline demonstration and analysis |
| `dataset script/Final_dataset_collection_10000.ipynb` | Dataset preparation and collection (10,000 samples) |

## Model Architecture

AUSPEX uses a sophisticated dual-stream architecture:
- **Stream 1 & 2**: Parallel CNN branches processing different feature representations
- **Fixed High-Pass Filter (HPF)**: Enhanced steganalysis signal detection
- **Residual Connections**: Improved gradient flow and feature learning
- **Adaptive Pooling**: Dynamic feature aggregation
- **Binary Classification**: Detection of steganographic content

## Troubleshooting

### Activation Issues

**Windows PowerShell Error:** If you encounter an execution policy error when activating the virtual environment:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
```

### Module Import Errors

If you encounter import errors or missing modules:

```bash
# Ensure virtual environment is activated (.venv) in prompt
# Upgrade pip
pip install --upgrade pip

# Reinstall all dependencies
pip install --upgrade -r backend/requirements.txt
```

### PyTorch Installation Issues

If torch installation fails or has compatibility issues:

**For CPU-only installations:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**For NVIDIA GPU acceleration (CUDA 11.8):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For AMD GPU or other configurations:**
Visit [PyTorch Installation Guide](https://pytorch.org/get-started/locally/) for your specific setup.

### Streamlit Connection Errors

If Streamlit fails to start or shows port errors:

```bash
# Use a different port
streamlit run model.py --server.port 8502
```

### File Not Found Errors

Ensure:
- You're running commands from the project **root directory**
- The model checkpoint exists at `backend/Auspex_Universal_v1_seed2026_best_epoch.pt`
- All file paths in scripts use forward slashes `/` or double backslashes `\\` on Windows

### Memory Issues

If you encounter out-of-memory errors during processing:
- Process smaller audio files or images
- Reduce batch size if modifying the pipeline
- Close other applications to free up system RAM

## Deactivating Virtual Environment

When finished, deactivate the virtual environment:

```bash
deactivate
```

The `.venv` prefix in your terminal prompt should disappear.

## Project Workflow Summary

1. ✓ **Setup:** Create and activate virtual environment
2. ✓ **Install:** Install dependencies from requirements.txt
3. ✓ **Verify:** Test installations with verification script
4. ✓ **Run:** Choose between web app (Streamlit) or Jupyter notebooks
5. ✓ **Analyze:** Upload files and view steganalysis results

## Citation

If you use this project in your research, please cite the AUSPEX model and this repository.

## License

This project is maintained for forensic steganalysis research purposes.

## Support

For issues, questions, or contributions, please refer to the project documentation or contact the development team.
