# VisualFusion LibTorch

🔥 **Real-Time EO-IR Image Alignment and Fusion System**

[![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org/)
[![LibTorch](https://img.shields.io/badge/LibTorch-2.0+-orange.svg)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-1.15+-purple.svg)](https://onnx.ai/)

## 🚀 Overview

VisualFusion LibTorch is a computer vision system for **EO-IR (Electro-Optical/Infrared) image alignment and fusion**. It uses deep learning models to detect feature points and compute homography matrices for accurate image registration, then creates fused outputs with advanced edge-preserving algorithms.

### ✨ Key Features

- 🎯 **Deep Learning Feature Detection**: Uses SemLA model for keypoint detection and matching
- 🖼️ **EO-IR Image Fusion**: Seamless fusion with shadow enhancement and edge preservation  
- 📐 **RANSAC Homography**: Robust estimation with outlier filtering
- 🎛️ **Homography Smoothing**: Temporal consistency with configurable smoothing parameters
- ⚙️ **Video & Image Cropping**: Support for VideoCut and PictureCut parameters
- 📊 **Performance Timing**: Built-in timer analysis for each processing stage

## 🏗️ Architecture

```
VisualFusion_libtorch/
├── IR_Convert_v21_libtorch/    # LibTorch implementation 
├── Onnx/                       # ONNX Runtime implementation
├── tensorRT/                   # TensorRT implementation (WIP)
└── convert_to_libtorch/        # Model conversion utilities
```

### 🔧 Supported Inference Engines

| Engine | Status | Model Format | Device Support |
|--------|--------|--------------|----------------|
| **LibTorch** | ✅ Ready | `.zip` (TorchScript) | CPU/CUDA |
| **ONNX Runtime** | ✅ Ready | `.onnx` | CPU |
| **TensorRT** | 🚧 WIP | `.trt` | CUDA |

## 📋 Requirements

### System Dependencies
- **OS**: Ubuntu 20.04+ 
- **CPU**: Multi-core processor
- **Memory**: 4GB RAM minimum
- **GPU**: NVIDIA GPU (optional, for CUDA acceleration)

### Software Dependencies
- **C++ Compiler**: GCC 9+
- **CMake**: 3.18+
- **OpenCV**: 4.5+
- **LibTorch**: For LibTorch version
- **ONNX Runtime**: For ONNX version

## 🛠️ Installation & Usage

### LibTorch Version

```bash
cd IR_Convert_v21_libtorch

# Build the project
bash gcc.sh

# Run with configuration
./build/out config/config.json
```

### ONNX Runtime Version  

```bash
cd Onnx

# Build the project
bash gcc.sh

# Run with configuration  
./build/out config/config.json
```

## ⚙️ Configuration

### Core Parameters

```json
{
    "input_dir": "/path/to/input",
    "output_dir": "/path/to/output", 
    "output": true,
    
    "device": "cuda",
    "pred_mode": "fp32",
    "model_path": "./model/SemLA_jit_cuda.zip",
    
    "output_width": 320,
    "output_height": 240,
    "pred_width": 320, 
    "pred_height": 240
}
```

### Video/Image Cropping

```json
{
    "VideoCut": true,
    "Vcut_x": 870,
    "Vcut_y": 235, 
    "Vcut_w": 2020,
    "Vcut_h": 1680,
    
    "PictureCut": true,
    "Pcut_x": 220,
    "Pcut_y": 0,
    "Pcut_w": 1920,
    "Pcut_h": 1080
}
```

### Fusion Settings

```json
{
    "fusion_shadow": true,
    "fusion_edge_border": 2,
    "fusion_threshold_equalization": 128,
    "fusion_threshold_equalization_low": 72,
    "fusion_threshold_equalization_high": 192,
    "fusion_threshold_equalization_zero": 64,
    "fusion_interpolation": "cubic"
}
```

### Perspective & Alignment

```json
{
    "perspective_check": true,
    "perspective_distance": 10,
    "perspective_accuracy": 0.85,
    
    "align_distance_last": 15.0,
    "align_distance_line": 10.0,
    "align_angle_mean": 10.0,
    "align_angle_sort": 0.7
}
```

### Homography Smoothing

```json
{
    "smooth_max_translation_diff": 80.0,
    "smooth_max_rotation_diff": 0.05,
    "smooth_alpha": 0.05
}
```

## 📁 Input Format

The system expects paired EO-IR images with `_EO` and `_IR` suffixes:

```
input/
├── scene_001_EO.jpg
├── scene_001_IR.jpg
├── scene_002_EO.jpg  
└── scene_002_IR.jpg
```

## 🎮 Processing Pipeline

1. **Input Loading**: Reads EO-IR image pairs from input directory
2. **Image Cropping**: Applies VideoCut/PictureCut if enabled
3. **Resizing**: Scales images to prediction and output sizes
4. **Feature Detection**: Uses SemLA model to find keypoint pairs
5. **Homography Computation**: RANSAC-based robust estimation
6. **Homography Smoothing**: Temporal consistency filtering
7. **Edge Detection**: Multi-scale edge extraction
8. **Image Fusion**: Shadow-enhanced blending with configurable interpolation
9. **Output Generation**: Saves combined visualization with feature points

## 🔍 Algorithm Components

### Feature Matching
- **Model**: SemLA (Semantic Line Association)
- **Input**: 320×240 grayscale image pairs
- **Output**: Corresponding keypoint coordinates
- **Post-processing**: RANSAC filtering with 8.0px threshold

### Homography Smoothing
- **Translation Threshold**: Configurable max pixel difference
- **Rotation Threshold**: Configurable max radians difference  
- **Smoothing Factor**: Weighted average with previous frames
- **Fallback Logic**: Handles large motion discontinuities

### Image Fusion
- **Edge Enhancement**: Canny edge detection with adaptive thresholds
- **Shadow Processing**: Histogram equalization with multiple threshold levels
- **Interpolation**: Linear or cubic resampling options
- **Blending**: Alpha composition with edge-aware weights

## 🐛 Troubleshooting

**Model Loading Issues**
```bash
# Check model file exists and is readable
ls -la model/
```

**Memory Errors**
```bash  
# Switch to CPU mode
sed -i 's/"device": "cuda"/"device": "cpu"/' config/config.json
```

**Poor Alignment**
```bash
# Adjust RANSAC parameters
# Increase perspective_accuracy (0.85 → 0.95)
# Decrease smooth_alpha for more stable tracking
```

## 📊 Performance

### Timing Analysis
The system includes built-in timing for each processing stage:
- Resize
- Gray conversion  
- Homography computation
- Edge detection
- Perspective transformation
- Image fusion
- Alignment processing

### Tested Resolutions
- **320×240**: Primary supported resolution
- **Custom sizes**: Configurable via config parameters

## 🔧 Development

### Build System
Uses CMake with custom `gcc.sh` build script for convenience.

### Code Structure
- `main.cpp`: Main processing pipeline
- `lib_image_fusion/`: Core computer vision algorithms
- `utils/`: Timing and utility functions
- `nlohmann/`: JSON configuration parsing

## 🙏 Acknowledgments

- [OpenCV](https://opencv.org/) for computer vision primitives
- [PyTorch](https://pytorch.org/) for deep learning framework
- [ONNX](https://onnx.ai/) for model interoperability
- Research community for advancement in image registration

---

<div align="center">
  <sub>Built with ❤️ for computer vision research and applications</sub>
</div>