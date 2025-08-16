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

---

<div align="center">
  <sub>Built for computer vision research and practical applications</sub>
</div>

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org/)
[![LibTorch](https://img.shields.io/badge/LibTorch-2.0+-orange.svg)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-1.15+-purple.svg)](https://onnx.ai/)

## 🚀 Overview

VisualFusion LibTorch is a cutting-edge computer vision system designed for **real-time multi-spectral image alignment and fusion**. It leverages deep learning models (LibTorch & ONNX) to perform accurate **EO-IR (Electro-Optical/Infrared) image registration** and creates seamless fused outputs for enhanced situational awareness.

### ✨ Key Features

- 🎯 **Real-Time Performance**: Optimized C++ implementation with GPU acceleration
- 🔍 **Deep Learning-Based Feature Matching**: Advanced keypoint detection and matching using PyTorch/ONNX models
- 🖼️ **Multi-Spectral Fusion**: Seamless EO-IR image fusion with edge-preserving algorithms
- 📐 **Robust Homography Estimation**: RANSAC-based outlier rejection with smart filtering
- 🎛️ **Adaptive Parameters**: Configurable fusion settings for different scenarios
- 📊 **Performance Analytics**: Built-in timing analysis and accuracy metrics
- 🎥 **Video Processing**: Support for both image sequences and video files

## 🏗️ Architecture

```
VisualFusion_libtorch/
├── IR_Convert_v21_libtorch/    # LibTorch implementation
├── Onnx/                       # ONNX Runtime implementation  
├── tensorRT/                   # TensorRT implementation (WIP)
└── convert_to_libtorch/        # Model conversion utilities
```

### 🔧 Supported Inference Engines

| Engine | Status | Performance | Use Case |
|--------|--------|-------------|----------|
| **LibTorch** | ✅ Ready | High | Development & Prototyping |
| **ONNX Runtime** | ✅ Ready | Very High | Production Deployment |
| **TensorRT** | 🚧 WIP | Ultra High | Edge Computing |

## 📋 Requirements

### System Dependencies
- **OS**: Ubuntu 20.04+ / Windows 10+
- **CPU**: Intel i5+ or AMD Ryzen 5+
- **Memory**: 8GB RAM (16GB recommended)
- **GPU**: NVIDIA GTX 1060+ (optional, for acceleration)

### Software Dependencies
- **C++ Compiler**: GCC 9+ or MSVC 2019+
- **CMake**: 3.18+
- **OpenCV**: 4.5+
- **LibTorch**: 2.0+ (for LibTorch version)
- **ONNX Runtime**: 1.15+ (for ONNX version)

## 🛠️ Installation

### Quick Start (Ubuntu)

```bash
# Clone the repository
git clone https://github.com/your-username/VisualFusion_libtorch.git
cd VisualFusion_libtorch

# Install OpenCV and dependencies
sudo apt update
sudo apt install -y libopencv-dev cmake build-essential

# Choose your preferred implementation
cd IR_Convert_v21_libtorch    # For LibTorch
# OR
cd Onnx                       # For ONNX Runtime
```

### LibTorch Setup

```bash
cd IR_Convert_v21_libtorch

# Download LibTorch (CPU version)
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcpu.zip
unzip libtorch-*.zip

# Build the project
bash gcc.sh

# Run with configuration
./build/out config/config.json
```

### ONNX Runtime Setup

```bash
cd Onnx

# Install ONNX Runtime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.15.1/onnxruntime-linux-x64-1.15.1.tgz
tar -xzf onnxruntime-*.tgz

# Build the project
bash gcc.sh

# Run with configuration
./build/out config/config.json
```

## ⚙️ Configuration

The system uses JSON configuration files for flexible parameter tuning:

```json
{
    "input_dir": "./input",
    "output_dir": "./output", 
    "model_path": "./model/SemLA_jit_cuda.zip",
    
    "output_width": 320,
    "output_height": 240,
    "pred_width": 320, 
    "pred_height": 240,
    
    "device": "cuda",
    "pred_mode": "fp32",
    
    "fusion_shadow": true,
    "fusion_edge_border": 2,
    "fusion_threshold_equalization": 128,
    
    "perspective_check": true,
    "perspective_accuracy": 0.85,
    
    "smooth_max_translation_diff": 15.0,
    "smooth_max_rotation_diff": 0.02,
    "smooth_alpha": 0.03
}
```

### Key Parameters

| Parameter | Description | Range | Default |
|-----------|-------------|-------|---------|
| `device` | Compute device | `cpu`/`cuda` | `cuda` |
| `pred_mode` | Precision mode | `fp32`/`fp16` | `fp32` |
| `fusion_shadow` | Shadow enhancement | `true`/`false` | `true` |
| `perspective_accuracy` | RANSAC threshold | 0.7-0.99 | 0.85 |
| `smooth_alpha` | Homography smoothing | 0.01-0.1 | 0.03 |

## 🎮 Usage

### Single Image Processing

```bash
# Process single EO-IR pair
./build/out config/config.json

# Input structure:
# input/
# ├── scene_001_EO.jpg
# └── scene_001_IR.jpg
```

### Video Processing

```bash
# Process synchronized EO-IR videos
./build/out config/video_config.json

# Input structure:  
# input/
# ├── video_EO.mp4
# └── video_IR.mp4
```

### Batch Processing

```bash
# Process entire directory
for config in configs/*.json; do
    ./build/out "$config"
done
```

## 📊 Performance

### Benchmark Results (NVIDIA RTX 3080)

<!-- | Resolution | LibTorch | ONNX Runtime | Memory Usage |
|------------|----------|--------------|--------------|
| 320×240 | 45 FPS | 67 FPS | 2.1 GB |
| 640×480 | 28 FPS | 42 FPS | 3.2 GB |
| 1920×1080 | 12 FPS | 18 FPS | 5.8 GB | -->

## 🔍 Algorithm Details

### Feature Extraction Pipeline
1. **Image Preprocessing**: Cropping and resizing images  
2. **Deep Feature Extraction**: CNN-based keypoint detection
3. **Descriptor Matching**: Learned descriptors with attention mechanism  
4. **Geometric Verification**: RANSAC approach

### Fusion Algorithm
1. **Edge Detection**: Multi-scale Canny with adaptive thresholds
2. **Shadow Enhancement**: Histogram equalization with local adaptation
3. **Blending**: Alpha composition with edge-aware weights
4. **Post-Processing**: Noise reduction and contrast enhancement

## 🐛 Troubleshooting

### Common Issues

**GPU Memory Error**
```bash
# Reduce batch size or switch to CPU
sed -i 's/"device": "cuda"/"device": "cpu"/' config/config.json
```

**Model Loading Failed**
```bash
# Check model path and permissions
ls -la model/
# Ensure model files are accessible
chmod +r model/*
```

**Poor Alignment Quality**
```bash
# Adjust RANSAC parameters
# Increase perspective_accuracy for stricter filtering
# Decrease smooth_alpha for more stable tracking
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

### Development Setup

```bash
# Install development dependencies
sudo apt install -y clang-format cppcheck valgrind

# Run code formatting
clang-format -i **/*.cpp **/*.h

# Run static analysis
cppcheck --enable=all --std=c++17 src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/your-username/VisualFusion_libtorch/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/your-username/VisualFusion_libtorch/discussions)
- 📧 **Contact**: support@visualfusion.ai

## 🙏 Acknowledgments

- [OpenCV](https://opencv.org/) for computer vision primitives
- [PyTorch](https://pytorch.org/) for deep learning framework
- [ONNX](https://onnx.ai/) for model interoperability
- Research community for advancement in image registration

---

<div align="center">
  <sub>Built with ❤️ for computer vision research and applications</sub>
</div>