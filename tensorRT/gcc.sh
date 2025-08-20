#!/bin/bash
# Clean the build directory to ensure a fresh build
echo "🧹 Cleaning build directory..."
rm -rf /circ330/forgithub/VisualFusion_libtorch/tensorRT/build
mkdir -p /circ330/forgithub/VisualFusion_libtorch/tensorRT/build

# Navigate to the build directory
cd /circ330/forgithub/VisualFusion_libtorch/tensorRT/build

# Run CMake and Make
echo "🛠️ Running CMake..."
cmake ..
echo "🏗️ Building project with Make..."
make -j$(nproc)


