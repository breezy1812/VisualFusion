#!/bin/bash
echo "🧹 Cleaning build directory..."
rm -rf build
mkdir -p build

cd build

echo "🛠️ Running CMake..."
cmake ..
echo "🏗️ Building project with Make..."
make -j$(nproc)

