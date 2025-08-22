#!/bin/bash

# ONNX Runtime 1.18.0 GPU 環境設定
export ONNXRUNTIME_ROOT_PATH="/circ330/onnxruntime-linux-x86_64-gpu-1.18.0"

# CUDA 環境設定 
export CUDA_HOME="/usr/local/cuda"
export CUDA_ROOT="/usr/local/cuda"
export PATH="/usr/local/cuda/bin:$PATH"

# 動態連結庫路徑設定
export LD_LIBRARY_PATH="/circ330/onnxruntime-linux-x86_64-gpu-1.18.0/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH" 
export LD_LIBRARY_PATH="/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

# cuDNN 版本驗證
if [ -f "/usr/local/cuda/lib64/libcudnn.so.8.9.0" ]; then
    echo "✅ cuDNN 8.9.0 已安裝"
else
    echo "⚠️  cuDNN 未正確安裝"
fi

echo "✅ ONNX Runtime GPU 1.18.0 環境設定完成"
echo "📁 ONNXRUNTIME_ROOT_PATH: $ONNXRUNTIME_ROOT_PATH"
echo "🔧 CUDA_HOME: $CUDA_HOME" 
echo "🧠 cuDNN: $(ls /usr/local/cuda/lib64/libcudnn.so.8.* | head -1)"
echo "📚 LD_LIBRARY_PATH: $LD_LIBRARY_PATH"Runtime 1.18.0 GPU 環境設定
export ONNXRUNTIME_ROOT_PATH="/circ330/onnxruntime-linux-x64-gpu-1.18.0"

# CUDA 環境設定
export CUDA_HOME="/usr/local/cuda"
export CUDA_ROOT="/usr/local/cuda"
export PATH="/usr/local/cuda/bin:$PATH"

# 動態連結庫路徑設定
export LD_LIBRARY_PATH="/circ330/onnxruntime-linux-x64-gpu-1.18.0/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH" 
export LD_LIBRARY_PATH="/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

echo "✅ ONNX Runtime GPU 1.18.0 環境設定完成"
echo "📁 ONNXRUNTIME_ROOT_PATH: $ONNXRUNTIME_ROOT_PATH"
echo "🔧 CUDA_HOME: $CUDA_HOME" 
echo "📚 LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
