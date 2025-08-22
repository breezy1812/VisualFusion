#!/bin/bash

source ./setup_env.sh

echo "🔧 快速編譯測試..."
cd /circ330/forgithub/VisualFusion_libtorch/Onnx
rm -rf build && mkdir build && cd build
cmake .. && make -j$(nproc)

if [ $? -ne 0 ]; then
    echo "❌ 編譯失敗"
    exit 1
fi

cd ..
echo ""
echo "🚀 測試優化後的 CUDA 推論（無 warm-up）..."
echo "========================================"

timeout 20s ./build/out 2>&1 | grep -E "(debug:|Inference time|Successfully loaded|CUDA|Warning|ERROR)"

echo ""
echo "✅ 測試完成！"
