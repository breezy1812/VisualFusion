#!/bin/bash

source ./setup_env.sh

echo "🔧 編譯最新版本..."
cd /circ330/forgithub/VisualFusion_libtorch/Onnx
rm -rf build
mkdir build && cd build
cmake .. && make -j$(nproc)

if [ $? -ne 0 ]; then
    echo "❌ 編譯失敗"
    exit 1
fi

cd ..

echo ""
echo "🚀 效能測試：CPU vs GPU"
echo "========================================"

# 測試 CPU 版本
echo "📊 測試 CPU 推論效能..."
cp config/config.json config/config.json.backup
sed -i 's/"device": "cuda"/"device": "cpu"/' config/config.json

echo "CPU 測試中..."
timeout 30s ./build/out 2>&1 | grep -E "(Inference time|Successfully loaded)"

echo ""

# 測試 GPU 版本  
echo "📊 測試 GPU 推論效能..."
sed -i 's/"device": "cpu"/"device": "cuda"/' config/config.json

echo "GPU 測試中..."
timeout 30s ./build/out 2>&1 | grep -E "(Inference time|Successfully loaded|CUDA execution)"

# 恢復設定檔
mv config/config.json.backup config/config.json

echo ""
echo "✅ 效能測試完成！"
echo "📝 詳細推論時間記錄在 onnx_inference_times.csv"
