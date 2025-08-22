#!/bin/bash

source ./setup_env.sh

echo "🎯 完整 CPU vs GPU 效能測試"
echo "========================================"

# 編譯程式
echo "🔧 編譯程式..."
rm -rf build && mkdir build && cd build
cmake .. && make -j$(nproc) > /dev/null 2>&1
cd ..

# 清空舊的 CSV 記錄
rm -f onnx_inference_times.csv

echo ""
echo "📊 CPU 推論測試 (3次)..."
cp config/config.json config/config.json.backup
sed -i 's/"device": "cuda"/"device": "cpu"/' config/config.json

for i in {1..3}; do
    echo "  測試 $i/3:"
    timeout 15s ./build/out 2>&1 | grep -E "(ONNX Inference time|debug: 模型推論完成)"
done

echo ""
echo "📊 GPU 推論測試 (3次)..."
sed -i 's/"device": "cpu"/"device": "cuda"/' config/config.json

for i in {1..3}; do
    echo "  測試 $i/3:"
    timeout 15s ./build/out 2>&1 | grep -E "(ONNX Inference time|debug: 模型推論完成|CUDA execution)"
done

# 恢復設定檔
mv config/config.json.backup config/config.json

echo ""
echo "✅ 測試完成！"
echo "📄 詳細推論時間記錄在 onnx_inference_times.csv"
if [ -f onnx_inference_times.csv ]; then
    echo ""
    echo "📈 推論時間摘要："
    cat onnx_inference_times.csv | head -10
fi
