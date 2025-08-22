#!/bin/bash

source ./setup_env.sh

echo "🎯 最終 ONNX GPU 測試（減少警告版本）"
echo "========================================"

# 編譯程式
echo "🔧 編譯最新版本..."
rm -rf build && mkdir build && cd build
cmake .. && make -j$(nproc) > /dev/null 2>&1

if [ $? -ne 0 ]; then
    echo "❌ 編譯失敗"
    exit 1
fi

cd ..

echo ""
echo "🚀 測試優化後的 CUDA 推論（應該減少警告）..."
echo "========================================"

# 確保使用 CUDA 模式
cp config/config.json config/config.json.backup
sed -i 's/"device": "cpu"/"device": "cuda"/' config/config.json

echo "執行測試..."
timeout 20s ./build/out 2>&1

# 恢復設定檔
mv config/config.json.backup config/config.json

echo ""
echo "✅ 測試完成！"
