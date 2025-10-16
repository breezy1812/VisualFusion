#!/bin/bash
# TensorRT 版本編譯和執行腳本

echo "======================================"
echo "  TensorRT 版本 - 編譯和執行腳本"
echo "======================================"
echo ""
echo "✅ 修改內容："
echo "   1. 移除特徵點和匹配線的繪製"
echo "   2. 提高 alpha 透明度（邊緣更明顯）"
echo "   3. 輸出兩個影片："
echo "      - xxx_compare.mp4 (IR | EO_warped | Fusion 並排)"
echo "      - xxx_fusion.mp4 (只有融合結果)"
echo ""

# 切換到 tensorRT 目錄
cd /circ330/forgithub/VisualFusion_libtorch/tensorRT

# 創建 build 目錄
echo "📁 創建 build 目錄..."
mkdir -p build
cd build

# 清理舊的編譯檔案
echo "🧹 清理舊的編譯檔案..."
rm -rf CMakeCache.txt CMakeFiles cmake_install.cmake Makefile

# 執行 CMake
echo ""
echo "🔧 執行 CMake 配置..."
cmake ..

if [ $? -ne 0 ]; then
    echo "❌ CMake 配置失敗！"
    exit 1
fi

# 編譯
echo ""
echo "🔨 開始編譯（使用 4 個並行任務）..."
make -j4

if [ $? -ne 0 ]; then
    echo "❌ 編譯失敗！"
    exit 1
fi

echo ""
echo "✅ 編譯成功！"
echo ""
echo "======================================"
echo "  執行程式"
echo "======================================"
echo ""
echo "💡 使用方式："
echo "   cd /circ330/forgithub/VisualFusion_libtorch/tensorRT/build"
echo "   ./main ../config/config.json"
echo ""
echo "📝 注意事項："
echo "   1. 確認 config.json 中的 input_dir 和 output_dir 路徑正確"
echo "   2. 確認 output 設為 true"
echo "   3. 確認 model_path 指向正確的 TensorRT engine"
echo ""
echo "🎬 輸出影片："
echo "   - xxx_compare.mp4: 三者並排對比"
echo "   - xxx_fusion.mp4: 純融合結果（無特徵點）"
echo ""

# 詢問是否要直接執行
read -p "是否要立即執行程式？(y/n): " answer
if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
    echo ""
    echo "🚀 開始執行..."
    ./main ../config/config.json
else
    echo ""
    echo "✅ 編譯完成！您可以稍後手動執行程式。"
fi
