#!/bin/bash

# 編譯和測試腳本 - 計時功能測試
# 設置錯誤時立即退出
set -e

echo "=== 開始編譯 IR_Convert_v21_libtorch 計時版本 ==="

# 切換到專案目錄
cd /circ330/forgithub/VisualFusion_libtorch/IR_Convert_v21_libtorch

# 建立build目錄
mkdir -p build
cd build

# 清理舊的build檔案
rm -rf *

echo "開始 CMake 配置..."
cmake ..

echo "開始編譯..."
make -j$(nproc)

echo "=== 編譯完成 ==="

# 檢查是否有可執行檔案
if [ -f "./test_imshow" ]; then
    echo "✅ 可執行檔 test_imshow 編譯成功"
    
    # 執行測試 (假設有測試圖片)
    echo "=== 執行計時測試 ==="
    
    # 檢查是否有測試圖片
    if [ -d "../input" ] && [ "$(ls -A ../input)" ]; then
        echo "發現測試圖片，開始執行..."
        ./test_imshow
        
        # 檢查是否生成了 CSV 檔案
        if [ -f "./timing_log.csv" ]; then
            echo "✅ 計時 CSV 檔案生成成功!"
            echo "=== 計時結果 ==="
            cat ./timing_log.csv
        else
            echo "⚠️  未找到計時 CSV 檔案"
        fi
        
        # 顯示輸出結果
        if [ -d "../output" ] && [ "$(ls -A ../output)" ]; then
            echo "✅ 輸出檔案生成成功:"
            ls -la ../output/
        fi
        
    else
        echo "⚠️  沒有找到測試圖片在 input/ 目錄"
        echo "請將測試圖片放入 ../input/ 目錄後重新執行"
    fi
    
elif [ -f "./main" ]; then
    echo "✅ 可執行檔 main 編譯成功"
    
    echo "=== 執行計時測試 ==="
    ./main
    
    # 檢查計時結果
    if [ -f "./timing_log.csv" ]; then
        echo "✅ 計時 CSV 檔案生成成功!"
        echo "=== 計時結果 ==="
        cat ./timing_log.csv
    else
        echo "⚠️  未找到計時 CSV 檔案"
    fi
    
else
    echo "❌ 沒有找到可執行檔案"
    echo "可用的檔案:"
    ls -la ./
fi

echo "=== 編譯測試完成 ==="
