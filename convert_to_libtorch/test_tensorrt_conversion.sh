#!/bin/bash

echo "=== TensorRT 模型轉換與測試 ==="

cd /circ330/forgithub/VisualFusion_libtorch/convert_to_libtorch

echo "步驟1: 轉換 ONNX 為 TensorRT (FP16 模式)..."
python3 export_onnx2tensorRT.py

echo "步驟2: 檢查轉換結果..."
if [ -f "/circ330/forgithub/VisualFusion_libtorch/tensorRT/model/trtModel/trt_1200kps_onlyCuda.engine" ]; then
    echo "✅ TensorRT 引擎轉換成功"
    echo "📊 檔案大小:"
    ls -lh /circ330/forgithub/VisualFusion_libtorch/tensorRT/model/trtModel/trt_1200kps_onlyCuda.engine
    
    echo "步驟3: 測試 TensorRT 模型推論..."
    cd /circ330/forgithub/VisualFusion_libtorch/tensorRT
    
    # 建立測試目錄
    mkdir -p build
    cd build
    
    # 編譯 TensorRT 版本
    echo "編譯 TensorRT 推論程式..."
    cmake .. && make -j$(nproc)
    
    if [ $? -eq 0 ]; then
        echo "✅ TensorRT 版本編譯成功"
        
        # 執行測試
        if [ -f "./main" ]; then
            echo "🚀 執行 TensorRT 推論測試..."
            ./main
            
            # 檢查計時結果
            if [ -f "timing_log.csv" ]; then
                echo "📊 TensorRT 推論時間："
                tail -10 timing_log.csv
                
                echo ""
                echo "💡 性能比較建議："
                echo "1. 比較 ONNX Runtime vs TensorRT 的推論時間"
                echo "2. TensorRT 通常提供更佳的 GPU 推論性能"
                echo "3. 檢查推論結果的準確性是否符合預期"
            fi
        fi
    else
        echo "❌ TensorRT 版本編譯失敗"
    fi
    
else
    echo "❌ TensorRT 引擎轉換失敗"
    echo "可能原因："
    echo "1. CUDA/TensorRT 環境問題"
    echo "2. ONNX 模型格式不相容" 
    echo "3. GPU 記憶體不足"
fi

echo "步驟4: 警告處理建議..."
echo "🔧 如要消除 INT64/INT32 警告，可以："
echo "1. 重新導出 ONNX 模型時指定 INT32 輸出"
echo "2. 使用 ONNX Simplifier 優化模型"
echo "3. 或忽略警告，因為對功能影響微小"

echo "=== TensorRT 測試完成 ==="
