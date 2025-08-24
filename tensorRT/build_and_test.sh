#!/bin/bash

# TensorRT 版本編譯和測試腳本
echo "==================== TensorRT 版本編譯和測試 ===================="

# 設定環境變數
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

cd /circ330/forgithub/VisualFusion_libtorch/tensorRT

# 清理並重新編譯
echo "清理之前的編譯結果..."
make clean

echo "重新編譯..."
make -j4

if [ $? -ne 0 ]; then
    echo "編譯失敗！"
    exit 1
fi

echo "編譯成功！開始測試..."

# 執行測試
echo "執行 TensorRT 推論測試..."
./out

echo "==================== 測試完成 ===================="
echo "推論時間記錄已保存到 tensorrt_inference_times.csv"
