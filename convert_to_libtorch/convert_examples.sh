#!/bin/bash

echo "===== SemLA PyTorch to TensorRT 轉換範例 ====="
echo ""

echo "🚀 範例 1: 使用 FP16 精度，OpSet 12 (推薦)"
echo "python export_onnx2tensorRT.py --fp16 --opset 12"
echo ""

echo "🚀 範例 2: 使用 FP32 精度，OpSet 12"
echo "python export_onnx2tensorRT.py --opset 12"
echo ""

echo "🚀 範例 3: 指定自訂路徑，FP16 模式"
echo "python export_onnx2tensorRT.py --model ./reg.ckpt --trt ./my_fp16_model.engine --fp16 --opset 12"
echo ""

echo "🚀 範例 4: 從已有的 ONNX 模型轉換"
echo "python export_onnx2tensorRT.py --onnx ./path/to/model.onnx --trt ./output.engine --fp16"
echo ""

echo "🚀 範例 5: 設定大工作空間 (2GB)"
echo "python export_onnx2tensorRT.py --fp16 --workspace-size 2048"
echo ""

echo "📝 注意事項:"
echo "- --fp16: 啟用 FP16 精度 (較快但可能略微降低精度)"
echo "- --opset: ONNX OpSet 版本 (預設為 12)"
echo "- --model: PyTorch 模型檢查點路徑 (預設為 ./reg.ckpt)"
echo "- --trt: 輸出 TensorRT 引擎路徑 (自動產生如果未指定)"
echo "- --workspace-size: TensorRT 工作空間大小 (MB，預設 1024)"
echo ""

echo "🎯 建議的 FP16 轉換指令:"
python export_onnx2tensorRT.py --fp16 --opset 12
