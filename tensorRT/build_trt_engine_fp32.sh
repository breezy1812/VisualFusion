#!/bin/bash
# PyTorch → ONNX → TensorRT 完整轉換流程
# 使用 trtexec（比 Python API 更穩定）

set -e

echo "========================================"
echo "PyTorch → ONNX → TensorRT Converter"
echo "========================================"

# 步驟 1：PyTorch → ONNX（使用 Python）
echo ""
echo "步驟 1/2: 轉換 PyTorch → ONNX..."
echo "----------------------------------------"

python3 <<'PYTHON_SCRIPT'
import os
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

import torch
import onnx
import onnxsim
import numpy as np

# 設定確定性
torch.manual_seed(42)
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if hasattr(torch.backends.cuda, 'matmul'):
    torch.backends.cuda.matmul.allow_tf32 = False
if hasattr(torch.backends.cudnn, 'allow_tf32'):
    torch.backends.cudnn.allow_tf32 = False

# 載入模型
from model_jit.SemLA import SemLA
device = torch.device("cuda")
fpMode = torch.float32

matcher = SemLA(device=device, fp=fpMode)
matcher.load_state_dict(torch.load("./reg.ckpt", map_location=device), strict=False)
matcher = matcher.eval().to(device, dtype=fpMode)
torch.set_grad_enabled(False)

print("✅ 模型已載入 (FP32)")

# 建立輸入
width = 320
height = 240
torch_input_1 = torch.randn(1, 1, height, width).to(device, dtype=fpMode)
torch_input_2 = torch.randn(1, 1, height, width).to(device, dtype=fpMode)

# 轉換為 ONNX
onnx_path = "/circ330/forgithub/VisualFusion_libtorch/tensorRT/model/temp_semla_fp32.onnx"
print(f"轉換為 ONNX (OpSet 12)...")

torch.onnx.export(
    matcher,
    (torch_input_1, torch_input_2),
    onnx_path,
    verbose=True,  # 顯示詳細訊息以偵錯
    opset_version=12,
    input_names=["vi_img", "ir_img"],
    output_names=["mkpt0", "mkpt1", "leng1", "leng2"],
    do_constant_folding=True,
    dynamic_axes=None,  # 固定尺寸
)
print(f"✅ ONNX 轉換完成")

# 簡化 ONNX
print("簡化 ONNX 模型...")
model_onnx = onnx.load(onnx_path)
model_simp, check = onnxsim.simplify(model_onnx)
if not check:
    print("⚠️  ONNX 簡化失敗，使用未簡化版本")
    model_simp = model_onnx

onnx.save(model_simp, onnx_path)
print(f"✅ ONNX 已儲存: {onnx_path}")

# 驗證 ONNX
try:
    onnx.checker.check_model(model_simp)
    print("✅ ONNX 驗證通過")
except Exception as e:
    print(f"⚠️  ONNX 驗證警告: {e}")

# 列出所有運算符
ops = set()
for node in model_simp.graph.node:
    ops.add(node.op_type)
print(f"\n📋 ONNX 運算符 ({len(ops)} 種):")
for op in sorted(ops):
    print(f"  - {op}")
PYTHON_SCRIPT

if [ ! -f "/circ330/forgithub/VisualFusion_libtorch/tensorRT/model/temp_semla_fp32.onnx" ]; then
    echo "❌ ONNX 轉換失敗"
    exit 1
fi

# 步驟 2：ONNX → TensorRT（使用 trtexec）
echo ""
echo "步驟 2/2: 轉換 ONNX → TensorRT..."
echo "----------------------------------------"

export NVIDIA_TF32_OVERRIDE=0
echo "✅ 環境變數: NVIDIA_TF32_OVERRIDE=0"

# 設置 TensorRT 動態庫路徑
export LD_LIBRARY_PATH=/circ330/TensorRT-8.4.3.1/lib:$LD_LIBRARY_PATH
echo "✅ LD_LIBRARY_PATH: /circ330/TensorRT-8.4.3.1/lib"

ONNX_MODEL="/circ330/forgithub/VisualFusion_libtorch/tensorRT/model/temp_semla_fp32.onnx"
OUTPUT_ENGINE="/circ330/forgithub/VisualFusion_libtorch/tensorRT/model/trt_semla_fp32_op12.engine"

echo "🔨 使用 trtexec 建立 TensorRT engine..."
echo "   - 輸入: $ONNX_MODEL"
echo "   - 輸出: $OUTPUT_ENGINE"
echo "   - 精度: FP32 (禁用 TF32)"
echo ""

/circ330/TensorRT-8.4.3.1/bin/trtexec \
    --onnx="$ONNX_MODEL" \
    --saveEngine="$OUTPUT_ENGINE" \
    --workspace=256 \
    --noTF32 \
    --verbose \
    --dumpLayerInfo \
    2>&1 | tee /circ330/trt_conversion.log

if [ -f "$OUTPUT_ENGINE" ]; then
    echo ""
    echo "========================================"
    echo "✅ 轉換成功！"
    echo "========================================"
    echo "📁 TensorRT Engine: $OUTPUT_ENGINE"
    echo "📏 檔案大小: $(du -h $OUTPUT_ENGINE | cut -f1)"
    echo ""
    echo "🧹 清理臨時檔案..."
    # rm -f "$ONNX_MODEL"
    echo "✅ 保留 ONNX 檔案供檢查: $ONNX_MODEL"
    echo ""
    echo "📝 完整日誌: /circ330/trt_conversion.log"
else
    echo ""
    echo "========================================"
    echo "❌ TensorRT 轉換失敗"
    echo "========================================"
    echo "請檢查日誌: /circ330/trt_conversion.log"
    exit 1
fi
