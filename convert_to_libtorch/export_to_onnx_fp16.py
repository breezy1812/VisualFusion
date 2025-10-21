
import os
os.environ['NVIDIA_TF32_OVERRIDE'] = '0'

os.environ['PYTHONHASHSEED'] = '42'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

os.environ['ORT_DISABLE_THREAD_SPINNING'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import torch
import random
import numpy as np

import onnx
import onnxsim
def set_deterministic():
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = False
        print("✅ 已禁用 CUDA matmul TF32（確保跨 GPU 一致性）")
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = False
        print("✅ 已禁用 cuDNN TF32（確保跨 GPU 一致性）")
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        print("Warning: torch.use_deterministic_algorithms not supported, continuing...")

set_deterministic()

def set_deterministic():
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = '42'
    if hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = False

from model_jit.SemLA import SemLA

device = torch.device('cuda')
fpMode = torch.float32

print("載入模型 (FP32)...")
matcher = SemLA(device=device, fp=fpMode)
matcher.load_state_dict(torch.load('./reg.ckpt', map_location=device), strict=False)
matcher = matcher.eval().to(device, dtype=fpMode)
matcher.eval()

width, height = 320, 240
print(f"建立輸入張量 (FP32)，尺寸: {height}x{width}")
dummy_input_1 = torch.randn(1, 1, height, width).to(device, dtype=fpMode)
dummy_input_2 = torch.randn(1, 1, height, width).to(device, dtype=fpMode)

output_dir = "../Onnx/model"
os.makedirs(output_dir, exist_ok=True)
onnx_path_fp32 = f"{output_dir}/onnx_op12_fp32.onnx"
onnx_path_fp16 = f"{output_dir}/onnx_op12_fp16.onnx"

print("轉換並導出 ONNX FP32 模型...")
torch.onnx.export(
    matcher,
    (dummy_input_1, dummy_input_2),
    onnx_path_fp32,
    verbose=False,
    opset_version=12,
    input_names=["vi_img", "ir_img"],
    output_names=["mkpt0", "mkpt1"],
    keep_initializers_as_inputs=True,
    training=torch.onnx.TrainingMode.EVAL,
)

print("正在使用 onnx-simplifier 進行模型簡化...")
model_onnx = onnx.load(onnx_path_fp32)
model_simp, check = onnxsim.simplify(model_onnx)

assert check, "ONNX 模型簡化失敗！"
onnx.save(model_simp, onnx_path_fp32)

print("✅ ONNX FP32 模型已儲存至:", onnx_path_fp32)

print("\n正在將 ONNX FP32 轉換為 FP16（自動排除 Cast 節點）...")
try:
    from onnxconverter_common import float16
    exclude_nodes = [node.name for node in model_simp.graph.node if 'Cast' in node.name]
    print(f"   發現 {len(exclude_nodes)} 個 Cast 節點，將排除轉換:")
    for name in exclude_nodes[:10]:
        print(f"      - {name}")
    if len(exclude_nodes) > 10:
        print(f"      ... 還有 {len(exclude_nodes) - 10} 個")
    model_fp16 = float16.convert_float_to_float16(
        model_simp, 
        keep_io_types=False,
        node_block_list=exclude_nodes
    )
    onnx.save(model_fp16, onnx_path_fp16)
    print("✅ ONNX FP16 模型已儲存至:", onnx_path_fp16)
    print("   ⚠️  注意：I/O 轉為 FP16，推論時輸入需用 FP16")
    print("   ⚠️  Cast 節點保持 FP32，避免型別不匹配")
except ImportError:
    print("⚠️  onnxconverter-common 未安裝")
    print("   安裝: pip install onnxconverter-common")
    print("   暫時跳過 FP16 轉換")
except Exception as e:
    print(f"⚠️  FP16 轉換失敗: {e}")
    print("   保留 FP32 模型")

try:
    onnx.checker.check_model(model_simp)
    print("✅ ONNX模型驗證通過")
except Exception as e:
    print("⚠️ ONNX模型驗證警告:", e)

print("\n請將配置檔中的 model_path 更新為:")
print(f"  FP32: \"{onnx_path_fp32}\"")
print(f"  FP16: \"{onnx_path_fp16}\" (如果已生成)")
