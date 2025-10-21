
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

print("載入模型...")
matcher = SemLA(device=device, fp=fpMode)
matcher.load_state_dict(torch.load('./reg.ckpt', map_location=device), strict=False)
matcher = matcher.eval().to(device, dtype=fpMode)
matcher.eval()

width, height = 320, 240
print(f"建立輸入張量，尺寸: {height}x{width}")
dummy_input_1 = torch.randn(1, 1, height, width).to(device)
dummy_input_2 = torch.randn(1, 1, height, width).to(device)

output_dir = "../Onnx/model"
os.makedirs(output_dir, exist_ok=True)
onnx_path = f"{output_dir}/SemLA_onnx_opset12_fp32.onnx"

print("轉換並導出 ONNX 原始模型...")
torch.onnx.export(
    matcher,
    (dummy_input_1, dummy_input_2),
    onnx_path,
    verbose=False,
    opset_version=12,
    input_names=["vi_img", "ir_img"],
    output_names=["mkpt0", "mkpt1"],
    keep_initializers_as_inputs=True,
    training=torch.onnx.TrainingMode.EVAL,
)

print("正在使用 onnx-simplifier 進行模型簡化...")
model_onnx = onnx.load(onnx_path)
model_simp, check = onnxsim.simplify(model_onnx)

assert check, "ONNX 模型簡化失敗！"
onnx.save(model_simp, onnx_path)

print("✅ ONNX 模型已簡化並儲存至:", onnx_path)

try:
    onnx.checker.check_model(model_simp)
    print("✅ ONNX模型驗證通過")
except Exception as e:
    print("⚠️ ONNX模型驗證警告:", e)

print("請將配置檔中的 model_path 更新為:")
print(f"    \"{onnx_path}\"")
