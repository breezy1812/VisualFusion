import os
import torch
import numpy as np
import random
from model_jit.SemLA import SemLA

# ============================================================================
# 🔒 設置完全確定性（FP16 模式）
# ============================================================================
def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # cuDNN 設置：確定性模式
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("✅ cuDNN deterministic mode enabled")
    
    # 禁用 TF32（RTX 30 系列的關鍵設置）
    if hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = False
        print("✅ CUDA matmul TF32 disabled")
    
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = False
        print("✅ cuDNN TF32 disabled")
    
    # 設置環境變量，強制禁用 TF32
    os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
    print("✅ NVIDIA_TF32_OVERRIDE = 0")
    
    # 確定性算法
    try:
        torch.use_deterministic_algorithms(True)
        print("✅ Deterministic algorithms enabled")
    except Exception:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        print("⚠️  Fallback to CUBLAS_WORKSPACE_CONFIG")
    
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    print(f"✅ Seeds set to {seed}, FP16 MODE ENABLED")
    print("   - TF32: DISABLED")
    print("   - FP16: ENFORCED")

# ============================================================================
# 主要流程：PyFP16 → LibTorch FP16（直接導出 FP16）
# ============================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    # 設置隨機種子（FP16 模式）
    set_all_seeds(42)
    torch.set_grad_enabled(False)

    matcher = SemLA(device=device, fp=torch.float16)
    matcher.load_state_dict(torch.load("./reg.ckpt", map_location=device), strict=False)
    matcher.eval()
    matcher = matcher.to(device, dtype=torch.float16)
    
    
    # dummy forward 初始化（FP16）
    dummy_input_rgb = torch.randn(1, 1, 240, 320, device=device, dtype=torch.float16)
    dummy_input_ir  = torch.randn(1, 1, 240, 320, device=device, dtype=torch.float16)
    with torch.no_grad():
        _ = matcher(dummy_input_rgb, dummy_input_ir)
    print("✅ dummy forward 完成，模型 buffer 已初始化（FP16）")
    
    # 轉換為 TorchScript
    matcher_scripted_fp16 = torch.jit.script(matcher)
    fp16_output_path = "../IR_Convert_v21_libtorch_nx/model/nx_SemLA_fp16.zip"
    torch.jit.save(matcher_scripted_fp16, fp16_output_path)
    print(f"✅ LibTorch FP16 模型已保存到: {fp16_output_path}")

if __name__ == "__main__":
    main()
