import os
import torch
import numpy as np
import random
from model_jit.SemLA import SemLA

def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    print(f"✅ Seeds set to {seed}, deterministic mode enabled")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fpMode = torch.float32

    set_all_seeds(42)
    torch.set_grad_enabled(False)

    print("正在載入原始模型...")
    matcher = SemLA(device=device, fp=fpMode)
    matcher.load_state_dict(torch.load("./reg.ckpt", map_location=device), strict=False)
    matcher.eval()
    matcher = matcher.to(device, dtype=fpMode)

    print("🔍 驗證 BatchNorm 層...")
    bn_count = 0
    for name, module in matcher.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            bn_count += 1
            module.eval()
    print(f"✅ 找到 {bn_count} 個 BatchNorm2d 層，全部已設置為 eval 模式")

    set_all_seeds(42)
    dummy_input_rgb = torch.randn(1, 1, 240, 320, device=device, dtype=fpMode)
    dummy_input_ir  = torch.randn(1, 1, 240, 320, device=device, dtype=fpMode)
    with torch.no_grad():
        _ = matcher(dummy_input_rgb, dummy_input_ir)
    print("✅ dummy forward 完成，模型 buffer 已初始化")

    print("\n=== 真實圖片測試 ===")
    rgb_img = torch.randn(1, 1, 240, 320, device=device, dtype=fpMode)
    ir_img  = torch.randn(1, 1, 240, 320, device=device, dtype=fpMode)

    with torch.no_grad():
        output_real = matcher(rgb_img, ir_img)
    print("✅ 真實圖片 forward 完成，輸出形狀:")

    print("\n=== 轉換 TorchScript 模型 ===")
    set_all_seeds(42)
    matcher_scripted = torch.jit.script(matcher)
    output_path = "../IR_Convert_v21_libtorch/model/SemLA_fp32.zip"
    torch.jit.save(matcher_scripted, output_path)
    print(f"✅ TorchScript 模型已保存到: {output_path}")

if __name__ == "__main__":
    main()
