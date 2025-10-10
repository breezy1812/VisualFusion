import os
import torch
import numpy as np
import random
from model_jit.SemLA import SemLA

# ============================================================================
# 🔒 設置完全確定性
# ============================================================================
def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # cuDNN 設置
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

# ============================================================================
# 主要流程
# ============================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fpMode = torch.float32

    # 設置隨機種子
    set_all_seeds(42)
    torch.set_grad_enabled(False)

    # -------------------- 載入模型 --------------------
    print("正在載入原始模型...")
    matcher = SemLA(device=device, fp=fpMode)
    matcher.load_state_dict(torch.load("./reg.ckpt", map_location=device), strict=False)
    matcher.eval()
    matcher = matcher.to(device, dtype=fpMode)

    # 驗證 BatchNorm 層
    print("🔍 驗證 BatchNorm 層...")
    bn_count = 0
    for name, module in matcher.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            bn_count += 1
            module.eval()
    print(f"✅ 找到 {bn_count} 個 BatchNorm2d 層，全部已設置為 eval 模式")

    # -------------------- dummy forward 初始化 --------------------
    set_all_seeds(42)
    dummy_input_rgb = torch.randn(1, 1, 240, 320, device=device, dtype=fpMode)
    dummy_input_ir  = torch.randn(1, 1, 240, 320, device=device, dtype=fpMode)
    with torch.no_grad():
        _ = matcher(dummy_input_rgb, dummy_input_ir)
    print("✅ dummy forward 完成，模型 buffer 已初始化")

    # -------------------- 真實圖片 forward 測試 --------------------
    print("\n=== 真實圖片測試 ===")
    # 假設你有真實圖片 tensor: rgb_img, ir_img
    # 這裡用隨機 tensor 模擬
    rgb_img = torch.randn(1, 1, 240, 320, device=device, dtype=fpMode)
    ir_img  = torch.randn(1, 1, 240, 320, device=device, dtype=fpMode)

    with torch.no_grad():
        output_real = matcher(rgb_img, ir_img)
    print("✅ 真實圖片 forward 完成，輸出形狀:")
    # for i, o in enumerate(output_real):
    #     print(f"  output[{i}]: {o.shape}")

    # -------------------- 保存 TorchScript 模型 --------------------
    print("\n=== 轉換 TorchScript 模型 ===")
    set_all_seeds(42)
    matcher_scripted = torch.jit.script(matcher)
    output_path = "/circ330/forgithub/VisualFusion_libtorch/IR_Convert_v21_libtorch/model/SemLA_fp32.zip"
    torch.jit.save(matcher_scripted, output_path)
    print(f"✅ TorchScript 模型已保存到: {output_path}")

#     # -------------------- 驗證 TorchScript 與原始模型一致性 --------------------
#     print("\n=== 驗證 TorchScript 模型 ===")
#     loaded_model = torch.jit.load(output_path, map_location=device)
#     loaded_model.eval()

#     set_all_seeds(42)
#     test_input_rgb = torch.randn(1, 1, 240, 320, device=device, dtype=fpMode)
#     test_input_ir  = torch.randn(1, 1, 240, 320, device=device, dtype=fpMode)

#     with torch.no_grad():
#         orig_out = matcher(test_input_rgb, test_input_ir)
#         js_out   = loaded_model(test_input_rgb, test_input_ir)

#     print("\n📊 數值一致性驗證:")
#     for i, (o, j) in enumerate(zip(orig_out, js_out)):
#         max_diff = torch.max(torch.abs(o - j)).item()
#         mean_diff = torch.mean(torch.abs(o - j)).item()
#         is_close = torch.allclose(o, j, atol=1e-6, rtol=1e-5)
#         status = "✅ 通過" if is_close else "⚠️ 有差異"
#         print(f"output[{i}]: {status}, max diff: {max_diff:.2e}, mean diff: {mean_diff:.2e}")

#     print("\n✅ 全部完成！模型可直接在 libtorch C++ 使用")

# # ============================================================================
if __name__ == "__main__":
    main()
