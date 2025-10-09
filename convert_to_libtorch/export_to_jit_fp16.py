import os
import torch
import numpy as np
import random
from model_jit.SemLA import SemLA

# ============================================================================
# 🔒 設置完全確定性（FP32 模式）
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
    
    print(f"✅ Seeds set to {seed}, FP32 MODE ENABLED")
    print("   - TF32: DISABLED")
    print("   - FP32: ENFORCED")

# ============================================================================
# 主要流程：PyFP32 → LibTorch FP32（業界推薦方案）
# FP16 轉換將在 C++ LibTorch 端動態執行
# ============================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("="*70)
    print("🔥 PyTorch to LibTorch FP32 Conversion Tool")
    print("   流程: PyTorch FP32 → LibTorch FP32")
    print("   ⭐ FP16 轉換將在 C++ 端執行 (業界推薦方案)")
    print("="*70)
    
    # 設置隨機種子（FP32 模式）
    set_all_seeds(42)
    torch.set_grad_enabled(False)

    # ============================================================================
    # 步驟 1: 載入 PyTorch FP32 模型
    # ============================================================================
    print("\n【步驟 1/2】載入 PyTorch FP32 模型...")
    matcher = SemLA(device=device, fp=torch.float32)
    matcher.load_state_dict(torch.load("./reg.ckpt", map_location=device), strict=False)
    matcher.eval()
    matcher = matcher.to(device, dtype=torch.float32)
    
    # 驗證所有參數都是 FP32
    print("🔍 驗證模型參數類型...")
    fp32_params = 0
    fp16_params = 0
    for name, param in matcher.named_parameters():
        if param.dtype == torch.float32:
            fp32_params += 1
        elif param.dtype == torch.float16:
            fp16_params += 1
            print(f"  ⚠️  FP16 參數: {name}")
    print(f"  ✅ FP32 參數: {fp32_params}")
    if fp16_params > 0:
        print(f"  ⚠️  FP16 參數: {fp16_params}（將保持為 FP32）")

    # 驗證 BatchNorm 層
    print("🔍 驗證 BatchNorm 層...")
    bn_count = 0
    for name, module in matcher.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            bn_count += 1
            module.eval()
    print(f"✅ 找到 {bn_count} 個 BatchNorm2d 層，全部已設置為 eval 模式")

    # ============================================================================
    # 步驟 2: 保存 LibTorch FP32 模型（僅此而已）
    # ============================================================================
    print("\n【步驟 2/2】轉換並保存 LibTorch FP32 模型...")
    set_all_seeds(42)
    
    # dummy forward 初始化
    dummy_input_rgb = torch.randn(1, 1, 240, 320, device=device, dtype=torch.float32)
    dummy_input_ir  = torch.randn(1, 1, 240, 320, device=device, dtype=torch.float32)
    with torch.no_grad():
        _ = matcher(dummy_input_rgb, dummy_input_ir)
    print("✅ dummy forward 完成，模型 buffer 已初始化")
    
    # 轉換為 TorchScript
    matcher_scripted_fp32 = torch.jit.script(matcher)
    fp32_output_path = "/circ330/forgithub/VisualFusion_libtorch/IR_Convert_v21_libtorch/model/SemLA_fp32.zip"
    torch.jit.save(matcher_scripted_fp32, fp32_output_path)
    print(f"✅ LibTorch FP32 模型已保存到: {fp32_output_path}")

    # ============================================================================
    # 驗證模型
    # ============================================================================
    print("\n【驗證】驗證 FP32 模型推論...")
    
    # 重新載入模型以驗證
    loaded_fp32_model = torch.jit.load(fp32_output_path, map_location=device)
    loaded_fp32_model.eval()
    
    set_all_seeds(42)
    test_input_rgb = torch.randn(1, 1, 240, 320, device=device, dtype=torch.float32)
    test_input_ir  = torch.randn(1, 1, 240, 320, device=device, dtype=torch.float32)

    with torch.no_grad():
        # FP32 模型推論
        fp32_out = loaded_fp32_model(test_input_rgb, test_input_ir)
        
        # 驗證輸出
        print("\n📊 FP32 模型輸出驗證:")
        for i, output in enumerate(fp32_out):
            print(f"output[{i}]: shape={output.shape}, dtype={output.dtype}")
        print("✅ FP32 模型推論成功")

    print("\n" + "="*70)
    print("✅ 轉換完成！")
    print(f"  - FP32 模型: {fp32_output_path}")
    print("")
    print("⭐ 業界推薦的 FP16 使用方式：")
    print("  1. 在 C++ LibTorch 中載入此 FP32 模型")
    print("  2. 使用 module.to(torch::kHalf) 動態轉換為 FP16")
    print("  3. 輸入資料也轉為 FP16：input.to(torch::kHalf)")
    print("")
    print("📝 C++ 範例代碼：")
    print("  torch::jit::script::Module module = torch::jit::load(\"SemLA_jit_cuda_fp32.zip\");")
    print("  module.to(torch::kCUDA);")
    print("  module.to(torch::kHalf);  // 動態轉 FP16")
    print("")
    print("  auto input_rgb_fp16 = input_rgb.to(torch::kHalf);")
    print("  auto input_ir_fp16 = input_ir.to(torch::kHalf);")
    print("  auto outputs = module.forward({input_rgb_fp16, input_ir_fp16});")
    print("="*70)

# ============================================================================
if __name__ == "__main__":
    main()
