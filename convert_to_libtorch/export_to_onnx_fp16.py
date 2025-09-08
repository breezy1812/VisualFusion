import torch
import os
import onnx

from model_jit.SemLA import SemLA

print("=== SemLA ONNX FP16 轉換腳本 (直接導出) ===")

# 使用CUDA來獲得最佳性能
device = torch.device("cuda")
fpMode = torch.float16
print(f"使用設備: {device}")

# 直接以 FP16 載入並轉換模型
print("正在載入並轉換模型為 FP16...")
matcher = SemLA(device=device, fp=fpMode)
matcher.load_state_dict(torch.load(f"./reg.ckpt", map_location=device), strict=False)
matcher = matcher.eval().to(device, dtype=fpMode)

# 使用與配置文件相符的尺寸
width = 320
height = 240

print(f"建立 FP16 輸入張量，尺寸: {height}x{width}")
torch_input_1 = torch.randn(1, 1, height, width).to(device, dtype=fpMode)
torch_input_2 = torch.randn(1, 1, height, width).to(device, dtype=fpMode)

# 確保輸出目錄存在
output_dir = "../Onnx/model"
os.makedirs(output_dir, exist_ok=True)

# 直接導出FP16 ONNX模型
# 使用一個新名稱以避免與舊模型混淆
fp16_output_path = f"{output_dir}/zfp16_op12_backup.onnx"

print(f"直接轉換為FP16 ONNX模型...")
print(f"最終路徑: {fp16_output_path}")

try:
    torch.onnx.export(
        matcher,
        (torch_input_1, torch_input_2),
        fp16_output_path,
        verbose=False,
        opset_version=12,
        input_names=["vi_img", "ir_img"],
        output_names=["mkpt0", "mkpt1", "leng1", "leng2"],
        do_constant_folding=True,
    )
    print("✅ FP16 ONNX模型直接轉換完成")

    # 驗證FP16模型
    onnx_model = onnx.load(fp16_output_path)
    onnx.checker.check_model(onnx_model)
    print("✅ FP16 ONNX模型驗證通過")

except Exception as e:
    print(f"❌ FP16 ONNX轉換失敗: {e}")
    exit(1)

# 檢查檔案大小比較
if os.path.exists(fp16_output_path):
    file_size = os.path.getsize(fp16_output_path) / (1024*1024)  # MB
    print(f"📊 FP16模型大小: {file_size:.2f} MB")

print("🎯 FP16 ONNX模型轉換完成！")
print(f"模型已儲存到: {fp16_output_path}")
print("🎯 建議更新config.json中的model_path為:")
print(f'    "{fp16_output_path}"')

# 提供測試建議
print("\n💡 測試建議:")
print("1. 使用更新後的 test_onnx_export.py 進行測試")
print("2. 確保推理時輸入的資料型別為 float16")
print("3. GPU環境下FP16可能提供更好的性能")