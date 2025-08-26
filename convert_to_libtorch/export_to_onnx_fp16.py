import torch
import os
import onnx
from onnxconverter_common import float16

from model_jit.SemLA import SemLA

print("=== SemLA ONNX FP16 轉換腳本 ===")

# 使用CUDA來獲得最佳性能
device = torch.device("cuda")
print(f"使用設備: {device}")

# 先以 FP32 載入模型
fpMode = torch.float32
print("正在載入模型...")
matcher = SemLA(device=device, fp=fpMode)
matcher.load_state_dict(torch.load(f"./reg.ckpt", map_location=device), strict=False)
matcher = matcher.eval().to(device, dtype=fpMode)

# 使用與配置文件相符的尺寸
width = 320
height = 240

print(f"建立輸入張量，尺寸: {height}x{width}")
torch_input_1 = torch.randn(1, 1, height, width).to(device, dtype=fpMode)
torch_input_2 = torch.randn(1, 1, height, width).to(device, dtype=fpMode)

# 確保輸出目錄存在
output_dir = "../Onnx/model/onnxModel"
os.makedirs(output_dir, exist_ok=True)

# 先導出FP32 ONNX模型
fp32_output_path = f"{output_dir}/SemLA_onnx_{width}x{height}_fp32_temp.onnx"
fp16_output_path = f"{output_dir}/zETOfp16op12_fp16_{device}.onnx"

print(f"步驟1: 轉換為FP32 ONNX模型...")
print(f"臨時路徑: {fp32_output_path}")

try:
    torch.onnx.export(
        matcher,
        (torch_input_1, torch_input_2),
        fp32_output_path,
        verbose=False,
        opset_version=12,  # 使用較新版本支援更多操作
        input_names=["vi_img", "ir_img"],
        output_names=["mkpt0", "mkpt1", "leng1", "leng2"],
        do_constant_folding=True,
        # 移除dynamic_axes，使用固定尺寸
    )
    print("✅ FP32 ONNX模型轉換完成")
    
    # 驗證FP32模型
    onnx_model = onnx.load(fp32_output_path)
    onnx.checker.check_model(onnx_model)
    print("✅ FP32 ONNX模型驗證通過")
    
except Exception as e:
    print(f"❌ FP32 ONNX轉換失敗: {e}")
    exit(1)

print(f"步驟2: 轉換為FP16 ONNX模型...")
print(f"最終路徑: {fp16_output_path}")

try:
    # 載入FP32模型並轉換為FP16
    fp32_model = onnx.load(fp32_output_path)
    
    # 轉換為FP16，保持輸入為FP32
    fp16_model = float16.convert_float_to_float16(
        fp32_model, 
        keep_io_types=True  # 保持輸入輸出為FP32以提高兼容性
    )
    
    # 儲存FP16模型
    onnx.save(fp16_model, fp16_output_path)
    print("✅ FP16 ONNX模型轉換完成")
    
    # 驗證FP16模型
    onnx.checker.check_model(fp16_model)
    print("✅ FP16 ONNX模型驗證通過")
    
    # 清理臨時檔案
    os.remove(fp32_output_path)
    print("🧹 清理臨時檔案完成")
    
except Exception as e:
    print(f"❌ FP16轉換失敗: {e}")
    print("可能需要安裝: pip install onnxconverter-common")
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
print("1. 更新config.json使用新的FP16模型路徑")
print("2. 確保推論環境支援FP16操作")
print("3. GPU環境下FP16可能提供更好的性能")
print("4. 比較FP32與FP16的推論精度差異")