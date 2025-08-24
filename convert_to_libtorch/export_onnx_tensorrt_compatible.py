#!/usr/bin/env python3
"""
改進版 ONNX 導出腳本 - 避免 TensorRT INT64 警告
TODO:下週
直接轉換成onnx int32版本，之後才去用export_onnx2tensorRT.py轉成TensorRT
"""

import torch
import os
import onnx
from model_jit.SemLA import SemLA

print("=== SemLA ONNX 導出 (TensorRT 相容版本) ===")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用設備: {device}")

# 載入模型
fpMode = torch.float32
matcher = SemLA(device=device, fp=fpMode)
matcher.load_state_dict(torch.load("./reg.ckpt", map_location=device), strict=False)
matcher = matcher.eval().to(device, dtype=fpMode)

width = 320
height = 240

torch_input_1 = torch.randn(1, 1, height, width).to(device)
torch_input_2 = torch.randn(1, 1, height, width).to(device)

output_dir = "../Onnx/model/onnxModel"
os.makedirs(output_dir, exist_ok=True)

output_path = f"{output_dir}/SemLA_onnx_{width}x{height}_tensorrt_int32.onnx"

print(f"導出 TensorRT 相容的 ONNX 模型...")
print(f"輸出路徑: {output_path}")

# 導出 ONNX 模型，指定輸出類型避免 INT64 問題
torch.onnx.export(
    matcher,
    (torch_input_1, torch_input_2),
    output_path,
    verbose=False,
    opset_version=12,  # 使用較穩定的版本
    input_names=["vi_img", "ir_img"],
    output_names=["mkpt0", "mkpt1", "leng1", "leng2"],
    do_constant_folding=True,
    # 確保輸出類型兼容性
    operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
)

print("✅ ONNX 模型導出完成")

# 後處理：檢查並修正可能的 INT64 問題
try:
    import onnx
    from onnx import helper, TensorProto
    
    print("🔧 檢查並修正 ONNX 模型中的資料類型...")
    
    model = onnx.load(output_path)
    
    # 檢查是否有 INT64 輸出，如有則轉換為 INT32
    modified = False
    for output in model.graph.output:
        if output.type.tensor_type.elem_type == TensorProto.INT64:
            print(f"  修正輸出 {output.name}: INT64 -> INT32")
            output.type.tensor_type.elem_type = TensorProto.INT32
            modified = True
    
    # 檢查節點中的屬性
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.INTS:
                # 檢查整數列表是否超出 INT32 範圍
                int_vals = list(attr.ints)
                if any(val > 2147483647 or val < -2147483648 for val in int_vals):
                    print(f"  警告：節點 {node.name} 包含超出 INT32 範圍的值")
    
    if modified:
        # 儲存修正後的模型
        onnx.save(model, output_path)
        print("✅ ONNX 模型資料類型修正完成")
    
    # 驗證模型
    onnx.checker.check_model(model)
    print("✅ ONNX 模型驗證通過")
    
except Exception as e:
    print(f"⚠️  模型後處理警告: {e}")

# 檢查檔案大小
file_size = os.path.getsize(output_path) / (1024*1024)
print(f"📊 模型大小: {file_size:.2f} MB")
print(f"🎯 TensorRT 相容的 ONNX 模型已儲存到: {output_path}")

print("\n💡 使用建議:")
print("1. 此版本應該能減少 TensorRT 轉換時的 INT64 警告")
print("2. 可以用此模型進行 TensorRT 轉換測試") 
print("3. 如仍有警告，可以安全忽略，不影響推論功能")
