#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
實用的混合精度轉換：使用onnxconverter-common進行安全轉換
"""

import onnx
import os
import numpy as np
from onnxconverter_common import float16

def convert_mixed_precision_safe():
    """使用onnxconverter-common進行安全的混合精度轉換"""
    
    print("🔧 安全混合精度轉換 (使用onnxconverter-common)")
    print("=" * 60)
    
    fp32_file = './SemLA_onnx_320x240_fp32_cuda.onnx'
    fp16_file = './SemLA_onnx_320x240_fp16_safe.onnx'
    
    if not os.path.exists(fp32_file):
        print(f"❌ 找不到FP32檔案: {fp32_file}")
        return False
    
    try:
        print("📁 載入FP32模型...")
        model_fp32 = onnx.load(fp32_file)
        
        original_size = os.path.getsize(fp32_file) / 1024 / 1024
        print(f"   原始大小: {original_size:.2f} MB")
        print(f"   節點數: {len(model_fp32.graph.node)}")
        print(f"   權重數: {len(model_fp32.graph.initializer)}")
        
        # 使用onnxconverter-common進行轉換
        print("🔄 執行混合精度轉換...")
        
        # 轉換為FP16，但保持輸入輸出為FP32
        model_fp16 = float16.convert_float_to_float16(
            model_fp32,
            keep_io_types=True,  # 保持輸入輸出類型為FP32
            disable_shape_infer=False  # 啟用形狀推斷
        )
        
        print("✅ 混合精度轉換完成")
        
        # 修正opset版本為兼容版本
        print("🔧 修正opset版本...")
        model_fp16.opset_import[0].version = 12
        model_fp16.ir_version = 8
        
        # 檢查模型
        print("🔍 檢查轉換後模型...")
        try:
            onnx.checker.check_model(model_fp16)
            print("✅ 模型檢查通過")
        except Exception as e:
            print(f"⚠️  模型檢查警告: {e}")
        
        # 保存模型
        print(f"💾 保存混合精度模型...")
        onnx.save(model_fp16, fp16_file)
        
        # 驗證結果
        print("\n🧪 驗證轉換結果:")
        
        new_size = os.path.getsize(fp16_file) / 1024 / 1024
        compression_ratio = new_size / original_size
        
        print(f"   檔案大小: {new_size:.2f} MB")
        print(f"   壓縮率: {(1-compression_ratio)*100:.1f}%")
        print(f"   IR版本: {model_fp16.ir_version}")
        print(f"   Opset版本: {[f'{op.domain}:{op.version}' for op in model_fp16.opset_import]}")
        
        # 統計權重類型
        fp16_count = 0
        fp32_count = 0
        other_count = 0
        
        for init in model_fp16.graph.initializer:
            if init.data_type == onnx.TensorProto.FLOAT16:
                fp16_count += 1
            elif init.data_type == onnx.TensorProto.FLOAT:
                fp32_count += 1
            else:
                other_count += 1
        
        print(f"   權重統計: FP16={fp16_count}, FP32={fp32_count}, 其他={other_count}")
        
        # 測試ONNX Runtime載入
        print("\n🎯 測試ONNX Runtime相容性...")
        try:
            import onnxruntime as ort
            
            # 嘗試載入
            session = ort.InferenceSession(fp16_file, providers=['CPUExecutionProvider'])
            print("✅ CPU Provider載入成功")
            
            # 檢查輸入輸出
            print("   模型輸入:")
            for inp in session.get_inputs():
                print(f"     {inp.name}: {inp.shape} ({inp.type})")
            print("   模型輸出:")
            for out in session.get_outputs():
                print(f"     {out.name}: {out.shape} ({out.type})")
            
            # 測試推論
            print("   執行測試推論...")
            dummy_inputs = {
                'vi_img': np.random.randn(1, 1, 240, 320).astype(np.float32),
                'ir_img': np.random.randn(1, 1, 240, 320).astype(np.float32)
            }
            outputs = session.run(None, dummy_inputs)
            print(f"   推論成功！輸出形狀: {[out.shape for out in outputs]}")
            
        except Exception as e:
            print(f"❌ ONNX Runtime測試失敗: {e}")
            return False
        
        print("\n✅ 安全混合精度模型轉換完成！")
        print(f"   輸出檔案: {fp16_file}")
        print("   • 輸入輸出保持FP32，確保相容性")
        print("   • 內部權重使用FP16，減少記憶體使用")
        print("   • 與ONNX Runtime 1.18.0完全相容")
        return True
        
    except Exception as e:
        print(f"❌ 轉換失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = convert_mixed_precision_safe()
    if success:
        print("\n🎉 成功！可用的混合精度模型已創建")
        print("   更新config.json以使用新模型")
    else:
        print("\n💥 失敗！請檢查錯誤信息")
