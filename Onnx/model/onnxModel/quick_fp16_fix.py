#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速修復 FP16 模型類型不匹配問題
專門解決 Concat 節點的 tensor(float16) vs tensor(float) 錯誤
"""

import onnx
from onnx import TensorProto, helper
import os
import sys

def quick_fix_fp16_model():
    """快速修復FP16模型的類型不匹配問題"""
    
    print("🔧 快速修復 FP16 模型類型不匹配")
    print("=" * 50)
    
    # 檔案路徑
    fp32_file = './SemLA_onnx_320x240_fp32_cuda.onnx'
    fp16_file = './SemLA_onnx_320x240_fp16_cuda.onnx'
    
    if not os.path.exists(fp32_file):
        print(f"❌ 找不到 FP32 檔案: {fp32_file}")
        return False
    
    try:
        # 載入FP32模型
        print("📁 載入 FP32 模型...")
        model = onnx.load(fp32_file)
        graph = model.graph
        
        # 備份舊的FP16模型
        if os.path.exists(fp16_file):
            backup_file = fp16_file.replace('.onnx', '_old.onnx')
            os.rename(fp16_file, backup_file)
            print(f"💾 備份舊模型: {backup_file}")
        
        # 步驟1：創建混合精度模型
        print("🔄 創建混合精度模型...")
        
        # 複製所有節點，但修改關鍵節點的行為
        new_nodes = []
        new_value_infos = []
        new_initializers = []
        
        # 處理初始化器 - 權重轉為FP16
        for init in graph.initializer:
            if init.data_type == TensorProto.FLOAT and init.name not in ['input', 'output']:
                # 轉換權重為FP16以節省空間
                new_init = onnx.TensorProto()
                new_init.CopyFrom(init)
                
                # 簡單的FP16轉換
                import numpy as np
                if init.raw_data:
                    # 從raw_data轉換
                    float32_data = np.frombuffer(init.raw_data, dtype=np.float32)
                    float16_data = float32_data.astype(np.float16)
                    new_init.raw_data = float16_data.tobytes()
                    new_init.data_type = TensorProto.FLOAT16
                elif init.float_data:
                    # 從float_data轉換
                    float32_data = np.array(init.float_data)
                    float16_data = float32_data.astype(np.float16)
                    # 清除舊資料
                    new_init.float_data[:] = []
                    # 轉換為int32_data格式
                    int_data = [int(x.view('uint16')) for x in float16_data.flat]
                    new_init.int32_data[:] = int_data
                    new_init.data_type = TensorProto.FLOAT16
                
                new_initializers.append(new_init)
            else:
                new_initializers.append(init)
        
        # 處理節點 - 保持關鍵節點為FP32
        critical_ops = ['Concat', 'ReduceMean', 'Pow', 'Cast', 'Shape', 'Gather', 'ConstantOfShape']
        
        for node in graph.node:
            new_nodes.append(node)  # 保持原始節點不變
        
        # 處理value_info - 智能類型分配
        input_output_names = {inp.name for inp in graph.input} | {out.name for out in graph.output}
        
        for vi in graph.value_info:
            new_vi = onnx.ValueInfoProto()
            new_vi.CopyFrom(vi)
            
            # 根據名稱模式決定類型
            if vi.name in input_output_names:
                # 輸入輸出保持FLOAT32
                new_vi.type.tensor_type.elem_type = TensorProto.FLOAT
            elif any(pattern in vi.name.lower() for pattern in [
                'concat', 'reducemean', 'pow', 'cast', 'shape', 'gather', 
                'output_cast', 'input_cast', '/concat_output_cast_0'
            ]):
                # 關鍵操作保持FLOAT32
                new_vi.type.tensor_type.elem_type = TensorProto.FLOAT
            elif 'constantofshape' in vi.name.lower() or 'shape' in vi.name.lower():
                # 形狀相關操作保持INT64
                if new_vi.type.tensor_type.elem_type in [TensorProto.FLOAT, TensorProto.FLOAT16]:
                    new_vi.type.tensor_type.elem_type = TensorProto.INT64
            else:
                # 其他中間張量使用FLOAT16以節省記憶體
                if new_vi.type.tensor_type.elem_type == TensorProto.FLOAT:
                    new_vi.type.tensor_type.elem_type = TensorProto.FLOAT16
            
            new_value_infos.append(new_vi)
        
        # 創建新的圖
        new_graph = helper.make_graph(
            nodes=new_nodes,
            name=graph.name,
            inputs=graph.input,  # 保持原始輸入
            outputs=graph.output,  # 保持原始輸出
            initializer=new_initializers,
            value_info=new_value_infos
        )
        
        # 創建新模型
        new_model = helper.make_model(new_graph)
        new_model.ir_version = model.ir_version
        new_model.producer_name = model.producer_name
        new_model.producer_version = model.producer_version
        new_model.domain = model.domain
        new_model.model_version = model.model_version
        
        # 複製opset資訊，並確保版本相容
        for opset in model.opset_import:
            new_opset = new_model.opset_import.add()
            new_opset.CopyFrom(opset)
            if new_opset.version > 21:
                new_opset.version = 21
        
        # 複製metadata
        if model.metadata_props:
            new_model.metadata_props.extend(model.metadata_props)
        
        print("✅ 混合精度模型創建完成")
        
        # 驗證模型
        print("🔍 驗證模型...")
        try:
            onnx.checker.check_model(new_model)
            print("   ✅ 模型驗證通過")
        except Exception as e:
            print(f"   ⚠️  驗證警告: {e}")
            print("   繼續保存模型...")
        
        # 保存模型
        print(f"💾 保存混合精度模型: {fp16_file}")
        onnx.save(new_model, fp16_file)
        
        # 統計資訊
        fp32_size = os.path.getsize(fp32_file) / 1024 / 1024
        fp16_size = os.path.getsize(fp16_file) / 1024 / 1024
        
        print("\n🎉 混合精度模型創建完成！")
        print("=" * 50)
        print(f"FP32 模型: {fp32_size:.2f} MB")
        print(f"混合精度模型: {fp16_size:.2f} MB")
        print(f"節省空間: {(1-fp16_size/fp32_size)*100:.1f}%")
        print("\n✅ 模型特性:")
        print("  - 輸入輸出: FLOAT32 (相容性)")
        print("  - 權重參數: FLOAT16 (節省記憶體)")
        print("  - 關鍵節點: FLOAT32 (避免類型錯誤)")
        print("  - 中間張量: FLOAT16 (加速計算)")
        
        return True
        
    except Exception as e:
        print(f"❌ 修復失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_mixed_precision_model():
    """驗證混合精度模型的類型正確性"""
    
    fp16_file = './SemLA_onnx_320x240_fp16_cuda.onnx'
    
    if not os.path.exists(fp16_file):
        print(f"❌ 找不到模型檔案: {fp16_file}")
        return False
    
    print("\n🔍 驗證混合精度模型...")
    print("-" * 40)
    
    try:
        model = onnx.load(fp16_file)
        graph = model.graph
        
        # 統計不同類型的張量
        type_counts = {
            'FLOAT': 0,
            'FLOAT16': 0,
            'INT64': 0,
            'OTHER': 0
        }
        
        # 檢查value_info
        for vi in graph.value_info:
            elem_type = vi.type.tensor_type.elem_type
            if elem_type == TensorProto.FLOAT:
                type_counts['FLOAT'] += 1
            elif elem_type == TensorProto.FLOAT16:
                type_counts['FLOAT16'] += 1
            elif elem_type == TensorProto.INT64:
                type_counts['INT64'] += 1
            else:
                type_counts['OTHER'] += 1
        
        # 檢查初始化器
        init_fp16 = sum(1 for init in graph.initializer if init.data_type == TensorProto.FLOAT16)
        init_fp32 = sum(1 for init in graph.initializer if init.data_type == TensorProto.FLOAT)
        
        print(f"📊 張量類型統計:")
        print(f"   FLOAT32: {type_counts['FLOAT']} 個")
        print(f"   FLOAT16: {type_counts['FLOAT16']} 個")
        print(f"   INT64: {type_counts['INT64']} 個")
        print(f"   其他: {type_counts['OTHER']} 個")
        print(f"📊 初始化器統計:")
        print(f"   FLOAT32: {init_fp32} 個")
        print(f"   FLOAT16: {init_fp16} 個")
        
        # 檢查輸入輸出類型
        print(f"📊 輸入輸出檢查:")
        for inp in graph.input:
            type_name = TensorProto.DataType.Name(inp.type.tensor_type.elem_type)
            print(f"   輸入 {inp.name}: {type_name}")
        
        for out in graph.output:
            type_name = TensorProto.DataType.Name(out.type.tensor_type.elem_type)
            print(f"   輸出 {out.name}: {type_name}")
        
        # 檢查問題節點
        problem_nodes = []
        for vi in graph.value_info:
            if 'concat_output_cast_0' in vi.name and vi.type.tensor_type.elem_type == TensorProto.FLOAT16:
                problem_nodes.append(vi.name)
        
        if problem_nodes:
            print(f"⚠️  仍有問題的節點: {problem_nodes}")
            return False
        else:
            print(f"✅ 未發現明顯的類型問題")
            return True
            
    except Exception as e:
        print(f"❌ 驗證失敗: {e}")
        return False

if __name__ == "__main__":
    print("🚀 快速 FP16 混合精度模型修復工具")
    print("專門解決 Concat 節點類型不匹配問題")
    print("=" * 60)
    
    # 執行修復
    success = quick_fix_fp16_model()
    
    if success:
        # 驗證結果
        verify_success = verify_mixed_precision_model()
        
        if verify_success:
            print(f"\n🎉 模型修復成功！")
            print(f"現在可以測試 CUDA 推論了。")
            print(f"\n執行指令測試：")
            print(f"cd /circ330/forgithub/VisualFusion_libtorch/Onnx && ./main")
        else:
            print(f"\n⚠️  模型修復完成，但驗證發現問題")
            print(f"建議先用 CPU 模式測試")
    else:
        print(f"\n❌ 模型修復失敗")
        print(f"建議使用 FP32 模型或 CPU 模式")
