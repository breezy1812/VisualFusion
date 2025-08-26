#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完全重建ONNX模型，強制使用opset 12，解決opset 23兼容性問題
"""

import onnx
from onnx import TensorProto, helper, numpy_helper
import os
import numpy as np

def rebuild_model_with_opset12():
    """完全重建模型，強制使用opset 12"""
    
    print("🔧 完全重建模型 - 強制 opset 12")
    print("=" * 60)
    
    fp16_file = './fp16.onnx'
    output_file = './SemLA_onnx_320x240_fp16_opset12.onnx'
    
    if not os.path.exists(fp16_file):
        print(f"❌ 找不到檔案: {fp16_file}")
        return False
    
    try:
        # 載入原始模型
        print("📁 載入原始模型...")
        original_model = onnx.load(fp16_file)
        original_graph = original_model.graph
        
        print(f"   原始opset版本: {[f'{op.domain}:{op.version}' for op in original_model.opset_import]}")
        
        # 收集所有必要組件
        print("📦 收集模型組件...")
        
        # 1. 收集初始化器（權重）
        initializers = []
        for init in original_graph.initializer:
            initializers.append(init)
        
        print(f"   初始化器數量: {len(initializers)}")
        
        # 2. 收集value_info（中間張量信息）
        value_infos = []
        for vi in original_graph.value_info:
            value_infos.append(vi)
        
        print(f"   value_info數量: {len(value_infos)}")
        
        # 3. 收集節點，並確保所有operator都兼容opset 12
        nodes = []
        incompatible_ops = []
        
        for node in original_graph.node:
            # 檢查是否有opset 12不支援的operator
            if node.op_type in ['CastLike', 'ScatterElements', 'GatherElements']:
                print(f"   ⚠️  發現可能不兼容的operator: {node.op_type}")
                incompatible_ops.append(node.op_type)
            
            nodes.append(node)
        
        print(f"   節點數量: {len(nodes)}")
        if incompatible_ops:
            print(f"   可能不兼容的操作: {set(incompatible_ops)}")
        
        # 4. 重新創建輸入
        inputs = []
        for inp in original_graph.input:
            inputs.append(inp)
        
        # 5. 重新創建輸出
        outputs = []
        for out in original_graph.output:
            outputs.append(out)
        
        print("🏗️  重建圖形...")
        
        # 創建新圖形
        new_graph = helper.make_graph(
            nodes=nodes,
            name=original_graph.name + "_opset12",
            inputs=inputs,
            outputs=outputs,
            initializer=initializers,
            value_info=value_infos
        )
        
        # 強制設置opset 12
        opset_imports = [
            helper.make_opsetid("", 17)  # 主要opset設為12
        ]
        
        print("📦 重建模型...")
        
        # 創建新模型
        new_model = helper.make_model(
            new_graph, 
            opset_imports=opset_imports,
            producer_name="rebuilt_opset12",
            producer_version="1.0"
        )
        
        # 設置元數據
        if original_model.model_version:
            new_model.model_version = original_model.model_version
        
        print("🔍 檢查模型...")
        
        # 檢查模型
        try:
            onnx.checker.check_model(new_model)
            print("✅ 模型檢查通過")
        except Exception as e:
            print(f"⚠️  模型檢查警告: {e}")
            print("   繼續保存模型...")
        
        # 保存模型
        print(f"💾 保存到: {output_file}")
        onnx.save(new_model, output_file)
        
        # 驗證結果
        print("\n🧪 驗證結果:")
        test_model = onnx.load(output_file)
        print(f"   新opset版本: {[f'{op.domain}:{op.version}' for op in test_model.opset_import]}")
        
        new_size = os.path.getsize(output_file) / 1024 / 1024
        print(f"   檔案大小: {new_size:.2f} MB")
        
        # 測試能否用ONNX Runtime載入
        print("\n🎯 測試ONNX Runtime載入...")
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(output_file, providers=['CPUExecutionProvider'])
            print("✅ ONNX Runtime成功載入模型")
            
            # 顯示輸入輸出信息
            print("   模型輸入:")
            for inp in session.get_inputs():
                print(f"     {inp.name}: {inp.shape} ({inp.type})")
            print("   模型輸出:")
            for out in session.get_outputs():
                print(f"     {out.name}: {out.shape} ({out.type})")
                
        except Exception as e:
            print(f"❌ ONNX Runtime載入失敗: {e}")
            return False
        
        print("\n✅ 模型重建完成！")
        print(f"   輸出檔案: {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ 重建失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = rebuild_model_with_opset12()
    if success:
        print("\n🎉 成功！模型已重建為opset 12版本")
    else:
        print("\n💥 失敗！請檢查錯誤信息")
