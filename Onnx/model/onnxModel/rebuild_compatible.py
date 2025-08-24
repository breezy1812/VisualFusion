#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完全重建ONNX模型，使用opset 12和IR version 8，確保與ONNX Runtime 1.18.0完全兼容
"""

import onnx
from onnx import TensorProto, helper, numpy_helper
import os
import numpy as np

def rebuild_model_compatible():
    """完全重建模型，確保與ONNX Runtime 1.18.0完全兼容"""
    
    print("🔧 重建兼容模型 - opset 12 + IR version 8")
    print("=" * 60)
    
    fp16_file = './SemLA_onnx_320x240_fp16_cuda.onnx'
    output_file = './SemLA_onnx_320x240_fp16_compatible.onnx'
    
    if not os.path.exists(fp16_file):
        print(f"❌ 找不到檔案: {fp16_file}")
        return False
    
    try:
        # 載入原始模型
        print("📁 載入原始模型...")
        original_model = onnx.load(fp16_file)
        original_graph = original_model.graph
        
        print(f"   原始IR版本: {original_model.ir_version}")
        print(f"   原始opset版本: {[f'{op.domain}:{op.version}' for op in original_model.opset_import]}")
        
        # 收集所有必要組件
        print("📦 收集模型組件...")
        
        # 1. 收集初始化器（權重）
        initializers = []
        for init in original_graph.initializer:
            initializers.append(init)
        
        # 2. 收集value_info（中間張量信息）
        value_infos = []
        for vi in original_graph.value_info:
            value_infos.append(vi)
        
        # 3. 收集節點
        nodes = []
        for node in original_graph.node:
            nodes.append(node)
        
        print(f"   初始化器: {len(initializers)}, value_info: {len(value_infos)}, 節點: {len(nodes)}")
        
        # 4. 重新創建輸入和輸出
        inputs = list(original_graph.input)
        outputs = list(original_graph.output)
        
        print("🏗️  重建圖形...")
        
        # 創建新圖形
        new_graph = helper.make_graph(
            nodes=nodes,
            name="SemLA_compatible_graph",
            inputs=inputs,
            outputs=outputs,
            initializer=initializers,
            value_info=value_infos
        )
        
        # 設置兼容的opset版本
        opset_imports = [
            helper.make_opsetid("", 12)  # 使用opset 12
        ]
        
        print("📦 重建模型...")
        
        # 創建新模型，明確設置IR版本
        new_model = helper.make_model(
            new_graph, 
            opset_imports=opset_imports,
            producer_name="compatible_rebuild",
            producer_version="1.0"
        )
        
        # 強制設置IR版本為8（ONNX Runtime 1.18.0兼容）
        new_model.ir_version = 8
        
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
        print(f"   IR版本: {test_model.ir_version}")
        print(f"   Opset版本: {[f'{op.domain}:{op.version}' for op in test_model.opset_import]}")
        
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
    success = rebuild_model_compatible()
    if success:
        print("\n🎉 成功！模型已重建為ONNX Runtime 1.18.0兼容版本")
        print("   可以更新config.json使用新模型檔案")
    else:
        print("\n💥 失敗！請檢查錯誤信息")
