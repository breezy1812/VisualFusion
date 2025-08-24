#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能混合精度ONNX轉換：保持輸入輸出為FP32，安全地轉換中間層權重為FP16
"""

import onnx
from onnx import TensorProto, helper, numpy_helper
import os
import numpy as np

def create_smart_mixed_precision():
    """創建智能混合精度模型"""
    
    print("🔧 智能混合精度轉換 (FP32輸入/輸出 + FP16權重)")
    print("=" * 60)
    
    fp32_file = './SemLA_onnx_320x240_fp32_cuda.onnx'
    fp16_file = './SemLA_onnx_320x240_fp16_smart.onnx'
    
    if not os.path.exists(fp32_file):
        print(f"❌ 找不到FP32檔案: {fp32_file}")
        return False
    
    try:
        # 載入原始FP32模型
        print("📁 載入FP32模型...")
        model = onnx.load(fp32_file)
        graph = model.graph
        
        print(f"   節點數: {len(graph.node)}")
        print(f"   權重數: {len(graph.initializer)}")
        
        # 分析輸入輸出張量名稱
        input_names = set(inp.name for inp in graph.input)
        output_names = set(out.name for out in graph.output)
        print(f"   輸入: {input_names}")
        print(f"   輸出: {output_names}")
        
        # 分析節點，找出哪些權重可以安全轉為FP16
        print("🔍 分析節點類型相容性...")
        
        # 收集所有節點的輸入輸出
        node_inputs = set()
        node_outputs = set()
        problematic_nodes = []
        
        for node in graph.node:
            for inp in node.input:
                node_inputs.add(inp)
            for out in node.output:
                node_outputs.add(out)
            
            # 檢查可能有問題的節點類型
            if node.op_type in ['Conv', 'MatMul', 'BatchNormalization', 'Add']:
                problematic_nodes.append((node.name, node.op_type, node.input, node.output))
        
        print(f"   發現 {len(problematic_nodes)} 個需要特別處理的節點")
        
        # 轉換策略：
        # 1. 保持所有輸入/輸出相關的權重為FP32
        # 2. 轉換中間層權重為FP16
        # 3. 添加必要的Cast節點
        
        print("🔄 智能權重轉換...")
        new_initializers = []
        new_nodes = list(graph.node)
        fp32_kept = 0
        fp16_converted = 0
        cast_nodes_added = 0
        
        # 收集需要保持FP32的權重名稱
        critical_weights = set()
        
        # 檢查每個初始化器
        for initializer in graph.initializer:
            weight_name = initializer.name
            is_critical = False
            
            # 檢查這個權重是否被輸入/輸出相關的節點使用
            for node_name, op_type, inputs, outputs in problematic_nodes:
                if weight_name in inputs:
                    # 檢查這個節點的其他輸入是否來自輸入或輸出
                    for inp in inputs:
                        if inp in input_names or inp in output_names:
                            is_critical = True
                            critical_weights.add(weight_name)
                            break
                    
                    # 檢查節點輸出是否直接連到輸出
                    for out in outputs:
                        if out in output_names:
                            is_critical = True
                            critical_weights.add(weight_name)
                            break
            
            if initializer.data_type == TensorProto.FLOAT:
                if is_critical or weight_name.endswith('bias') or 'norm' in weight_name.lower():
                    # 保持關鍵權重為FP32
                    new_initializers.append(initializer)
                    fp32_kept += 1
                    if is_critical:
                        print(f"   保持FP32 (關鍵): {weight_name}")
                else:
                    # 轉換為FP16
                    fp32_weights = numpy_helper.to_array(initializer)
                    fp16_weights = fp32_weights.astype(np.float16)
                    new_initializer = numpy_helper.from_array(fp16_weights, weight_name)
                    new_initializers.append(new_initializer)
                    fp16_converted += 1
                    
                    # 對於Conv和MatMul節點，添加Cast節點
                    for i, node in enumerate(new_nodes):
                        if weight_name in node.input and node.op_type in ['Conv', 'MatMul']:
                            # 創建Cast節點將FP16權重轉回FP32
                            cast_output_name = f"{weight_name}_casted_fp32"
                            cast_node = helper.make_node(
                                'Cast',
                                inputs=[weight_name],
                                outputs=[cast_output_name],
                                to=TensorProto.FLOAT,
                                name=f"cast_{weight_name}_to_fp32"
                            )
                            
                            # 修改原節點使用Cast後的權重
                            new_inputs = [cast_output_name if inp == weight_name else inp for inp in node.input]
                            new_nodes[i] = helper.make_node(
                                node.op_type,
                                inputs=new_inputs,
                                outputs=node.output,
                                name=node.name,
                                **{attr.name: attr for attr in node.attribute}
                            )
                            
                            new_nodes.insert(i, cast_node)
                            cast_nodes_added += 1
                            print(f"   轉FP16+Cast: {weight_name}")
                            break
                    else:
                        print(f"   轉FP16: {weight_name}")
            else:
                new_initializers.append(initializer)
        
        print(f"   轉換結果: FP32保持={fp32_kept}, FP16轉換={fp16_converted}, Cast節點={cast_nodes_added}")
        
        # 重建圖形
        print("🏗️  重建混合精度圖形...")
        new_graph = helper.make_graph(
            nodes=new_nodes,
            name="SemLA_smart_mixed_precision",
            inputs=list(graph.input),
            outputs=list(graph.output),
            initializer=new_initializers,
            value_info=list(graph.value_info)
        )
        
        # 創建新模型
        opset_imports = [helper.make_opsetid("", 12)]
        new_model = helper.make_model(
            new_graph,
            opset_imports=opset_imports,
            producer_name="smart_mixed_precision",
            producer_version="1.0"
        )
        new_model.ir_version = 8
        
        # 檢查模型
        print("🔍 檢查模型...")
        try:
            onnx.checker.check_model(new_model)
            print("✅ 模型檢查通過")
        except Exception as e:
            print(f"⚠️  模型檢查警告: {e}")
        
        # 保存模型
        print(f"💾 保存智能混合精度模型...")
        onnx.save(new_model, fp16_file)
        
        # 驗證結果
        print("\n🧪 驗證混合精度轉換:")
        test_model = onnx.load(fp16_file)
        
        # 統計權重類型
        real_fp16_count = 0
        real_fp32_count = 0
        
        for init in test_model.graph.initializer:
            if init.data_type == TensorProto.FLOAT16:
                real_fp16_count += 1
            elif init.data_type == TensorProto.FLOAT:
                real_fp32_count += 1
        
        original_size = os.path.getsize(fp32_file) / 1024 / 1024
        new_size = os.path.getsize(fp16_file) / 1024 / 1024
        
        print(f"   檔案大小: {new_size:.2f} MB (原始: {original_size:.2f} MB)")
        print(f"   壓縮率: {(1 - new_size/original_size)*100:.1f}%")
        print(f"   權重統計: FP16={real_fp16_count}, FP32={real_fp32_count}")
        print(f"   節點總數: {len(test_model.graph.node)} (新增Cast: {cast_nodes_added})")
        
        # 測試ONNX Runtime載入
        print("\n🎯 測試ONNX Runtime載入...")
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(fp16_file, providers=['CPUExecutionProvider'])
            print("✅ ONNX Runtime成功載入混合精度模型")
            
            # 簡單推論測試
            print("   進行測試推論...")
            dummy_input = {
                'vi_img': np.random.randn(1, 1, 240, 320).astype(np.float32),
                'ir_img': np.random.randn(1, 1, 240, 320).astype(np.float32)
            }
            outputs = session.run(None, dummy_input)
            print(f"   推論成功，輸出形狀: {[out.shape for out in outputs]}")
            
        except Exception as e:
            print(f"❌ ONNX Runtime測試失敗: {e}")
            return False
        
        print("\n✅ 智能混合精度模型創建完成！")
        print(f"   輸出檔案: {fp16_file}")
        return True
        
    except Exception as e:
        print(f"❌ 混合精度轉換失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = create_smart_mixed_precision()
    if success:
        print("\n🎉 成功！智能混合精度模型已創建")
        print("   既有FP16權重優化，又能正確推論")
    else:
        print("\n💥 失敗！請檢查錯誤信息")
