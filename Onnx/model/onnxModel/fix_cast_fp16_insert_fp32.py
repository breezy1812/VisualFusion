#!/usr/bin/env python3
# fix_cast_fp16_insert_fp32.py
# 用途：自動在把 tensor 轉成 float16 的 Cast 節點後插入 Cast->float (FP32)
# 以解決 ORT 載入時 "Type (tensor(float16)) ... expected (tensor(float))" 的錯誤
# 使用前請備份原檔

import onnx
from onnx import helper, TensorProto, numpy_helper
import sys, os, copy
from collections import defaultdict

SRC = "/circ330/forgithub/VisualFusion_libtorch/Onnx/model/onnxModel/opset19_fp16.onnx"        # 改成你的檔名
DST = "/circ330/forgithub/VisualFusion_libtorch/Onnx/model/onnxModel/opset19_fp16_fixed.onnx"  # 修補後輸出檔名 (會自動備份原檔為 *.bak)

def main():
    if not os.path.exists(SRC):
        print("Source not found:", SRC)
        return 1

    model = onnx.load(SRC)
    graph = model.graph

    # backup
    bak = SRC + ".bak"
    if not os.path.exists(bak):
        print("Backing up original ->", bak)
        onnx.save(model, bak)

    print("Loaded:", SRC)
    print("Nodes before:", len(graph.node))

    # We'll build new nodes list, inserting a Cast->FLOAT after every Cast->FLOAT16
    new_nodes = []
    insert_count = 0

    for node in graph.node:
        new_nodes.append(copy.deepcopy(node))
        # find Cast nodes with attribute to == FLOAT16
        if node.op_type == "Cast":
            # find attribute 'to'
            to_attr = None
            for a in node.attribute:
                if a.name == "to":
                    to_attr = a.i
                    break
            if to_attr == TensorProto.FLOAT16:
                # generate a new name and output name
                # support multiple outputs - handle each output
                for i, out_name in enumerate(node.output):
                    new_out_name = out_name + "_castback_fp32"
                    # create Cast node to FLOAT
                    cast_back_node = helper.make_node(
                        "Cast",
                        inputs=[out_name],
                        outputs=[new_out_name],
                        name=(node.name + "_castback_fp32") if node.name else (out_name + "_castback_fp32"),
                        to=TensorProto.FLOAT
                    )
                    new_nodes.append(cast_back_node)
                    insert_count += 1

                    # Now rewrite consumers in new_nodes so that any node that consumes out_name
                    # (which we already appended or will append later) should be adjusted.
                    # Simpler approach: after building whole new_nodes, we'll scan and replace inputs globally.
                    # For now record mapping:
                    # We'll perform a second pass to rewrite inputs.
    # build mapping of old->new for outputs we created
    # Collect all created cast-back outputs by scanning new_nodes for names ending with '_castback_fp32'
    created_map = {}
    for node in new_nodes:
        if node.op_type == "Cast" and node.name and node.name.endswith("_castback_fp32"):
            if len(node.input) >= 1 and len(node.output) >= 1:
                created_map[node.input[0]] = node.output[0]

    # Second pass: replace inputs in non-cast-back nodes that reference the original out names
    replaced = 0
    for node in new_nodes:
        # skip the inserted cast-back nodes themselves
        if node.op_type == "Cast" and node.name and node.name.endswith("_castback_fp32"):
            continue
        for i, inp in enumerate(node.input):
            if inp in created_map:
                node.input[i] = created_map[inp]
                replaced += 1

    print(f"Inserted {insert_count} Cast-back nodes, rewired {replaced} inputs.")

    # Construct new graph
    new_graph = helper.make_graph(
        nodes=new_nodes,
        name=graph.name + "_force_castback",
        inputs=graph.input,
        outputs=graph.output,
        initializer=graph.initializer,
        value_info=list(graph.value_info)
    )

    new_model = helper.make_model(new_graph)
    # copy metadata and opset_import
    new_model.ir_version = model.ir_version
    new_model.producer_name = model.producer_name + "_force_castback"
    new_model.producer_version = model.producer_version
    new_model.domain = model.domain
    new_model.model_version = model.model_version
    # copy opset imports
    new_model.opset_import.extend(model.opset_import)

    # Add value_info for new tensors (cast-back outputs)
    for old, new_out in created_map.items():
        vi = helper.make_tensor_value_info(new_out, TensorProto.FLOAT, shape=None)
        new_model.graph.value_info.append(vi)

    # validate & save
    try:
        onnx.checker.check_model(new_model)
        print("ONNX checker: passed")
    except Exception as e:
        print("ONNX checker: FAILED:", e)
        print("Saving anyway for inspection.")

    print("Saving fixed model ->", DST)
    onnx.save(new_model, DST)
    print("Done. Please test loading with ONNXRuntime (CPU/CUDA).")

if __name__ == "__main__":
    main()