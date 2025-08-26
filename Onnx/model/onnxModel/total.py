#!/usr/bin/env python3
"""
fp32_to_fp16_opset19_pipeline.py

單一腳本完成：
- 確保 ONNX opset <= 19（若必要，嘗試 version_converter）
- FP32 -> 安全 FP16 轉換（keep_io_types=True, keep_initializers=True if available）
- 若 ONNXRuntime 載入失敗（type mismatch），在敏感 ops 後自動插入 Cast->FLOAT（可設定 op list）
- 檢查 (onnx.checker) 並嘗試載入 ONNX Runtime (CPU & CUDA)

使用：
python3 fp32_to_fp16_opset19_pipeline.py --src SemLA_onnx_fp32.onnx

輸出：
- <src_basename>_op19_fp16.onnx       # 初次轉換結果（若成功）
- <src_basename>_op19_fp16_fixed.onnx # 若需要修補後的結果
"""

import os
import sys
import argparse
import shutil
import copy
import onnx
from onnx import helper, TensorProto
from onnx import version_converter
import traceback

# Try import float16 converter
try:
    from onnxconverter_common import float16 as oc_float16
except Exception:
    oc_float16 = None

# Optional ONNX Runtime import for load testing (may not be installed)
try:
    import onnxruntime as ort
except Exception:
    ort = None

def backup_file(path):
    bak = path + ".orig.bak"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
    return bak

def print_opset_info(model):
    return [(o.domain, o.version) for o in model.opset_import]

def ensure_opset_19(model_path, target_path):
    """
    Load model_path, if ai.onnx opset > 19 then try to convert to 19,
    otherwise ensure ai.onnx opset set to 19 (force header).
    Returns loaded model (opset adjusted) and a flag whether conversion occurred.
    """
    print(f"[1] Load ONNX: {model_path}")
    model = onnx.load(model_path)
    orig = print_opset_info(model)
    print("    original opset_import:", orig)

    # find ai.onnx (domain == "" or "ai.onnx") version
    ai_versions = [o.version for o in model.opset_import if (o.domain == "" or o.domain == "ai.onnx")]
    ai_version = ai_versions[0] if ai_versions else None

    if ai_version is None:
        print("    No ai.onnx opset entry found. Adding opset 19.")
        # add opset import
        new_op = model.opset_import.add()
        new_op.domain = ""
        new_op.version = 19
        onnx.save(model, target_path)
        return model, True

    if ai_version <= 19:
        print("    ai.onnx opset <= 19, nothing to convert.")
        # ensure explicit set to 19 if less? prefer to keep (we can force later if desired)
        # Save to target_path for clarity
        onnx.save(model, target_path)
        return model, (ai_version != 19)

    # ai_version > 19: try version_converter
    print(f"    ai.onnx opset {ai_version} > 19: attempting version_converter -> 19")
    try:
        m_conv = version_converter.convert_version(model, 19)
        print("    version_converter succeeded.")
        onnx.save(m_conv, target_path)
        return m_conv, True
    except Exception as e:
        print("    version_converter failed:", e)
        print("    Will attempt to force header opset to 19 and continue (may still fail at load time).")
        # Force header change
        # modify in place: ensure single ai.onnx entry with version 19
        # Rebuild opset_import
        new_opset = {}
        for op in model.opset_import:
            dom = op.domain if op.domain is not None else ""
            if dom == "" or dom == "ai.onnx":
                new_opset[""] = 19
            else:
                new_opset[dom] = max(new_opset.get(dom, 0), op.version)
        if "" not in new_opset:
            new_opset[""] = 19
        # replace opset_import
        del model.opset_import[:]
        for dom, ver in new_opset.items():
            new = model.opset_import.add()
            new.domain = dom
            new.version = ver
        onnx.save(model, target_path)
        return model, True

def convert_fp16_safe(fp32_path, fp16_path):
    """
    Convert FP32 (opset19) model to FP16 using onnxconverter_common with safe flags.
    Returns FP16 model (onnx.ModelProto).
    """
    if oc_float16 is None:
        raise RuntimeError("onnxconverter_common.float16 is not available in this environment. Install onnxconverter-common.")
    print("[2] Converting FP32 -> FP16 (safe)...")
    m = onnx.load(fp32_path)
    # Try with keep_initializers if available
    try:
        m16 = oc_float16.convert_float_to_float16(m, keep_io_types=True, keep_initializers=True)
        print("    convert_float_to_float16(..., keep_io_types=True, keep_initializers=True) succeeded.")
    except TypeError as e:
        # fallback
        print("    keep_initializers unsupported, fallback to keep_io_types only.")
        m16 = oc_float16.convert_float_to_float16(m, keep_io_types=True)
    return m16

def save_model_and_check(model, out_path):
    print(f"[save] Saving to {out_path} ...")
    onnx.save(model, out_path)
    print("    Running onnx.checker.check_model ...")
    try:
        onnx.checker.check_model(model)
        print("    onnx.checker: PASSED")
        return True
    except Exception as e:
        print("    onnx.checker: FAILED ->", e)
        return False

def try_onnxruntime_load(path):
    if ort is None:
        print("[ORT] onnxruntime not installed; skipping load test.")
        return {"cpu": False, "cuda": False, "error": "onnxruntime not available"}
    results = {}
    print("[3] Testing ONNX Runtime load (CPU)")
    try:
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        results["cpu"] = True
        print("    CPU load OK. providers:", sess.get_providers())
    except Exception as e:
        results["cpu"] = False
        results["cpu_error"] = str(e)
        print("    CPU load FAILED:", e)
    print("Testing ONNX Runtime load (CUDA)")
    try:
        sess = ort.InferenceSession(path, providers=["CUDAExecutionProvider","CPUExecutionProvider"])
        results["cuda"] = True
        print("    CUDA load OK. providers:", sess.get_providers())
    except Exception as e:
        results["cuda"] = False
        results["cuda_error"] = str(e)
        print("    CUDA load FAILED:", e)
    return results

def build_consumer_map(graph):
    consumers = {}
    for node in graph.node:
        for idx, name in enumerate(node.input):
            consumers.setdefault(name, []).append((node, idx))
    return consumers

def insert_cast_back_for_ops(model, op_types):
    """
    For each node in model.graph.node whose op_type is in op_types AND has at least one output,
    insert a Cast node after it (Cast to FLOAT) and rewire all consumers to consume the new Cast output.
    This is an aggressive fix to resolve float16->float32 mismatch at consumers.
    Returns modified model and count of insertions.
    """
    graph = model.graph
    consumers = build_consumer_map(graph)

    new_nodes = []
    created_map = {}  # old_output -> new_output
    insert_count = 0

    # copy nodes, insert cast-back nodes immediately after each matching node
    for node in graph.node:
        new_nodes.append(copy.deepcopy(node))
        if node.op_type in op_types:
            # For each output of this node (usually 1)
            for out_name in node.output:
                new_out = out_name + "_castback_fp32"
                # create Cast node to FLOAT
                cast_node_name = (node.name + "_castback_fp32") if node.name else (out_name + "_castback_fp32")
                cast_node = helper.make_node(
                    "Cast",
                    inputs=[out_name],
                    outputs=[new_out],
                    name=cast_node_name,
                    to=TensorProto.FLOAT
                )
                new_nodes.append(cast_node)
                created_map[out_name] = new_out
                insert_count += 1

    # Rewire: replace inputs in nodes (skip cast-back nodes)
    replaced = 0
    for node in new_nodes:
        if node.op_type == "Cast" and node.name and node.name.endswith("_castback_fp32"):
            continue
        for i, inp in enumerate(node.input):
            if inp in created_map:
                node.input[i] = created_map[inp]
                replaced += 1

    # Add value_info for created outputs (mark as FLOAT)
    for old, new in created_map.items():
        vi = helper.make_tensor_value_info(new, TensorProto.FLOAT, shape=None)
        model.graph.value_info.append(vi)

    # Build new graph
    new_graph = helper.make_graph(
        nodes=new_nodes,
        name=graph.name + "_castback_fixed",
        inputs=graph.input,
        outputs=graph.output,
        initializer=graph.initializer,
        value_info=list(model.graph.value_info)
    )

    new_model = helper.make_model(new_graph)
    # copy metadata & opset_import
    new_model.ir_version = model.ir_version
    new_model.producer_name = model.producer_name + "_castback_fixed"
    new_model.producer_version = model.producer_version
    new_model.domain = model.domain
    new_model.model_version = model.model_version
    # copy opset_import
    del new_model.opset_import[:]
    for op in model.opset_import:
        new_entry = new_model.opset_import.add()
        new_entry.domain = op.domain
        new_entry.version = op.version

    return new_model, insert_count, replaced

def main():
    p = argparse.ArgumentParser(description="FP32->FP16 + opset19 pipeline with auto-fix for type mismatches")
    p.add_argument("--src", default='/circ330/forgithub/VisualFusion_libtorch/Onnx/model/onnxModel/SemLA_onnx_opset17_fixed1200pts_cuda.onnx', help="Source FP32 ONNX model path")
    p.add_argument("--out", default=None, help="Output FP16 model path (optional)")
    p.add_argument("--fix-ops", default="Concat,Add,Softmax,ReduceMax,ReduceMean,ReduceSum,Cast,Mul,Div", help="Comma-separated op types to auto-insert Cast-back after")
    p.add_argument("--no-auto-fix", action="store_true", help="Do not auto-insert cast-back nodes even if loader fails")
    p.add_argument("--skip-ort-test", action="store_true", help="Skip onnxruntime load test (may be missing)")
    args = p.parse_args()

    src = args.src
    if not os.path.exists(src):
        print("Source file not found:", src); sys.exit(1)

    base = os.path.splitext(os.path.basename(src))[0]
    workspace = os.path.dirname(src) or "."

    # filenames
    fp32_op19_path = os.path.join(workspace, base + "_op19_fp32.onnx")
    fp16_path = args.out or os.path.join(workspace, base + "_op19_fp16_total.onnx")
    fp16_fixed_path = os.path.join(workspace, base + "_op19_fp16_fixed_total.onnx")

    print("=== FP32 -> FP16 + opset19 pipeline ===")
    print("Source:", src)
    print("Intermediate FP32(op19):", fp32_op19_path)
    print("FP16 target:", fp16_path)
    print("FP16 fixed (if needed):", fp16_fixed_path)

    # Step 0: backup
    bak = backup_file(src)
    print("Backup of source created at:", bak)

    # Step 1: ensure opset <=19 / set to 19 (try conversion)
    try:
        model_op19, changed = ensure_opset_19(src, fp32_op19_path)
        print("Opset check/convert done. Saved:", fp32_op19_path)
        print("opset_import now:", print_opset_info(model_op19))
    except Exception as e:
        print("Failed to ensure opset19:", e)
        traceback.print_exc()
        sys.exit(1)

    # Step 2: convert to FP16
    try:
        m_fp16 = convert_fp16_safe(fp32_op19_path, fp16_path)
        # save and checker
        save_ok = save_model_and_check(m_fp16, fp16_path)
    except Exception as e:
        print("FP16 conversion failed:", e)
        traceback.print_exc()
        sys.exit(1)

    # Step 3: test with ONNX Runtime
    ort_results = None
    if not args.skip_ort_test:
        ort_results = try_onnxruntime_load(fp16_path)

    # If ort load failed due to type mismatch, optionally auto-fix
    need_fix = False
    if ort_results:
        cpu_ok = ort_results.get("cpu", False)
        cuda_ok = ort_results.get("cuda", False)
        if not cpu_ok or not cuda_ok:
            # Inspect errors for "Type (tensor(float16)) ... expected type (tensor(float))"
            # If any error messages contain this phrase, mark need_fix
            errors = []
            if not cpu_ok:
                errors.append(ort_results.get("cpu_error", ""))
            if not cuda_ok:
                errors.append(ort_results.get("cuda_error", ""))
            combined = "\n".join(errors)
            if "does not match expected type (tensor(float))" in combined or "expected type (tensor(float))" in combined:
                need_fix = True

    # Auto-fix if needed and not disabled
    if need_fix and not args.no_auto_fix:
        print("[4] Detected type-mismatch errors -> performing auto fix by inserting Cast-back after ops:", args.fix_ops)
        op_types = [s.strip() for s in args.fix_ops.split(",") if s.strip()]
        # load the current fp16 model from file to ensure consistent state
        model_fp16_loaded = onnx.load(fp16_path)
        fixed_model, inserted, replaced = insert_cast_back_for_ops(model_fp16_loaded, op_types)
        print(f"    Inserted {inserted} cast-back nodes; rewired {replaced} inputs.")
        # save and check
        save_model_and_check(fixed_model, fp16_fixed_path)
        # test load again
        if not args.skip_ort_test:
            ort_results_fixed = try_onnxruntime_load(fp16_fixed_path)
            print("ORT results after fix:", ort_results_fixed)
        else:
            print("Skipping ORT load test on fixed model.")
        print("Pipeline finished. Fixed file:", fp16_fixed_path)
        return

    print("No auto-fix needed or disabled. Pipeline finished. FP16 file:", fp16_path)
    if ort_results:
        print("ORT load results:", ort_results)

if __name__ == "__main__":
    main()
