# check_opset.py
import onnx, sys
files = [
    # "SemLA_onnx_opset19_fixed1200pts_cuda.onnx",   # 你的 FP32 op19 原本檔（示範）
    # "opset19_fp16.onnx",                           # 你轉出來的 fp16 檔
    # "opset19_fp16_fixed.onnx",                     # 你修補後的檔（現在會報錯）
    # "opset19_fp16_fixed_op19_only.onnx",
    "SemLA_onnx_opset19_fixed1200pts_cuda.onnx",
    
]
# 只檢查實際存在的檔案
for f in files:
    try:
        m = onnx.load(f)
        print(f"--- {f} ---")
        print("opset_import:", [(o.domain, o.version) for o in m.opset_import])
    except Exception as e:
        print(f"Could not load {f}: {e}")
