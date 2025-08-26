# convert_fp16_safe.py
import onnx
from onnxconverter_common import float16
import sys

SRC_FP32 = "/circ330/forgithub/VisualFusion_libtorch/Onnx/model/onnxModel/SemLA_onnx_opset19_fixed1200pts_cuda.onnx"
DST_FP16 = "/circ330/forgithub/VisualFusion_libtorch/Onnx/model/onnxModel/opset19_fp16.onnx"

print("Load:", SRC_FP32)
m = onnx.load(SRC_FP32)

print("Converting to FP16 (keep_io_types=True, keep_initializers=True if available)...")
# keep_initializers may or may not be available in your version; if not, it will be ignored.
try:
    m_fp16 = float16.convert_float_to_float16(m, keep_io_types=True, keep_initializers=True)
except TypeError:
    # fallback if keep_initializers not supported
    m_fp16 = float16.convert_float_to_float16(m, keep_io_types=True)

print("Saving:", DST_FP16)
onnx.save(m_fp16, DST_FP16)

print("Running onnx.checker...")
try:
    onnx.checker.check_model(m_fp16)
    print("ONNX checker: OK")
except Exception as e:
    print("ONNX checker failed:", e)
    raise

print("Done. Saved:", DST_FP16)
