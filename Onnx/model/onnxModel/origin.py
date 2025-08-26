# convert_to_opset19.py
import onnx
from onnx import version_converter

src = "/circ330/forgithub/VisualFusion_libtorch/Onnx/model/onnxModel/SemLA_onnx_opset17_fixed1200pts_cuda.onnx"
dst = "/circ330/forgithub/VisualFusion_libtorch/Onnx/model/onnxModel/SemLA_onnx_opset19_fixed1200pts_cuda.onnx"
target = 19

m = onnx.load(src)
print("原 opset:", [(o.domain, o.version) for o in m.opset_import])

try:
    m_new = version_converter.convert_version(m, target)
    onnx.save(m_new, dst)
    print("Converted to opset", target, "->", dst)
    onnx.checker.check_model(m_new)
    print("checker passed")
except Exception as e:
    print("轉換失敗:", e)
    raise
