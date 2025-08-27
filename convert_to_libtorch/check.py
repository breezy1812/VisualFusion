import onnx
import onnx.mapping

def check_onnx_dtype(onnx_path: str):
    model = onnx.load(onnx_path)
    print(f"Checking ONNX model: {onnx_path}")

    # 檢查所有 Input
    for inp in model.graph.input:
        t = inp.type.tensor_type
        if t.elem_type in onnx.mapping.TENSOR_TYPE_TO_NP_TYPE:
            np_dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[t.elem_type]
            print(f"  Input: {inp.name}, dtype = {np_dtype}")
        else:
            print(f"  Input: {inp.name}, dtype = Unknown ({t.elem_type})")

    # 檢查所有 Output
    for out in model.graph.output:
        t = out.type.tensor_type
        if t.elem_type in onnx.mapping.TENSOR_TYPE_TO_NP_TYPE:
            np_dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[t.elem_type]
            print(f"  Output: {out.name}, dtype = {np_dtype}")
        else:
            print(f"  Output: {out.name}, dtype = Unknown ({t.elem_type})")

if __name__ == "__main__":
    onnx_path = "/circ330/forgithub/VisualFusion_libtorch/Onnx/model/onnxModel/zfp16_op12.onnx"  # ← 換成你的模型路徑
    check_onnx_dtype(onnx_path)
