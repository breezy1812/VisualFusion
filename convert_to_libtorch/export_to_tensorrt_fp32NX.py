#!/usr/bin/env python3
"""
PyTorch to TensorRT FP32 Conversion Script
從 PyTorch 模型轉換為 TensorRT FP32 引擎（使用 trtexec，更穩定）

基於: build_trt_engine_fp32.sh
使用 trtexec 命令列工具（比 Python API 更穩定）

Usage:
    python export_onnx2tensorRT.py
"""

import os
import sys
import subprocess
import argparse

def main():
    """固定輸出 FP32 TensorRT 模型（使用 trtexec）"""
    parser = argparse.ArgumentParser(description='Convert PyTorch model to TensorRT FP32 engine using trtexec')
    parser.add_argument('--model', type=str,
                       default='./reg.ckpt',
                       help='Path to PyTorch model checkpoint (default: ./reg.ckpt)')
    parser.add_argument('--onnx-output', type=str, 
                       default='../tensorRT_nx/model/NX/Nxfp32.onnx',
                       help='Path to temporary ONNX file')
    parser.add_argument('--trt-output', type=str, 
                       default='../tensorRT_nx/model/NX/trt_Nx_fp32.engine',
                       help='Path to output TensorRT engine')
    parser.add_argument('--opset', type=int, default=12,
                       help='ONNX opset version (default: 12)')
    parser.add_argument('--trtexec-path', type=str,
                       default='/usr/src/tensorrt/bin/trtexec',
                       help='Path to trtexec binary')

    args = parser.parse_args()

    print("=" * 70)
    print("🎯 PyTorch to TensorRT FP32 Conversion Tool (using trtexec)")
    print("=" * 70)
    print("📋 Configuration:")
    print(f"  🧠 PyTorch model: {args.model}")
    print(f"  📄 ONNX output: {args.onnx_output}")
    print(f"  💾 TRT engine: {args.trt_output}")
    print(f"  🚀 Precision: FP32 (TF32 disabled)")
    print(f"  📦 ONNX OpSet: {args.opset}")
    print(f"  🔧 trtexec: {args.trtexec_path}")
    print("=" * 70)
    
    # 步驟 1: PyTorch → ONNX
    print("\n" + "=" * 70)
    print("步驟 1/2: 轉換 PyTorch → ONNX...")
    print("=" * 70)
    
    if not convert_pytorch_to_onnx(args.model, args.onnx_output, args.opset):
        print("\n❌ PyTorch → ONNX 轉換失敗")
        return 1
    
    # 步驟 2: ONNX → TensorRT (使用 trtexec)
    print("\n" + "=" * 70)
    print("步驟 2/2: 轉換 ONNX → TensorRT (使用 trtexec)...")
    print("=" * 70)
    
    if not convert_onnx_to_trt(args.onnx_output, args.trt_output, args.trtexec_path):
        print("\n❌ ONNX → TensorRT 轉換失敗")
        return 1
    
    print("\n" + "=" * 70)
    print("✅ 轉換完成！")
    print("=" * 70)
    print(f"📌 TensorRT FP32 engine: {args.trt_output}")
    print("🔧 Update your configuration files to use this new engine.")
    print("=" * 70)
    
    return 0

def convert_pytorch_to_onnx(model_path, onnx_path, opset_version):
    """步驟 1: 使用 Python 將 PyTorch 轉換為 ONNX"""
    
    python_script = f'''
import os
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

import torch
import onnx
import onnxsim

# 設定確定性
torch.manual_seed(42)
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if hasattr(torch.backends.cuda, 'matmul'):
    torch.backends.cuda.matmul.allow_tf32 = False
if hasattr(torch.backends.cudnn, 'allow_tf32'):
    torch.backends.cudnn.allow_tf32 = False

print("✅ TF32 已禁用 (NVIDIA_TF32_OVERRIDE=0)")

# 載入模型
from model_jit.SemLA import SemLA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fpMode = torch.float32

print(f"使用設備: {{device}}")
print("正在載入模型 (FP32)...")

matcher = SemLA(device=device, fp=fpMode)
matcher.load_state_dict(torch.load("{model_path}", map_location=device), strict=False)
matcher = matcher.eval().to(device, dtype=fpMode)
torch.set_grad_enabled(False)

print("✅ 模型已載入 (FP32)")

# 建立輸入
width = 320
height = 240
torch_input_1 = torch.randn(1, 1, height, width).to(device, dtype=fpMode)
torch_input_2 = torch.randn(1, 1, height, width).to(device, dtype=fpMode)

# 轉換為 ONNX
onnx_path = "{onnx_path}"
print(f"轉換為 ONNX (OpSet {opset_version})...")

# 確保輸出目錄存在
os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

torch.onnx.export(
    matcher,
    (torch_input_1, torch_input_2),
    onnx_path,
    verbose=False,
    opset_version={opset_version},
    input_names=["vi_img", "ir_img"],
    output_names=["mkpt0", "mkpt1"],
    do_constant_folding=True,
    dynamic_axes=None,
)
print(f"✅ ONNX 轉換完成")

# 簡化 ONNX
print("簡化 ONNX 模型...")
model_onnx = onnx.load(onnx_path)
model_simp, check = onnxsim.simplify(model_onnx)
if not check:
    print("⚠️  ONNX 簡化失敗，使用未簡化版本")
    model_simp = model_onnx

onnx.save(model_simp, onnx_path)
print(f"✅ ONNX 已儲存: {{onnx_path}}")

# 驗證 ONNX
try:
    onnx.checker.check_model(model_simp)
    print("✅ ONNX 驗證通過")
except Exception as e:
    print(f"⚠️  ONNX 驗證警告: {{e}}")

# 列出所有運算符
ops = set()
for node in model_simp.graph.node:
    ops.add(node.op_type)
print(f"\\n📋 ONNX 運算符 ({{len(ops)}} 種):")
for op in sorted(ops):
    print(f"  - {{op}}")
'''
    
    try:
        # 執行 Python 腳本
        result = subprocess.run(
            ['python3', '-c', python_script],
            capture_output=False,
            text=True,
            check=True
        )
        
        # 檢查 ONNX 檔案是否存在
        if not os.path.exists(onnx_path):
            print(f"❌ ONNX 檔案不存在: {onnx_path}")
            return False
            
        print(f"✅ ONNX 檔案已建立: {onnx_path}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ PyTorch → ONNX 轉換失敗")
        print(f"   錯誤: {e}")
        return False
    except Exception as e:
        print(f"❌ 發生錯誤: {e}")
        return False

def convert_onnx_to_trt(onnx_path, trt_path, trtexec_path):
    """步驟 2: 使用 trtexec 將 ONNX 轉換為 TensorRT"""
    
    # 設定環境變數
    env = os.environ.copy()
    env["NVIDIA_TF32_OVERRIDE"] = "0"
    env["LD_LIBRARY_PATH"] = "/circ330/TensorRT-8.4.3.1/lib:" + env.get("LD_LIBRARY_PATH", "")
    
    print(f"✅ 環境變數: NVIDIA_TF32_OVERRIDE=0")
    print(f"✅ LD_LIBRARY_PATH: {env['LD_LIBRARY_PATH']}")
    
    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(trt_path), exist_ok=True)
    
    print("\n🔨 使用 trtexec 建立 TensorRT engine...")
    print(f"   - 輸入: {onnx_path}")
    print(f"   - 輸出: {trt_path}")
    print(f"   - 精度: FP32 (禁用 TF32)")
    print("")
    
    # 構建 trtexec 命令
    cmd = [
        trtexec_path,
        f"--onnx={onnx_path}",
        f"--saveEngine={trt_path}",
        "--noTF32",
        "--verbose",
        "--dumpLayerInfo"
    ]
    
    log_file = "../trt_conversion_fp32.log"
    
    try:
        # 執行 trtexec，同時輸出到 console 和 log 檔案
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                bufsize=1
            )
            
            # 即時輸出並寫入 log
            for line in process.stdout:
                print(line, end='')
                f.write(line)
            
            process.wait()
            
            if process.returncode != 0:
                print(f"\n❌ trtexec 返回錯誤碼: {process.returncode}")
                return False
        
        # 檢查輸出檔案
        if os.path.exists(trt_path):
            file_size = os.path.getsize(trt_path) / (1024 * 1024)
            print(f"\n✅ TensorRT Engine 已建立")
            print(f"📁 檔案: {trt_path}")
            print(f"📏 大小: {file_size:.2f} MB")
            print(f"📝 完整日誌: {log_file}")
            return True
        else:
            print(f"\n❌ TensorRT Engine 檔案不存在: {trt_path}")
            print(f"📝 請檢查日誌: {log_file}")
            return False
            
    except FileNotFoundError:
        print(f"❌ 找不到 trtexec: {trtexec_path}")
        print("   請確認 TensorRT 路徑正確")
        return False
    except Exception as e:
        print(f"❌ 發生錯誤: {e}")
        return False

if __name__ == "__main__":
    sys.exit(main())
