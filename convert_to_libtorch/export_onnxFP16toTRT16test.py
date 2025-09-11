#!/usr/bin/env python3
"""
PyTorch to TensorRT Conversion Script (支援 FP16)
從 PyTorch 模型直接轉換為 TensorRT 引擎，支援 FP16 模式

轉換流程：
- PyTorch 模型：始終保持 FP32 精度以確保數值穩定性
- ONNX 模型：始終保持 FP32 精度以確保相容性
- TensorRT 引擎：可選擇 FP16 精度以提升推理速度

基於:
- /circ330/forgithub/VisualFusion_libtorch/convert_to_libtorch/model_jit 模型
- SemLA PyTorch 模型直接轉換

Usage:
    python export_onnxFP16toTRT16test.py --fp16  # TensorRT 使用 FP16，PyTorch/ONNX 保持 FP32
    python export_onnxFP16toTRT16test.py         # 全程使用 FP32 模式
    python export_onnxFP16toTRT16test.py --fp16 --opset 12
"""

import tensorrt as trt
import numpy as np
import os
import argparse
import torch
import onnx
import tempfile
from pathlib import Path

# 導入 SemLA 模型
from model_jit.SemLA import SemLA

class PyTorchToTensorRTConverter:
    def __init__(self):
        # 創建 TensorRT logger，使用 WARNING 等級避免過多輸出
        self.logger = trt.Logger(trt.Logger.WARNING)

    def export_pytorch_to_onnx(self, use_fp16=False, opset_version=12, model_path="./reg.ckpt"):
        """
        從 PyTorch 模型導出 ONNX 模型
        
        Args:
            use_fp16: 是否在 TensorRT 中使用 FP16 精度（ONNX 模型保持 FP32）
            opset_version: ONNX opset 版本
            model_path: PyTorch 模型檢查點路徑
        
        Returns:
            str: 臨時 ONNX 檔案路徑
        """
        print("🎯 從 PyTorch 模型開始轉換...")
        
        # 使用 CUDA 來獲得最佳性能
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用設備: {device}")

        # 統一使用 FP32 進行 PyTorch 到 ONNX 的轉換，確保數值穩定性
        fpMode = torch.float32
        if use_fp16:
            print("正在載入模型 (FP32)，稍後在 TensorRT 中啟用 FP16...")
        else:
            print("正在載入模型 (FP32)...")
            
        matcher = SemLA(device=device, fp=fpMode)
        matcher.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        matcher = matcher.eval().to(device, dtype=fpMode)
        print("✅ 模型已載入 (FP32)")

        # 使用與配置文件相符的尺寸
        width = 320
        height = 240

        print(f"建立輸入張量，尺寸: {height}x{width}, 精度: FP32")
        torch_input_1 = torch.randn(1, 1, height, width).to(device, dtype=fpMode)
        torch_input_2 = torch.randn(1, 1, height, width).to(device, dtype=fpMode)

        # 創建臨時 ONNX 檔案
        temp_onnx = tempfile.NamedTemporaryFile(suffix='.onnx', delete=False)
        onnx_path = temp_onnx.name
        temp_onnx.close()

        tensorrt_precision = "FP16" if use_fp16 else "FP32"
        print(f"轉換為 FP32 ONNX 模型（TensorRT 將使用 {tensorrt_precision}）...")
        print(f"ONNX OpSet 版本: {opset_version}")

        try:
            torch.onnx.export(
                matcher,
                (torch_input_1, torch_input_2),
                onnx_path,
                verbose=False,
                opset_version=opset_version,
                input_names=["vi_img", "ir_img"],
                output_names=["mkpt0", "mkpt1", "leng1", "leng2"],
                do_constant_folding=True,
            )
            print(f"✅ FP32 ONNX 模型轉換完成")
            return onnx_path
            
        except Exception as e:
            print(f"❌ ONNX 轉換失敗: {e}")
            if os.path.exists(onnx_path):
                os.unlink(onnx_path)
            return None

    def pytorch_to_tensorrt(self, model_path="./reg.ckpt", trt_path=None, use_fp16=False, opset_version=12, max_workspace_size=1<<30):
        """
        從 PyTorch 模型直接轉換為 TensorRT 引擎
        
        Args:
            model_path: PyTorch 模型檢查點路徑
            trt_path: 輸出 TensorRT 引擎路徑
            use_fp16: 是否使用 FP16 精度
            opset_version: ONNX opset 版本
            max_workspace_size: 最大工作空間大小
        
        Returns:
            bool: 轉換是否成功
        """
        print("🚀 PyTorch to TensorRT 完整轉換流程")
        print("=" * 50)
        
        # 步驟 1: 從 PyTorch 轉換為 ONNX
        temp_onnx_path = self.export_pytorch_to_onnx(use_fp16, opset_version, model_path)
        if not temp_onnx_path:
            return False
            
        # 步驟 2: 從 ONNX 轉換為 TensorRT
        precision_str = "fp16" if use_fp16 else "fp32"
        if trt_path is None:
            trt_path = f"./trt_semla_{precision_str}_op{opset_version}.engine"
            
        print(f"\n🔄 轉換 ONNX 為 TensorRT 引擎...")
        success = self.convert_onnx_to_trt(
            onnx_path=temp_onnx_path,
            trt_path=trt_path,
            fp16_mode=use_fp16,
            max_workspace_size=max_workspace_size
        )
        
        # 清理臨時 ONNX 檔案
        try:
            os.unlink(temp_onnx_path)
            print(f"🗑️  清理臨時檔案: {temp_onnx_path}")
        except:
            pass
            
        return success

    def convert_onnx_to_trt(self, onnx_path, trt_path, max_batch_size=1, fp16_mode=True, max_workspace_size=1<<30):
        """
        將 ONNX 模型轉換為 TensorRT 引擎

        Args:
            onnx_path: 輸入 ONNX 模型路徑
            trt_path: 輸出 TensorRT 引擎路徑
            max_batch_size: 最大批次大小
            fp16_mode: 是否啟用 FP16 精度
            max_workspace_size: 最大工作空間大小 (bytes)
        """
        print(f"🔄 Converting ONNX to TensorRT...")
        print(f"📁 Input ONNX: {onnx_path}")
        print(f"💾 Output TRT: {trt_path}")

        # 檢查輸入文件
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

        # 創建 builder 和 network
        builder = trt.Builder(self.logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, self.logger)
        
        # 解析 ONNX 模型
        print("📖 Parsing ONNX model...")
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                print("❌ Failed to parse ONNX model")
                for error in range(parser.num_errors):
                    print(f"  Error {error}: {parser.get_error(error)}")
                return False

        print("✅ ONNX model parsed successfully")

        # 顯示網路信息
        print(f"📊 Network information:")
        print(f"  🔢 Number of inputs: {network.num_inputs}")
        print(f"  🔢 Number of outputs: {network.num_outputs}")

        for i in range(network.num_inputs):
            tensor = network.get_input(i)
            print(f"  📥 Input {i}: {tensor.name}")
            print(f"      Shape: {tensor.shape}")
            print(f"      Dtype: {tensor.dtype}")

        for i in range(network.num_outputs):
            tensor = network.get_output(i)
            print(f"  📤 Output {i}: {tensor.name}")
            print(f"      Shape: {tensor.shape}")
            print(f"      Dtype: {tensor.dtype}")

        # 創建建構配置
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)

        # 啟用 FP16 精度（如果支持且請求）
        if fp16_mode and builder.platform_has_fast_fp16:
            print("🚀 Enabling FP16 precision for faster inference")
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            print("⚡ Using FP32 precision for stability")

        # 設定優化配置文件（針對固定輸入形狀）
        profile = builder.create_optimization_profile()
        

        # 基於 SemLA 模型的固定形狀設定
        # 輸入: vi_img (1, 1, 240, 320), ir_img (1, 1, 240, 320)
        input_shapes = [
            (1, 1, 240, 320),  # vi_img
            (1, 1, 240, 320)   # ir_img
        ]

        for i in range(network.num_inputs):
            tensor = network.get_input(i)
            shape = input_shapes[i]
            print(f"⚙️  Setting optimization profile for {tensor.name}: {shape}")
            # 確保所有 min, opt, max 都是固定形狀，避免動態尺寸問題
            profile.set_shape(tensor.name, shape, shape, shape)

        # 移除 is_valid() 檢查，因為 TensorRT 10.x 版本沒有這個方法
        config.add_optimization_profile(profile)

        # 建構 TensorRT 引擎
        print("🔄 Building TensorRT engine... (this may take several minutes)")
        print("   💭 Please be patient, optimizing network for your hardware...")

        serialized_engine = builder.build_serialized_network(network, config)

        if serialized_engine is None:
            print("❌ Failed to build TensorRT engine")
            return False

        # 儲存引擎到文件
        os.makedirs(os.path.dirname(trt_path), exist_ok=True)
        with open(trt_path, 'wb') as f:
            f.write(serialized_engine)

        print(f"✅ TensorRT engine saved successfully!")
        print(f"💾 Engine file: {trt_path}")
        print(f"📏 File size: {os.path.getsize(trt_path) / (1024*1024):.2f} MB")

        # 驗證引擎
        return self.validate_engine(trt_path)

    def validate_engine(self, trt_path):
        """驗證創建的 TensorRT 引擎"""
        print("🔍 Validating TensorRT engine...")

        try:
            # 載入引擎
            runtime = trt.Runtime(self.logger)
            with open(trt_path, 'rb') as f:
                engine_data = f.read()

            engine = runtime.deserialize_cuda_engine(engine_data)
            if not engine:
                print("❌ Failed to load created engine")
                return False

            context = engine.create_execution_context()
            if not context:
                print("❌ Failed to create execution context")
                return False

            print("📋 Engine validation results:")
            print(f"  🔢 Number of bindings: {engine.num_bindings}")

            for i in range(engine.num_bindings):
                name = engine.get_binding_name(i)
                shape = engine.get_binding_shape(i)
                dtype = engine.get_binding_dtype(i)
                is_input = engine.binding_is_input(i)
                binding_type = "Input" if is_input else "Output"

                print(f"  {'📥' if is_input else '📤'} {binding_type} {i}: {name}")
                print(f"      Shape: {shape}")
                print(f"      Dtype: {dtype}")

            print("✅ Engine validation passed!")
            return True

        except Exception as e:
            print(f"❌ Engine validation failed: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch/ONNX model to TensorRT engine')
    parser.add_argument('--model', type=str,
                       default='./reg.ckpt',
                       help='Path to PyTorch model checkpoint (default: ./reg.ckpt)')
    parser.add_argument('--onnx', type=str, default=None,
                       help='Path to input ONNX model (if provided, skip PyTorch conversion)')
    parser.add_argument('--trt', type=str, default=None,
                       help='Path to output TensorRT engine (auto-generated if not provided)')
    parser.add_argument('--fp16', action='store_true', default=False,
                       help='Enable FP16 precision in TensorRT (PyTorch and ONNX remain FP32)')
    parser.add_argument('--opset', type=int, default=12,
                       help='ONNX opset version (default: 12)')
    parser.add_argument('--workspace-size', type=int, default=1024,
                       help='Max workspace size in MB (default: 1024)')

    args = parser.parse_args()

    print("🎯 PyTorch/ONNX to TensorRT Conversion Tool")
    print("=" * 60)
    print("📋 Configuration:")
    
    # 創建轉換器
    converter = PyTorchToTensorRTConverter()
    
    if args.onnx:
        # 從 ONNX 轉換模式
        print(f"  📁 ONNX model: {args.onnx}")
        print(f"  💾 TRT engine: {args.trt}")
        print(f"  🚀 FP16 mode: {args.fp16}")
        print(f"  💾 Workspace: {args.workspace_size} MB")
        print("=" * 60)
        
        success = converter.convert_onnx_to_trt(
            onnx_path=args.onnx,
            trt_path=args.trt,
            fp16_mode=args.fp16,
            max_workspace_size=args.workspace_size * 1024 * 1024
        )
    else:
        # 從 PyTorch 完整轉換模式
        precision_str = "fp16" if args.fp16 else "fp32"
        if args.trt is None:
            args.trt = f"./trt_semla_{precision_str}_op{args.opset}.engine"
            
        print(f"  🧠 PyTorch model: {args.model}")
        print(f"  💾 TRT engine: {args.trt}")
        print(f"  🚀 FP16 mode: {args.fp16}")
        print(f"  🔧 ONNX OpSet: {args.opset}")
        print(f"  💾 Workspace: {args.workspace_size} MB")
        print("=" * 60)
        
        success = converter.pytorch_to_tensorrt(
            model_path=args.model,
            trt_path=args.trt,
            use_fp16=args.fp16,
            opset_version=args.opset,
            max_workspace_size=args.workspace_size * 1024 * 1024
        )

    if success:
        print("\n🎉 Conversion completed successfully!")
        print(f"📌 TensorRT engine: {args.trt}")
        print("🔧 Update your configuration files to use this new engine.")
    else:
        print("\n💥 Conversion failed!")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
