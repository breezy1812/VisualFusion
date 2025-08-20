#!/usr/bin/env python3
"""
ONNX to TensorRT Conversion Script
將固定形狀的 ONNX 模型轉換為 TensorRT 引擎

基於:
- /circ330/forgithub/VisualFusion_libtorch/convert_to_libtorch/model_jit 模型
- /circ330/forgithub/VisualFusion_libtorch/convert_to_libtorch/export_to_onnx_fixed.py 生成的固定形狀 ONNX 模型

Usage:
    python export_onnx2tensorRT.py
"""

import tensorrt as trt
import numpy as np
import os
import argparse
from pathlib import Path

class ONNXToTensorRTConverter:
    def __init__(self):
        # 創建 TensorRT logger，使用 WARNING 等級避免過多輸出
        self.logger = trt.Logger(trt.Logger.WARNING)

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
    parser = argparse.ArgumentParser(description='Convert ONNX model to TensorRT engine')
    parser.add_argument('--onnx', type=str,
                       default='/circ330/forgithub/VisualFusion_libtorch/Onnx/model/onnxModel/SemLA_onnx_320x240_fixed1200pts.onnx',
                       help='Path to input ONNX model')
    parser.add_argument('--trt', type=str,
                       default='/circ330/forgithub/VisualFusion_libtorch/tensorRT/model/trtModel/trt_1200kps.engine',
                       help='Path to output TensorRT engine')
    parser.add_argument('--fp16', action='store_true', default=False,
                       help='Enable FP16 precision (default: True)')
    parser.add_argument('--workspace-size', type=int, default=1024,
                       help='Max workspace size in MB (default: 1024)')

    args = parser.parse_args()

    print("🎯 ONNX to TensorRT Conversion Tool")
    print("=" * 50)
    print("📋 Configuration:")
    print(f"  📁 ONNX model: {args.onnx}")
    print(f"  💾 TRT engine: {args.trt}")
    print(f"  🚀 FP16 mode: {args.fp16}")
    print(f"  💾 Workspace: {args.workspace_size} MB")
    print("=" * 50)

    # 創建轉換器並執行轉換
    converter = ONNXToTensorRTConverter()

    success = converter.convert_onnx_to_trt(
        onnx_path=args.onnx,
        trt_path=args.trt,
        fp16_mode=args.fp16,
        max_workspace_size=args.workspace_size * 1024 * 1024  # Convert MB to bytes
    )

    if success:
        print("\n🎉 Conversion completed successfully!")
        print(f"📌 You can now use the TensorRT engine: {args.trt}")
        print("🔧 Update your configuration files to use this new engine.")
    else:
        print("\n💥 Conversion failed!")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
