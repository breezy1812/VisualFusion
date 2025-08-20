#!/usr/bin/env python3
"""
TensorRT Model Test Script
測試 TensorRT 引擎是否能正常執行推論並產生正確的特徵點輸出

Usage:
    python test_trt_inference.py --engine model/trtModel/tetmodel.trt --input1 test_eo.jpg --input2 test_ir.jpg
"""

import sys
import argparse
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from pathlib import Path

class TensorRTInference:
    def __init__(self, engine_path):
        """初始化 TensorRT 引擎"""
        print(f"[TRT] Loading engine: {engine_path}")
        
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        if not self.engine:
            raise RuntimeError("Failed to load TensorRT engine")
        
        self.context = self.engine.create_execution_context()
        if not self.context:
            raise RuntimeError("Failed to create execution context")
        
        self.input_specs = []
        self.output_specs = []
        
        # <<< 修正點 1: 使用新的 TensorRT API >>>
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name) # 使用名稱獲取 shape
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            
            is_dynamic = -1 in shape
            
            binding_info = {
                'name': name,
                'shape': tuple(shape), # 轉為 tuple
                'dtype': dtype,
                'is_dynamic': is_dynamic,
                'index': self.engine.get_binding_index(name) # 獲取 binding 索引
            }
            
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_specs.append(binding_info)
                print(f"[TRT] Input {i}: {name}, shape={tuple(shape)}, dtype={dtype}, dynamic={is_dynamic}")
            else:
                self.output_specs.append(binding_info)
                print(f"[TRT] Output {i}: {name}, shape={tuple(shape)}, dtype={dtype}, dynamic={is_dynamic}")
        
        self.has_dynamic_shapes = any(spec['is_dynamic'] for spec in self.input_specs)
        print(f"[TRT] Engine has dynamic input shapes: {self.has_dynamic_shapes}")
        
        print(f"[TRT] Engine loaded successfully, {len(self.input_specs)} inputs, {len(self.output_specs)} outputs")

        # <<< 修正點 2: 創建 CUDA stream 以進行非同步操作 >>>
        self.stream = cuda.Stream()
    
    def preprocess_image(self, image_path, target_size=(320, 240)):
        """預處理圖像"""
        print(f"[TRT] Preprocessing image: {image_path}")
        
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        img = cv2.resize(img, (target_size[0], target_size[1])) # (width, height) for cv2.resize
        
        img = img.astype(np.float32) / 255.0
        
        img = np.expand_dims(np.expand_dims(img, axis=0), axis=0) # (H,W) -> (1,1,H,W)
        
        print(f"[TRT] Preprocessed shape: {img.shape}, dtype: {img.dtype}, range: [{img.min():.3f}, {img.max():.3f}]")
        return img
    
    def inference(self, eo_image_path, ir_image_path):
        """執行推論"""
        print("[TRT] Starting inference...")
        
        eo_img = self.preprocess_image(eo_image_path)
        ir_img = self.preprocess_image(ir_image_path)
        
        host_inputs = [eo_img, ir_img]
        
        # 即使是靜態引擎，也要設定 input shape
        for i, spec in enumerate(self.input_specs):
            self.context.set_input_shape(spec['name'], host_inputs[i].shape)

        # --- 以下是主要修正區域 ---

        # 分配記憶體並「設定 Tensor 位址」，而不是建立 bindings 列表
        device_inputs = []
        host_outputs = []
        device_outputs = []

        # 處理輸入
        for i, spec in enumerate(self.input_specs):
            h_input = np.ascontiguousarray(host_inputs[i].astype(spec['dtype']))
            d_input = cuda.mem_alloc(h_input.nbytes)
            device_inputs.append(d_input)
            # <<< 修正點 1: 設定輸入 Tensor 的 GPU 位址 >>>
            self.context.set_tensor_address(spec['name'], int(d_input))
            cuda.memcpy_htod_async(d_input, h_input, self.stream)
            
        # 處理輸出
        for spec in self.output_specs:
            output_shape = self.context.get_tensor_shape(spec['name'])
            h_output = np.empty(tuple(output_shape), dtype=spec['dtype'])
            d_output = cuda.mem_alloc(h_output.nbytes)
            host_outputs.append(h_output)
            device_outputs.append(d_output)
            # <<< 修正點 2: 設定輸出 Tensor 的 GPU 位址 >>>
            self.context.set_tensor_address(spec['name'], int(d_output))
        
        # 執行非同步推論
        print("[TRT] Executing inference...")
        # <<< 修正點 3: 呼叫 execute_async_v3 時不再傳入 bindings >>>
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # 將結果從 GPU 非同步複製回 CPU
        for i in range(len(self.output_specs)):
            cuda.memcpy_dtoh_async(host_outputs[i], device_outputs[i], self.stream)

        # 等待所有 CUDA 操作完成
        self.stream.synchronize()
        
        output_data = host_outputs
        
        for i, (spec, h_output) in enumerate(zip(self.output_specs, output_data)):
            print(f"[TRT] Output '{spec['name']}' shape: {h_output.shape}, dtype: {h_output.dtype}")
            if h_output.size > 0:
                print(f"[TRT] Output '{spec['name']}' range: [{h_output.min():.3f}, {h_output.max():.3f}]")

        # 手動釋放 device memory (可選，但在循環推論中很重要)
        for d_mem in device_inputs + device_outputs:
            d_mem.free()

        return output_data
    
    def extract_keypoints(self, output_data):
        """從輸出中提取特徵點 - 處理固定形狀輸出格式"""
        if len(output_data) != 4:
            print(f"[TRT] Warning: Expected 4 outputs, got {len(output_data)}")
            return [], []
        
        # 假設輸出順序與引擎定義一致
        leng1, mkpt0, leng2, mkpt1 = output_data
        
        print(f"[TRT] leng1: {leng1}, shape: {leng1.shape}")
        print(f"[TRT] mkpt0 shape: {mkpt0.shape}")
        print(f"[TRT] leng2: {leng2}, shape: {leng2.shape}")
        print(f"[TRT] mkpt1 shape: {mkpt1.shape}")
        
        # <<< 修正點 5: 簡化並修正純量值的提取 >>>
        # 對於 shape=() 的 0-D NumPy 陣列, 直接用 int() 轉換即可
        leng1 = int(leng1)
        leng2 = int(leng2)

        # 從 mkpt0 中提取前 num_eo_points 個有效特徵點
        mkpt0 = mkpt0.astype(np.float32)[:leng1]
        mkpt1 = mkpt1.astype(np.float32)[:leng2]

        _, mask = cv2.findHomography(mkpt0, mkpt1, cv2.RANSAC, 5.0)

        mkpt0 = mkpt0[mask.ravel() == 1]
        mkpt1 = mkpt1[mask.ravel() == 1]

        eo_points = [(int(x), int(y)) for x, y in mkpt0 if x != 0 or y != 0]

        # 從 mkpt1 中提取前 num_ir_points 個有效特徵點
        ir_points = [(int(x), int(y)) for x, y in mkpt1 if x != 0 or y != 0]
        
        print(f"[TRT] Extracted {len(eo_points)} EO keypoints, {len(ir_points)} IR keypoints")
        return eo_points, ir_points
    
    def visualize_keypoints(self, eo_image_path, ir_image_path, eo_points, ir_points, output_path="trt_test_result.jpg"):
        """可視化特徵點配對結果"""
        print(f"[TRT] Creating visualization: {output_path}")
        
        eo_img = cv2.imread(str(eo_image_path))
        ir_img = cv2.imread(str(ir_image_path))
        
        target_size = (320, 240)
        eo_img = cv2.resize(eo_img, target_size)
        ir_img = cv2.resize(ir_img, target_size)
        
        combined = np.hstack([ir_img, eo_img])
        
        min_pairs = min(len(eo_points), len(ir_points))
        for i in range(min_pairs):
            ir_pt = ir_points[i]
            eo_pt = (eo_points[i][0] + target_size[0], eo_points[i][1])
            
            cv2.circle(combined, ir_pt, 3, (0, 255, 0), -1)
            cv2.circle(combined, eo_pt, 3, (0, 0, 255), -1)
            cv2.line(combined, ir_pt, eo_pt, (255, 0, 0), 1)
        
        cv2.putText(combined, f"IR ({len(ir_points)} pts)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(combined, f"EO ({len(eo_points)} pts)", (target_size[0] + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(combined, f"Pairs: {min_pairs}", (target_size[0] // 2 - 50, combined.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imwrite(output_path, combined)
        print(f"[TRT] Saved visualization to: {output_path}")
        
        return min_pairs > 0

def main():
    img1_path = "/circ330/videodata/Version3/2024-07-10_15-38-54_EO.jpg"
    img2_path = "/circ330/videodata/Version3/2024-07-10_15-38-54_IR.jpg"
    parser = argparse.ArgumentParser(description='Test TensorRT model inference')
    parser.add_argument('--engine',default='/circ330/forgithub/VisualFusion_libtorch/tensorRT/model/trtModel/trt_1200kps.trt' , help='Path to TensorRT engine file (.trt)')
    parser.add_argument('--input1',default=img1_path, help='Path to first input image (EO)')
    parser.add_argument('--input2', default=img2_path, help='Path to second input image (IR)')
    parser.add_argument('--output', default='/circ330/forgithub/VisualFusion_libtorch/tensorRT/output/trt_test_result.jpg', help='Output visualization path')
    
    args = parser.parse_args()
    
    if not Path(args.engine).exists():
        print(f"Error: Engine file not found: {args.engine}")
        return False
    if not Path(args.input1).exists():
        print(f"Error: Input image 1 not found: {args.input1}")
        return False
    if not Path(args.input2).exists():
        print(f"Error: Input image 2 not found: {args.input2}")
        return False
    
    try:
        trt_inference = TensorRTInference(args.engine)
        outputs = trt_inference.inference(args.input1, args.input2)
        eo_points, ir_points = trt_inference.extract_keypoints(outputs)
        success = trt_inference.visualize_keypoints(args.input1, args.input2, eo_points, ir_points, args.output)
        
        if success and len(eo_points) > 0 and len(ir_points) > 0:
            print(f"✅ [SUCCESS] TensorRT inference successful!")
            print(f"   - EO keypoints: {len(eo_points)}")
            print(f"   - IR keypoints: {len(ir_points)}")
            print(f"   - Matched pairs: {min(len(eo_points), len(ir_points))}")
            print(f"   - Visualization saved: {args.output}")
            return True
        else:
            print("❌ [FAILED] No valid keypoints detected")
            return False
            
    except Exception as e:
        print(f"❌ [ERROR] TensorRT inference failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)