import os
import sys
import torch
import tensorrt as trt
import onnx
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit  # 自動初始化 CUDA

# 檔案路徑
input_path = "/circ330/forgithub/VisualFusion_libtorch/Onnx/model/onnxModel/SemLA_onnx_320x240_opencv_compatible.onnx"
output_path = "/circ330/forgithub/VisualFusion_libtorch/convert_to_libtorch/converModel/SemLA_tensorrt.engine"

# 確保輸出目錄存在
os.makedirs(os.path.dirname(output_path), exist_ok=True)

def setup_cuda_environment():
    """設置 CUDA 環境並選擇 GPU"""
    print("=== 設置 CUDA 環境 ===")
    
    # 檢查 CUDA 是否可用
    if not torch.cuda.is_available():
        print("錯誤: CUDA 不可用")
        return False
    
    # 獲取可用的 GPU
    num_gpus = torch.cuda.device_count()
    print(f"可用的 GPU 數量: {num_gpus}")
    
    # 顯示所有 GPU 信息
    for i in range(num_gpus):
        gpu_props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {gpu_props.name}")
        print(f"  記憶體: {gpu_props.total_memory / 1024**3:.1f} GB")
        print(f"  計算能力: {gpu_props.major}.{gpu_props.minor}")
    
    # 設置 GPU 設備
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    gpu_id = int(cuda_visible_devices.split(',')[0])
    
    if gpu_id >= num_gpus:
        gpu_id = 0
        print(f"警告: 指定的 GPU {gpu_id} 不存在，使用 GPU 0")
    
    torch.cuda.set_device(gpu_id)
    print(f"使用 GPU {gpu_id}: {torch.cuda.get_device_properties(gpu_id).name}")
    
    # 清理 GPU 記憶體
    torch.cuda.empty_cache()
    
    return True

# TensorRT Logger 設置為較低的日志級別以減少警告
TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

def check_onnx_model(onnx_path):
    """檢查 ONNX 模型的基本信息"""
    print("=== 檢查 ONNX 模型 ===")
    model = onnx.load(onnx_path)
    print(f"ONNX 模型版本: {model.ir_version}")
    print(f"生產者名稱: {model.producer_name}")
    print(f"生產者版本: {model.producer_version}")
    
    # 檢查輸入
    print("\n輸入張量:")
    for input_tensor in model.graph.input:
        print(f"  名稱: {input_tensor.name}")
        shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
        print(f"  形狀: {shape}")
        print(f"  類型: {input_tensor.type.tensor_type.elem_type}")
    
    # 檢查輸出
    print("\n輸出張量:")
    for output_tensor in model.graph.output:
        print(f"  名稱: {output_tensor.name}")
        shape = [dim.dim_value if dim.dim_value > 0 else -1 for dim in output_tensor.type.tensor_type.shape.dim]
        print(f"  形狀: {shape}")
        print(f"  類型: {output_tensor.type.tensor_type.elem_type}")
    
    return True

def onnx2trt(onnx_path, engine_path):
    """將 ONNX 模型轉換為 TensorRT Engine"""
    print(f"\n=== 開始轉換 ONNX 到 TensorRT ===")
    print(f"輸入檔案: {onnx_path}")
    print(f"輸出檔案: {engine_path}")
    
    # 檢查輸入檔案是否存在
    if not os.path.exists(onnx_path):
        print(f"錯誤: ONNX 檔案不存在: {onnx_path}")
        return False
    
    try:
        # 創建 builder
        builder = trt.Builder(TRT_LOGGER)
        
        # 設置更保守的配置以避免記憶體問題
        if hasattr(builder, 'max_batch_size'):
            builder.max_batch_size = 1
        
        # 創建網路，使用顯式批次大小
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        
        # 創建 ONNX 解析器
        parser = trt.OnnxParser(network, TRT_LOGGER)
        
        # 解析 ONNX 檔案
        print("解析 ONNX 檔案...")
        with open(onnx_path, 'rb') as model:
            model_data = model.read()
            success = parser.parse(model_data)
            
        if not success:
            print("錯誤: ONNX 解析失敗")
            for i in range(parser.num_errors):
                error = parser.get_error(i)
                print(f"  錯誤 {i}: {error}")
            return False
        
        print("ONNX 解析成功!")
        
        # 創建建構器配置
        config = builder.create_builder_config()
        
        # 設定較小的工作空間大小以避免記憶體問題 (512MB)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 512 * 1024 * 1024)
        
        # 暫時不啟用 FP16，先確保基本轉換成功
        print("使用 FP32 精度進行轉換")
        
        # 創建優化設定檔
        profile = builder.create_optimization_profile()
        
        # 設定輸入張量的維度（固定批次大小）
        input_shape = (1, 1, 240, 320)
        min_shape = input_shape
        opt_shape = input_shape  
        max_shape = input_shape
        
        profile.set_shape("vi_img", min_shape, opt_shape, max_shape)
        profile.set_shape("ir_img", min_shape, opt_shape, max_shape)
        
        config.add_optimization_profile(profile)
        
        print("建構 TensorRT 引擎...")
        print("這可能需要幾分鐘時間...")
        
        # 建構引擎 - 使用較新的 API
        plan = builder.build_serialized_network(network, config)
        
        if plan is None:
            print("錯誤: TensorRT 引擎建構失敗")
            return False
        
        # 儲存引擎
        print(f"儲存 TensorRT 引擎到: {engine_path}")
        with open(engine_path, 'wb') as f:
            f.write(plan)
        
        print("✅ TensorRT 轉換成功!")
        
        # 驗證生成的引擎
        return verify_engine(engine_path)
        
    except Exception as e:
        print(f"錯誤: TensorRT 轉換失敗: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def verify_engine(engine_path):
    """驗證生成的 TensorRT 引擎"""
    print(f"\n=== 驗證 TensorRT 引擎 ===")
    
    try:
        # 載入引擎
        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = runtime.deserialize_cuda_engine(engine_data)
        
        if engine is None:
            print("錯誤: 無法載入 TensorRT 引擎")
            return False
        
        print("✅ TensorRT 引擎載入成功!")
        
        # 顯示引擎信息
        if hasattr(engine, 'name'):
            print(f"引擎名稱: {engine.name}")
        if hasattr(engine, 'max_batch_size'):
            print(f"最大批次大小: {engine.max_batch_size}")
        if hasattr(engine, 'device_memory_size'):
            print(f"工作空間大小: {engine.device_memory_size} bytes")
        print(f"綁定數量: {engine.num_bindings}")
        
        # 顯示輸入/輸出信息
        for i in range(engine.num_bindings):
            binding_name = engine.get_binding_name(i)
            binding_shape = engine.get_binding_shape(i)
            binding_dtype = engine.get_binding_dtype(i)
            
            if engine.binding_is_input(i):
                print(f"輸入 {i}: {binding_name}, 形狀: {binding_shape}, 類型: {binding_dtype}")
            else:
                print(f"輸出 {i}: {binding_name}, 形狀: {binding_shape}, 類型: {binding_dtype}")
        
        return True
        
    except Exception as e:
        print(f"錯誤: 引擎驗證失敗: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== SemLA ONNX 到 TensorRT 轉換工具 ===")
    
    # 設置 CUDA 環境
    if not setup_cuda_environment():
        print("❌ CUDA 環境設置失敗!")
        sys.exit(1)
    
    # 檢查 ONNX 模型
    if check_onnx_model(input_path):
        # 轉換到 TensorRT
        success = onnx2trt(input_path, output_path)
        
        if success:
            print(f"\n🎉 轉換完成! TensorRT 引擎已儲存到: {output_path}")
        else:
            print("\n❌ 轉換失敗!")
    else:
        print("❌ ONNX 模型檢查失敗!")
