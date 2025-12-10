"""
pytorch_fusion_v2.py

改進版圖像融合 - 解決邊緣縮小和邊框問題
使用 Sobel 邊緣檢測 + 拉普拉斯增強，不使用 roll 操作

termianl: 
# 導出模型 (edge_border=4, 高度=240, 寬度=320)
python3 pytorch_fusion_v2.py export 4 240 320
"""

import torch
import torch.nn.functional as F


class ImageFusionV2:
    """
    改進版 - 與原版相同的陰影邏輯，但使用 padding 替代 roll 避免循環邊界問題
    """
    
    def __init__(self, edge_border: int = 3, image_width: int = 320, image_height: int = 240, device: str = "cuda"):
        """
        Args:
            edge_border: 邊緣寬度/偏移量 (唯一可調整參數)
            image_width: 圖像寬度
            image_height: 圖像高度
            device: 運算設備
        """
        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        self.edge_border = edge_border
        self.edge_strength = 0.5  # 固定邊緣強度
        self.image_width = image_width
        self.image_height = image_height
        
        # Sobel 算子 
        self.sobel_x = torch.tensor([[[[0, 0, 0],
                                        [1, 0, -1],
                                        [0, 0, 0]]]], dtype=torch.float32, device=self.device)
        self.sobel_y = torch.tensor([[[[0, 1, 0],
                                        [0, 0, 0],
                                        [0, -1, 0]]]], dtype=torch.float32, device=self.device)
        
        # 高斯核 
        self.gaussian = torch.tensor([[[[0.0751, 0.1238, 0.0751],
                                        [0.1238, 0.2042, 0.1238],
                                        [0.0751, 0.1238, 0.0751]]]], dtype=torch.float32, device=self.device)

    

    def _shift_no_wrap(self, x: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
        """
        平移但不循環，邊界填充 0
        只支持向右(dim=3)和向下(dim=2)偏移
        shift > 0: 向正方向移動 ,shift = border
        dim: 2 = Y方向(向下), 3 = X方向(向右)
        """
        if dim == 3:  # X 方向 - 向右
            result = F.pad(x[:, :, :, :-shift], (shift, 0, 0, 0), mode='constant', value=0)
        else:  # dim == 2, Y 方向 - 向下
            result = F.pad(x[:, :, :-shift, :], (0, 0, shift, 0), mode='constant', value=0)
        
        return result


    def edge(self, source: torch.Tensor) -> torch.Tensor:
        """
        邊緣檢測 + 陰影效果
        """
        if source.dim() != 4 or source.size(0) != 1 or source.size(1) != 1:
            raise RuntimeError("edge: Source must be a 4D tensor with shape [1, 1, H, W]")
        
        # 使用 reflect padding 進行卷積
        padded = F.pad(source, (1, 1, 1, 1), mode='reflect') #防止失真、不會讓圖像尺寸縮小
        blur = F.conv2d(padded, self.gaussian, padding=0)
        
        padded_blur = F.pad(blur, (1, 1, 1, 1), mode='reflect')
        sobel_x = F.conv2d(padded_blur, self.sobel_x, padding=0)
        sobel_y = F.conv2d(padded_blur, self.sobel_y, padding=0)
        
        border = self.edge_border
        
        # X 方向: 向右偏移
        edge_x = self._shift_no_wrap(sobel_x, border, dim=3)
        
        # Y 方向: 向下偏移
        edge_y = self._shift_no_wrap(sobel_y, border, dim=2)
        
        # 計算偏移後的邊緣幅值
        edge_shifted = torch.sqrt(edge_x ** 2 + edge_y ** 2)
        
        # 原始邊緣幅值
        sobel = torch.sqrt(sobel_x ** 2 + sobel_y ** 2)
        
        # 陰影效果: 原始邊緣 - 偏移邊緣
        edge = sobel - edge_shifted
        
        return edge

    def fusion(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        融合邊緣和目標圖像
        """
        if source.dim() != 4 or source.size(0) != 1 or source.size(1) != 1:
            raise RuntimeError("fusion: Source must be a 4D tensor with shape [1, 1, H, W]")
        if target.dim() != 4 or target.size(0) != 1 or target.size(1) != 3:
            raise RuntimeError("fusion: Target must be a 4D tensor with shape [1, 3, H, W]")
        
        # 擴展到 3 通道
        edge_3ch = source.repeat(1, 3, 1, 1)
        
        # 加法融合
        fused = target + edge_3ch
        
        return torch.clamp(fused, 0, 1)

    def forward(self, eo_gray: torch.Tensor, ir_color: torch.Tensor) -> torch.Tensor:
        """
        完整流程
        """
        e = self.edge(eo_gray)
        return self.fusion(e, ir_color)


if __name__ == "__main__":
    import time
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("=" * 50)
    
    # 測試簡化版
    print("Testing ImageFusionV2...")
    fusion_v2 = ImageFusionV2(edge_border=8, device=device)
    
    eo_gray = torch.rand(1, 1, 240, 320, device=fusion_v2.device)
    ir_color = torch.rand(1, 3, 240, 320, device=fusion_v2.device)
    
    with torch.no_grad():
        # 預熱
        _ = fusion_v2.forward(eo_gray, ir_color)
        
        # 計時
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        
        for _ in range(100):
            result = fusion_v2.forward(eo_gray, ir_color)
            
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - start
        
    print(f"  Input eo_gray: {eo_gray.shape}")
    print(f"  Input ir_color: {ir_color.shape}")
    print(f"  Output: {result.shape}")
    print(f"  Avg time per frame: {elapsed/100*1000:.2f} ms")
    print()
    
    # 測試簡化版
    print("Testing ImageFusionV2...")
    fusion_simple = ImageFusionV2(edge_border=4, device=device)
    
    with torch.no_grad():
        # 預熱
        _ = fusion_simple.forward(eo_gray, ir_color)
        
        # 計時
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        
        for _ in range(100):
            result_simple = fusion_simple.forward(eo_gray, ir_color)
            
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - start
        
    print(f"  Input eo_gray: {eo_gray.shape}")
    print(f"  Input ir_color: {ir_color.shape}")
    print(f"  Output: {result_simple.shape}")
    print(f"  Avg time per frame: {elapsed/100*1000:.2f} ms")
    print()
    
    print()
    print("=" * 50)
    print("Done!")


# ============================================================================
# TensorRT 轉換相關 - nn.Module 版本
# ============================================================================

class ImageFusionModule(torch.nn.Module):
    """
    可導出為 ONNX/TensorRT 的 nn.Module 版本
    """
    
    def __init__(self, edge_border: int = 3):
        super().__init__()
        self.edge_border = edge_border
        
        # 註冊為 buffer (常量，不參與訓練)
        self.register_buffer('sobel_x', torch.tensor([[[[0, 0, 0],
                                                         [1, 0, -1],
                                                         [0, 0, 0]]]], dtype=torch.float32))
        self.register_buffer('sobel_y', torch.tensor([[[[0, 1, 0],
                                                         [0, 0, 0],
                                                         [0, -1, 0]]]], dtype=torch.float32))
        self.register_buffer('gaussian', torch.tensor([[[[0.0751, 0.1238, 0.0751],
                                                          [0.1238, 0.2042, 0.1238],
                                                          [0.0751, 0.1238, 0.0751]]]], dtype=torch.float32))

    def _shift_no_wrap(self, x: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
        """
        平移但不循環，邊界填充 0
        只支持向右(dim=3)和向下(dim=2)偏移
        """
        if dim == 3:  # X 方向 - 向右
            result = F.pad(x[:, :, :, :-shift], (shift, 0, 0, 0), mode='constant', value=0)
        else:  # dim == 2, Y 方向 - 向下
            result = F.pad(x[:, :, :-shift, :], (0, 0, shift, 0), mode='constant', value=0)
        return result

    def edge(self, source: torch.Tensor) -> torch.Tensor:
        """邊緣檢測 + 陰影效果"""
        # 使用 reflect padding 進行卷積
        padded = F.pad(source, (1, 1, 1, 1), mode='reflect')
        blur = F.conv2d(padded, self.gaussian, padding=0)
        
        padded_blur = F.pad(blur, (1, 1, 1, 1), mode='reflect')
        sobel_x = F.conv2d(padded_blur, self.sobel_x, padding=0)
        sobel_y = F.conv2d(padded_blur, self.sobel_y, padding=0)
        
        border = self.edge_border
        
        # X 方向: 向右偏移
        edge_x = self._shift_no_wrap(sobel_x, border, dim=3)
        
        # Y 方向: 向下偏移
        edge_y = self._shift_no_wrap(sobel_y, border, dim=2)
        
        # 計算邊緣
        edge_shifted = torch.sqrt(edge_x ** 2 + edge_y ** 2)
        sobel = torch.sqrt(sobel_x ** 2 + sobel_y ** 2)
        edge = sobel - edge_shifted
        
        return edge

    def fusion(self, edge: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """融合邊緣和目標圖像"""
        edge_3ch = edge.repeat(1, 3, 1, 1)
        fused = target + edge_3ch
        return torch.clamp(fused, 0, 1)

    def forward(self, eo_gray: torch.Tensor, ir_color: torch.Tensor) -> torch.Tensor:
        """
        完整流程
        Args:
            eo_gray: [1, 1, H, W] EO 灰度圖
            ir_color: [1, 3, H, W] IR 彩色圖
        Returns:
            fused: [1, 3, H, W] 融合結果
        """
        e = self.edge(eo_gray)
        return self.fusion(e, ir_color)


def export_to_onnx(model: ImageFusionModule, onnx_path: str, 
                   height: int = 240, width: int = 320, opset_version: int = 17):

    model.eval()
    # 獲取設備 (從 buffer 獲取)
    device = model.sobel_x.device
    
    # 創建示例輸入
    dummy_eo_gray = torch.randn(1, 1, height, width, device=device)
    dummy_ir_color = torch.randn(1, 3, height, width, device=device)
    
    print(f"Exporting to ONNX: {onnx_path}")
    print(f"  Input shape: eo_gray={dummy_eo_gray.shape}, ir_color={dummy_ir_color.shape}")
    
    torch.onnx.export(
        model,
        (dummy_eo_gray, dummy_ir_color),
        onnx_path,
        input_names=['eo_gray', 'ir_color'],
        output_names=['fused'],
        dynamic_axes={
            'eo_gray': {0: 'batch', 2: 'height', 3: 'width'},
            'ir_color': {0: 'batch', 2: 'height', 3: 'width'},
            'fused': {0: 'batch', 2: 'height', 3: 'width'}
        },
        opset_version=opset_version,
        do_constant_folding=True
    )
    
    print(f"ONNX model saved to: {onnx_path}")
    
    # 驗證 ONNX 模型
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verified successfully!")
    except ImportError:
        print("Warning: onnx package not installed, skipping verification")
    except Exception as e:
        print(f"Warning: ONNX verification failed: {e}")


def convert_onnx_to_tensorrt(onnx_path: str, trt_path: str, 
                              height: int = 240, width: int = 320,
                              workspace_mb: int = 256):

    try:
        import tensorrt as trt
    except ImportError:
        print("Error: tensorrt package not installed!")
        print("Please install: pip install tensorrt")
        return False
    
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    
    print(f"Converting ONNX to TensorRT: {onnx_path} -> {trt_path}")
    
    print(f"  Workspace: {workspace_mb} MB")
    
    # 創建 builder
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # 解析 ONNX
    print("Parsing ONNX model...")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(f"ONNX Parse Error: {parser.get_error(error)}")
            return False
    print("ONNX parsing successful!")
    
    # 配置 builder 
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_mb * 1024 * 1024))
    
    # 設置優化配置 (固定尺寸)
    profile = builder.create_optimization_profile()
    profile.set_shape('eo_gray', (1, 1, height, width), (1, 1, height, width), (1, 1, height, width))
    profile.set_shape('ir_color', (1, 3, height, width), (1, 3, height, width), (1, 3, height, width))
    config.add_optimization_profile(profile)
    
    # 構建引擎
    print("Building TensorRT engine... (this may take a while)")
    try:
        serialized_engine = builder.build_serialized_network(network, config)
    except Exception as e:
        print(f"Error during engine building: {e}")
        return False
    
    if serialized_engine is None:
        print("Error: Failed to build TensorRT engine!")
        return False
    
    # 保存引擎
    with open(trt_path, 'wb') as f:
        f.write(serialized_engine)
    
    print(f"TensorRT engine saved to: {trt_path}")
    return True


def export_and_convert(edge_border: int = 4, height: int = 240, width: int = 320,
                       output_dir: str = "/circ330/forgithub/VisualFusion_libtorch/tensorRT_nx/model/NX"):
    """
    一鍵導出 ONNX 並轉換為 TensorRT
    
    Args:
        edge_border: 邊緣寬度參數
        height: 圖像高度
        width: 圖像寬度
        output_dir: 輸出目錄
    """
    import os
    
    # 創建模型
    model = ImageFusionModule(edge_border=edge_border).cuda().eval()
    
    # 文件路徑
    onnx_path = os.path.join(output_dir, f"border_{edge_border}_fusion.onnx")
    trt_path = os.path.join(output_dir, f"border_{edge_border}_fusion.trt")
    
    # 導出 ONNX
    export_to_onnx(model, onnx_path, height=height, width=width)
    
    # 轉換為 TensorRT (256MB workspace)
    success = convert_onnx_to_tensorrt(onnx_path, trt_path, height=height, width=width, 
                                        workspace_mb=256)
    
    if success:
        print()
        print("=" * 50)
        print("Export completed successfully!")
        print(f"  ONNX: {onnx_path}")
        print(f"  TRT:  {trt_path}")
    
    return success


# TensorRT 推理類
class ImageFusionTRT:
    """
    使用 TensorRT 引擎進行推理
    """
    
    def __init__(self, trt_path: str):
        """
        Args:
            trt_path: TensorRT 引擎文件路徑
        """
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
        except ImportError as e:
            raise ImportError(f"Required packages not installed: {e}")
        
        self.trt = trt
        self.cuda = cuda
        
        # 載入引擎
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(trt_path, 'rb') as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # 分配記憶體
        self._allocate_buffers()
        
        print(f"TensorRT engine loaded: {trt_path}")
    
    def _allocate_buffers(self):
        """分配 GPU 記憶體"""
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = self.cuda.Stream()
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = self.trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.context.get_tensor_shape(name)
            size = self.trt.volume(shape)
            
            # 分配 host 和 device 記憶體
            host_mem = self.cuda.pagelocked_empty(size, dtype)
            device_mem = self.cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.get_tensor_mode(name) == self.trt.TensorIOMode.INPUT:
                self.inputs.append({'name': name, 'host': host_mem, 'device': device_mem, 'shape': shape})
            else:
                self.outputs.append({'name': name, 'host': host_mem, 'device': device_mem, 'shape': shape})
    
    def infer(self, eo_gray: torch.Tensor, ir_color: torch.Tensor) -> torch.Tensor:
        """
        執行推理
        
        Args:
            eo_gray: [1, 1, H, W] EO 灰度圖 (GPU tensor)
            ir_color: [1, 3, H, W] IR 彩色圖 (GPU tensor)
            
        Returns:
            fused: [1, 3, H, W] 融合結果 (GPU tensor)
        """
        import numpy as np
        
        # 複製輸入到 host
        eo_np = eo_gray.cpu().numpy().ravel()
        ir_np = ir_color.cpu().numpy().ravel()
        
        np.copyto(self.inputs[0]['host'], eo_np)
        np.copyto(self.inputs[1]['host'], ir_np)
        
        # 複製到 device
        for inp in self.inputs:
            self.cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        
        # 設置 tensor 地址
        for inp in self.inputs:
            self.context.set_tensor_address(inp['name'], int(inp['device']))
        for out in self.outputs:
            self.context.set_tensor_address(out['name'], int(out['device']))
        
        # 執行推理
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        # 複製輸出到 host
        for out in self.outputs:
            self.cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        
        self.stream.synchronize()
        
        # 轉換為 tensor
        output_shape = self.outputs[0]['shape']
        result = torch.from_numpy(self.outputs[0]['host'].reshape(output_shape)).cuda()
        
        return result


if __name__ == "__main__":
    import sys
    import os
    
    '''
    使用方式:
       python3 pytorch2trt_fusion_v2.py export <edge_border> <height> <width>
       python3 pytorch2trt_fusion_v2.py export 4 240 320
    '''
    DEFAULT_EDGE_BORDER = 1           # 邊緣粗細 (數值越大，邊緣越粗)
    DEFAULT_HEIGHT = 240              # 圖像高度
    DEFAULT_WIDTH = 320               # 圖像寬度
    DEFAULT_OUTPUT_DIR = "/circ330/forgithub/VisualFusion_libtorch/tensorRT_nx/model/NX"
    if len(sys.argv) > 1 and sys.argv[1] == "export":
        # 導出模式
        edge_border = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_EDGE_BORDER
        height = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_HEIGHT
        width = int(sys.argv[4]) if len(sys.argv) > 4 else DEFAULT_WIDTH
        output_dir = sys.argv[5] if len(sys.argv) > 5 else DEFAULT_OUTPUT_DIR
        
        print("=" * 60)
        print("Exporting ImageFusionV2 to ONNX and TensorRT ")
        print("=" * 60)
        print(f"  edge_border:  {edge_border}")
        print(f"  image size:   {width} x {height}")
        print(f"  output_dir:   {output_dir}")
        print("=" * 60)
        
        # 確保輸出目錄存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 執行導出
        export_and_convert(
            edge_border=edge_border, 
            height=height, 
            width=width, 
            output_dir=output_dir
        )
        
        print()
        print("=" * 60)
        print("完成！")
        print(f"TRT 引擎已導出到: {output_dir}/border_{edge_border}_fusion.trt")
        print()
        print("如需使用此引擎，請更新 config.json:")
        print(f'  "fusion_trt_engine": "./model/NX/border_{edge_border}_fusion.trt"')
        print("=" * 60)
    else:
        # 測試模式 (原有代碼)
        import time
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device}")
        print("=" * 50)
