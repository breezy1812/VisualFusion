"""
pytorch_fusion_v2.py

改進版圖像融合 - 解決邊緣縮小和邊框問題
使用 Sobel 邊緣檢測 + 拉普拉斯增強，不使用 roll 操作

termianl: 
# 導出模型 (edge_border=4, 高度=240, 寬度=320)
python3 pytorch2trt_fusion_v2.py export 4 240 320
"""

import torch
import torch.nn.functional as F


class ImageFusionV2:
    """
    改進版圖像融合類
    - 使用標準 Sobel 邊緣檢測
    - 使用 reflect padding 避免邊界問題
    - 不使用 roll 操作，避免邊緣偏移
    """

    def __init__(self, edge_strength: float = 1.0, blur_kernel_size: int = 3, 
                 image_width: int = 320, image_height: int = 240, device: str = "cuda"):
        """
        初始化融合器
        
        Args:
            edge_strength: 邊緣強度係數 (越大邊緣越明顯)
            blur_kernel_size: 高斯模糊核大小
            image_width: 圖像寬度
            image_height: 圖像高度
            device: 運算設備
        """
        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.edge_strength = edge_strength
        self.image_width = image_width
        self.image_height = image_height

        # Sobel 算子 - 標準版本
        self.sobel_x = torch.tensor([[[[-1, 0, 1],
                                        [-2, 0, 2],
                                        [-1, 0, 1]]]], dtype=torch.float32, device=self.device)
        
        self.sobel_y = torch.tensor([[[[-1, -2, -1],
                                        [ 0,  0,  0],
                                        [ 1,  2,  1]]]], dtype=torch.float32, device=self.device)

        # 高斯模糊核 (3x3)
        self.gaussian_3x3 = torch.tensor([[[[0.0625, 0.125, 0.0625],
                                            [0.125,  0.25,  0.125],
                                            [0.0625, 0.125, 0.0625]]]], dtype=torch.float32, device=self.device)
        
        # 拉普拉斯算子 - 用於邊緣增強
        self.laplacian = torch.tensor([[[[ 0, -1,  0],
                                         [-1,  4, -1],
                                         [ 0, -1,  0]]]], dtype=torch.float32, device=self.device)

    def _conv2d_reflect(self, x: torch.Tensor, kernel: torch.Tensor, padding: int = 1) -> torch.Tensor:
        """
        使用 reflect padding 的卷積，避免邊界問題
        """
        # 使用 reflect padding 而不是 zero padding
        x_padded = F.pad(x, (padding, padding, padding, padding), mode='reflect')
        return F.conv2d(x_padded, kernel, padding=0)

    def edge_detect_sobel(self, source: torch.Tensor) -> torch.Tensor:
        """
        標準 Sobel 邊緣檢測
        
        Args:
            source: [1, 1, H, W] 灰度圖
            
        Returns:
            edge: [1, 1, H, W] 邊緣圖
        """
        if source.dim() != 4 or source.size(0) != 1 or source.size(1) != 1:
            raise RuntimeError("edge_detect_sobel: Source must be a 4D tensor with shape [1, 1, H, W]")

        # 先做高斯模糊降噪
        blur = self._conv2d_reflect(source, self.gaussian_3x3, padding=1)

        # Sobel 梯度
        grad_x = self._conv2d_reflect(blur, self.sobel_x, padding=1)
        grad_y = self._conv2d_reflect(blur, self.sobel_y, padding=1)

        # 計算梯度幅值
        edge = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        
        return edge

    def edge_detect_laplacian(self, source: torch.Tensor) -> torch.Tensor:
        """
        拉普拉斯邊緣檢測
        
        Args:
            source: [1, 1, H, W] 灰度圖
            
        Returns:
            edge: [1, 1, H, W] 邊緣圖
        """
        if source.dim() != 4 or source.size(0) != 1 or source.size(1) != 1:
            raise RuntimeError("edge_detect_laplacian: Source must be a 4D tensor with shape [1, 1, H, W]")

        # 先做高斯模糊降噪
        blur = self._conv2d_reflect(source, self.gaussian_3x3, padding=1)

        # 拉普拉斯邊緣
        edge = self._conv2d_reflect(blur, self.laplacian, padding=1)
        
        # 取絕對值
        edge = torch.abs(edge)
        
        return edge

    def edge_detect_combined(self, source: torch.Tensor) -> torch.Tensor:
        """
        組合邊緣檢測 (Sobel + Laplacian)
        
        Args:
            source: [1, 1, H, W] 灰度圖
            
        Returns:
            edge: [1, 1, H, W] 邊緣圖
        """
        sobel_edge = self.edge_detect_sobel(source)
        laplacian_edge = self.edge_detect_laplacian(source)
        
        # 組合兩種邊緣檢測結果
        edge = (sobel_edge + laplacian_edge * 0.5) / 1.5
        
        return edge

    def normalize_edge(self, edge: torch.Tensor, method: str = "minmax") -> torch.Tensor:
        """
        正規化邊緣圖
        
        Args:
            edge: 邊緣圖
            method: 正規化方法 ("minmax", "std", "adaptive")
            
        Returns:
            normalized edge
        """
        if method == "minmax":
            # Min-Max 正規化到 [0, 1]
            e_min = edge.min()
            e_max = edge.max()
            if e_max - e_min > 1e-8:
                edge = (edge - e_min) / (e_max - e_min)
            else:
                edge = torch.zeros_like(edge)
                
        elif method == "std":
            # 標準化後映射到 [0, 1]
            mean = edge.mean()
            std = edge.std()
            if std > 1e-8:
                edge = (edge - mean) / (std * 3)  # 3 sigma
                edge = torch.clamp(edge, -1, 1) * 0.5 + 0.5
            else:
                edge = torch.zeros_like(edge)
                
        elif method == "adaptive":
            # 自適應正規化
            # 使用百分位數避免異常值影響
            flat = edge.flatten()
            p2 = torch.quantile(flat, 0.02)
            p98 = torch.quantile(flat, 0.98)
            if p98 - p2 > 1e-8:
                edge = torch.clamp((edge - p2) / (p98 - p2), 0, 1)
            else:
                edge = torch.zeros_like(edge)
                
        return edge

    def fusion_additive(self, edge: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        加法融合 - 將邊緣疊加到目標圖像上
        
        Args:
            edge: [1, 1, H, W] 邊緣圖
            target: [1, 3, H, W] 目標彩色圖
            
        Returns:
            fused: [1, 3, H, W] 融合結果
        """
        # 擴展邊緣到 3 通道
        edge_3ch = edge.repeat(1, 3, 1, 1)
        
        # 加法融合
        fused = target + edge_3ch * self.edge_strength
        
        return torch.clamp(fused, 0, 1)

    def fusion_blend(self, edge: torch.Tensor, target: torch.Tensor, 
                     source_color: torch.Tensor = None) -> torch.Tensor:
        """
        混合融合 - 使用邊緣作為混合權重
        
        Args:
            edge: [1, 1, H, W] 邊緣圖
            target: [1, 3, H, W] 目標彩色圖 (IR)
            source_color: [1, 3, H, W] 來源彩色圖 (EO)，可選
            
        Returns:
            fused: [1, 3, H, W] 融合結果
        """
        # 使用邊緣作為混合權重
        weight = edge * self.edge_strength
        weight = torch.clamp(weight, 0, 1)
        weight_3ch = weight.repeat(1, 3, 1, 1)
        
        if source_color is not None:
            # 有來源彩色圖時，根據邊緣權重混合兩張圖
            fused = target * (1 - weight_3ch) + source_color * weight_3ch
        else:
            # 無來源彩色圖時，增強邊緣區域亮度
            fused = target + weight_3ch * 0.5
            
        return torch.clamp(fused, 0, 1)

    def fusion_highlight(self, edge: torch.Tensor, target: torch.Tensor, 
                         highlight_color: tuple = (0, 1, 0)) -> torch.Tensor:
        """
        高亮融合 - 用指定顏色高亮邊緣
        
        Args:
            edge: [1, 1, H, W] 邊緣圖
            target: [1, 3, H, W] 目標彩色圖
            highlight_color: RGB 顏色 (0-1 範圍)
            
        Returns:
            fused: [1, 3, H, W] 融合結果
        """
        # 創建高亮顏色層
        h, w = edge.shape[2], edge.shape[3]
        highlight = torch.zeros((1, 3, h, w), dtype=edge.dtype, device=edge.device)
        highlight[:, 0, :, :] = highlight_color[0]
        highlight[:, 1, :, :] = highlight_color[1]
        highlight[:, 2, :, :] = highlight_color[2]
        
        # 使用邊緣作為混合權重
        weight = edge * self.edge_strength
        weight = torch.clamp(weight, 0, 1)
        weight_3ch = weight.repeat(1, 3, 1, 1)
        
        # 混合
        fused = target * (1 - weight_3ch) + highlight * weight_3ch
        
        return torch.clamp(fused, 0, 1)

    def forward(self, eo_gray: torch.Tensor, ir_color: torch.Tensor, 
                mode: str = "additive", normalize: str = "minmax") -> torch.Tensor:
        """
        完整融合流程
        
        Args:
            eo_gray: [1, 1, H, W] EO 灰度圖
            ir_color: [1, 3, H, W] IR 彩色圖
            mode: 融合模式 ("additive", "blend", "highlight")
            normalize: 正規化方法 ("minmax", "std", "adaptive")
            
        Returns:
            fused: [1, 3, H, W] 融合結果
        """
        # 邊緣檢測
        edge = self.edge_detect_sobel(eo_gray)
        
        # 正規化
        edge = self.normalize_edge(edge, method=normalize)
        
        # 融合
        if mode == "additive":
            fused = self.fusion_additive(edge, ir_color)
        elif mode == "blend":
            fused = self.fusion_blend(edge, ir_color)
        elif mode == "highlight":
            fused = self.fusion_highlight(edge, ir_color)
        else:
            raise ValueError(f"Unknown fusion mode: {mode}")
            
        return fused

    def forward_with_edge(self, eo_gray: torch.Tensor, ir_color: torch.Tensor,
                          mode: str = "additive", normalize: str = "minmax") -> tuple:
        """
        完整融合流程，同時返回邊緣圖
        
        Returns:
            (fused, edge): 融合結果和邊緣圖
        """
        edge = self.edge_detect_sobel(eo_gray)
        edge = self.normalize_edge(edge, method=normalize)
        
        if mode == "additive":
            fused = self.fusion_additive(edge, ir_color)
        elif mode == "blend":
            fused = self.fusion_blend(edge, ir_color)
        elif mode == "highlight":
            fused = self.fusion_highlight(edge, ir_color)
        else:
            raise ValueError(f"Unknown fusion mode: {mode}")
            
        return fused, edge


class ImageFusionV2Simple:
    """
    改進版 - 與原版相同的陰影邏輯，但使用 padding 替代 roll 避免循環邊界問題
    """
    
    def __init__(self, edge_border: int = 3, image_width: int = 320, image_height: int = 240, device: str = "cuda"):
        """
        Args:
            edge_border: 邊緣寬度/偏移量 (與原版相同)
            image_width: 圖像寬度
            image_height: 圖像高度
            device: 運算設備
        """
        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        self.edge_border = edge_border
        self.image_width = image_width
        self.image_height = image_height
        
        # Sobel 算子 (與原版相同)
        self.sobel_x = torch.tensor([[[[0, 0, 0],
                                        [1, 0, -1],
                                        [0, 0, 0]]]], dtype=torch.float32, device=self.device)
        self.sobel_y = torch.tensor([[[[0, 1, 0],
                                        [0, 0, 0],
                                        [0, -1, 0]]]], dtype=torch.float32, device=self.device)
        
        # 高斯核 (與原版相同)
        self.gaussian = torch.tensor([[[[0.0751, 0.1238, 0.0751],
                                        [0.1238, 0.2042, 0.1238],
                                        [0.0751, 0.1238, 0.0751]]]], dtype=torch.float32, device=self.device)

    def _shift_no_wrap(self, x: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
        """
        平移但不循環，邊界填充 0
        shift > 0: 向正方向移動
        dim: 2 = Y方向, 3 = X方向
        """
        n, c, h, w = x.shape
        
        if dim == 3:  # X 方向
            if shift > 0:  # 向右
                result = F.pad(x[:, :, :, :-shift], (shift, 0, 0, 0), mode='constant', value=0)
            elif shift < 0:  # 向左
                result = F.pad(x[:, :, :, -shift:], (0, -shift, 0, 0), mode='constant', value=0)
            else:
                result = x
        else:  # dim == 2, Y 方向
            if shift > 0:  # 向下
                result = F.pad(x[:, :, :-shift, :], (0, 0, shift, 0), mode='constant', value=0)
            elif shift < 0:  # 向上
                result = F.pad(x[:, :, -shift:, :], (0, 0, 0, -shift), mode='constant', value=0)
            else:
                result = x
        
        return result

    def edge(self, source: torch.Tensor) -> torch.Tensor:
        """
        邊緣檢測 + 陰影效果 (與原版相同的邏輯)
        """
        if source.dim() != 4 or source.size(0) != 1 or source.size(1) != 1:
            raise RuntimeError("edge: Source must be a 4D tensor with shape [1, 1, H, W]")
        
        # 使用 reflect padding 進行卷積
        padded = F.pad(source, (1, 1, 1, 1), mode='reflect')
        blur = F.conv2d(padded, self.gaussian, padding=0)
        
        padded_blur = F.pad(blur, (1, 1, 1, 1), mode='reflect')
        sobel_x = F.conv2d(padded_blur, self.sobel_x, padding=0)
        sobel_y = F.conv2d(padded_blur, self.sobel_y, padding=0)
        
        border = self.edge_border
        
        # X 方向: 與原版相同的邏輯，但用 _shift_no_wrap 替代 roll
        edge_x = torch.zeros_like(sobel_x)
        # sobel_x < 1.0 (實際上是所有負值和小正值): 向右偏移
        shifted_right = self._shift_no_wrap(sobel_x, border, dim=3)
        edge_x = torch.where(sobel_x < 1.0, shifted_right, edge_x)
        # sobel_x > 1.0 (實際上不太可能發生，因為值很小): 向左偏移
        shifted_left = self._shift_no_wrap(-sobel_x, -border, dim=3)
        edge_x = torch.where(sobel_x > 1.0, shifted_left, edge_x)
        
        # Y 方向
        edge_y = torch.zeros_like(sobel_y)
        shifted_down = self._shift_no_wrap(sobel_y, border, dim=2)
        edge_y = torch.where(sobel_y < 1.0, shifted_down, edge_y)
        shifted_up = self._shift_no_wrap(-sobel_y, -border, dim=2)
        edge_y = torch.where(sobel_y > 1.0, shifted_up, edge_y)
        
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
    
    # 測試完整版
    print("Testing ImageFusionV2 (Full version)...")
    fusion_v2 = ImageFusionV2(edge_strength=0.5, device=device)
    
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
    print("Testing ImageFusionV2Simple...")
    fusion_simple = ImageFusionV2Simple(edge_border=4, device=device)
    
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
        """平移但不循環，邊界填充 0"""
        if dim == 3:  # X 方向
            if shift > 0:  # 向右
                result = F.pad(x[:, :, :, :-shift], (shift, 0, 0, 0), mode='constant', value=0)
            elif shift < 0:  # 向左
                result = F.pad(x[:, :, :, -shift:], (0, -shift, 0, 0), mode='constant', value=0)
            else:
                result = x
        else:  # dim == 2, Y 方向
            if shift > 0:  # 向下
                result = F.pad(x[:, :, :-shift, :], (0, 0, shift, 0), mode='constant', value=0)
            elif shift < 0:  # 向上
                result = F.pad(x[:, :, -shift:, :], (0, 0, 0, -shift), mode='constant', value=0)
            else:
                result = x
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
        
        # X 方向
        edge_x = torch.zeros_like(sobel_x)
        shifted_right = self._shift_no_wrap(sobel_x, border, dim=3)
        edge_x = torch.where(sobel_x < 1.0, shifted_right, edge_x)
        shifted_left = self._shift_no_wrap(-sobel_x, -border, dim=3)
        edge_x = torch.where(sobel_x > 1.0, shifted_left, edge_x)
        
        # Y 方向
        edge_y = torch.zeros_like(sobel_y)
        shifted_down = self._shift_no_wrap(sobel_y, border, dim=2)
        edge_y = torch.where(sobel_y < 1.0, shifted_down, edge_y)
        shifted_up = self._shift_no_wrap(-sobel_y, -border, dim=2)
        edge_y = torch.where(sobel_y > 1.0, shifted_up, edge_y)
        
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
    """
    導出模型為 ONNX 格式
    
    Args:
        model: ImageFusionModule 實例
        onnx_path: 輸出的 ONNX 文件路徑
        height: 圖像高度
        width: 圖像寬度
        opset_version: ONNX opset 版本
    """
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
    """
    將 ONNX 模型轉換為 TensorRT 引擎 (固定使用 FP32)
    
    Args:
        onnx_path: ONNX 模型路徑
        trt_path: 輸出的 TensorRT 引擎路徑
        height: 圖像高度
        width: 圖像寬度
        workspace_mb: 工作空間大小 (MB)，預設 256MB 適合 Jetson
    """
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
    # ============================================================
    # 導出設定 - 在這裡修改預設參數
    # ============================================================
    DEFAULT_EDGE_BORDER = 1           # 邊緣粗細 (數值越大，邊緣越粗)
    DEFAULT_HEIGHT = 240              # 圖像高度
    DEFAULT_WIDTH = 320               # 圖像寬度
    DEFAULT_OUTPUT_DIR = "/circ330/forgithub/VisualFusion_libtorch/tensorRT_nx/model/NX"
    # ============================================================
    
    
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
