import torch
import os
import random
import numpy as np
from torch import nn
from .utils import CBR, DWConv, MLP, MLP2, DWT_2D, StructureAttention
from .utils import (
    n_c_h_w_2_n_hw_c,
    n_h_w_c_2_n_c_h_w,
    n_c_h_w_2_n_c_hw,
    n_hw_c_2_n_c_h_w,
)

# # ============================================================================
# # 🔒 完整的確定性設置（確保 RTX 30 系列與 GTX 1080 Ti 一致）
# # 直接在模塊導入時執行，無需函數包裝，避免 JIT 轉換問題
# # ============================================================================

# 1. Python 隨機種子
random.seed(42)

# 2. NumPy 隨機種子
np.random.seed(42)

# 3. PyTorch CPU 隨機種子
torch.manual_seed(42)

# 4. PyTorch GPU 隨機種子
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# 5. 強制使用確定性算法
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ============================================================================
# 🔥 關鍵：禁用 TF32（RTX 30 系列的關鍵設置）
# ============================================================================
# TF32 在 Ampere 架構（RTX 3070/3080/3090）上默認啟用
# 會導致 BatchNorm2d 計算結果與 Pascal 架構（GTX 1080 Ti）不一致

# 禁用 CUDA matmul 的 TF32
if hasattr(torch.backends.cuda, 'matmul'):
    torch.backends.cuda.matmul.allow_tf32 = False

# 禁用 cuDNN 的 TF32
if hasattr(torch.backends.cudnn, 'allow_tf32'):
    torch.backends.cudnn.allow_tf32 = False

# 6. 設置環境變量（禁用所有異步和並行優化）
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 同步執行，禁用異步優化
os.environ['PYTHONHASHSEED'] = '42'


class SemLA_Reg(nn.Module):
    """
    The registration section of SemLA
    """

    def __init__(self, device, fp=torch.float32):
        super().__init__()
        # Discrete Wavelet Transform (For feature map downsampling)
        self.fp = fp
        self.dwt = DWT_2D(wave="haar", fp=fp)

        self.reg0 = JConv(1, 8)
        self.reg1 = JConv(32, 16)
        self.reg2 = JConv(64, 32)
        self.reg3 = JConv(128, 256)
        self.pred_reg = nn.Sequential(JConv(256, 256), JConv(256, 256), JConv(256, 256))

        self.sa0 = JConv(256, 256)
        self.sa1 = JConv(256, 128)
        self.sa2 = JConv(128, 32)
        self.sa3 = JConv(32, 1)
        self.pred_sa = nn.Sigmoid()

        self.csc0 = CrossModalAttention(256)
        self.csc1 = CrossModalAttention(256)

        self.ssr = SemanticStructureRepresentation(device, fp)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Extraction of registration features
        
        x0 = self.reg0(x)
        x1 = self.reg1(self.dwt(x0))
        x2 = self.reg2(self.dwt(x1))
        x3 = self.reg3(self.dwt(x2))
        feat_reg = self.pred_reg(x3)
        # return x1, x1

        # trace modify
        bs2 = feat_reg.shape[0] //2
        (feat_reg_vi, feat_reg_ir) = feat_reg.split(bs2)
        h = feat_reg.shape[2]
        w = feat_reg.shape[3]

        # Predicting semantic awareness maps for infrared images
        feat_sa_ir = self.sa0(feat_reg_ir)
        feat_sa_ir = self.sa1(feat_sa_ir)
        feat_sa_ir = self.sa2(feat_sa_ir)
        feat_sa_ir = self.sa3(feat_sa_ir)
        feat_sa_ir = self.pred_sa(feat_sa_ir)
        # return feat_sa_ir, feat_sa_ir
    
        # Flatten
        feat_sa_ir_flatten = n_c_h_w_2_n_c_hw(feat_sa_ir)
        feat_reg_vi_flatten_ = n_c_h_w_2_n_hw_c(feat_reg_vi)
        feat_reg_ir_flatten = n_c_h_w_2_n_hw_c(feat_reg_ir)

        # Feature Similarity Calculation
        feat_reg_vi_flatten = (
            feat_reg_vi_flatten_ / feat_reg_vi_flatten_.shape[-1] ** 0.5
        )
        feat_reg_ir_flatten = feat_reg_ir_flatten / feat_reg_ir_flatten.shape[-1] ** 0.5

        attention = (
            torch.einsum("nlc,nsc->nls", feat_reg_vi_flatten, feat_reg_ir_flatten) / 0.1
        )
        attention = attention.softmax(dim=1).to(dtype=self.fp)

        # Generate cross-modal guidance information
        attention = torch.einsum("nls,ncs->nls", attention, feat_sa_ir_flatten)
        attention = torch.sum(attention, dim=2).to(dtype=self.fp)

        # Calibration of semantic features of visible images
        feat_reg_vi_ca = self.csc0(feat_reg_vi_flatten_, attention * 1.5)
        feat_reg_vi_ca = self.csc1(feat_reg_vi_ca, attention * 1.5)
        feat_reg_vi_ca = n_hw_c_2_n_c_h_w(feat_reg_vi_ca, h=h, w=w)

        # Predicting semantic awareness maps for visible images
        feat_sa_vi = self.sa0(feat_reg_vi_ca)
        feat_sa_vi = self.sa1(feat_sa_vi)
        feat_sa_vi = self.sa2(feat_sa_vi)
        feat_sa_vi = self.sa3(feat_sa_vi)
        feat_sa_vi = self.pred_sa(feat_sa_vi)

        # Semantic structure representation learning
        feat_reg_vi_str, feat_reg_ir_str = self.ssr(feat_sa_vi, feat_sa_ir)
        feat_reg_vi_final = feat_reg_vi + feat_reg_vi_str
        feat_reg_ir_final = feat_reg_ir + feat_reg_ir_str

        return feat_reg_vi_final, feat_reg_ir_final, feat_sa_vi, feat_sa_ir
        return feat_sa_vi, feat_sa_ir
        # return feat_reg_vi_final, feat_reg_ir_final


class JConv(nn.Module):
    """Joint Convolutional blocks

    Args:
        'x' (torch.Tensor): (N, C, H, W)
    """

    def __init__(self, in_channels, out_channels):
        super(JConv, self).__init__()
        self.feat_trans = CBR(in_channels, out_channels)
        self.dwconv = DWConv(out_channels)
        self.norm = nn.BatchNorm2d(out_channels, eps=1e-5)  # 使用官方 BatchNorm2d
        self.mlp = MLP(out_channels, bias=True)

    def forward(self, x):
        # 確定性設置已在模塊導入時完成
        x = self.feat_trans(x)
        x = x + self.dwconv(x)
        out = self.norm(x)
        x = x + self.mlp(out)
        return x


class CrossModalAttention(nn.Module):
    """Cross-modal semantic calibration

    Args:
        'feat' (torch.Tensor): (N, L)
        'attention' (torch.Tensor): (N, L, C)
    """

    def __init__(self, dim, mlp_ratio=2, qkv_bias=False):
        super(CrossModalAttention, self).__init__()
        self.qkv = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_out = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP2(dim, mlp_ratio)

    def forward(self, feat, attention):
        shortcut = feat
        feat = self.norm1(feat)
        feat = self.qkv(feat)
        x = torch.einsum("nl, nlc -> nlc", attention, feat)
        x = self.proj_out(x)
        x = x + shortcut
        x = x + self.mlp(self.norm2(x))
        return x


class SemanticStructureRepresentation(nn.Module):
    """Cross-modal semantic calibration

    Args:
        'feat' (torch.Tensor): (N, L)
        'attention' (torch.Tensor): (N, L, C)
    """

    def __init__(self, device, fp=torch.float32):
        super(SemanticStructureRepresentation, self).__init__()
        self.grid_embedding = JConv(2, 256)
        self.semantic_embedding = JConv(256, 256)
        self.attention = StructureAttention(256, 8)
        self.device = device
        self.fp = fp

    def forward(self, feat_sa_vi, feat_sa_ir):
        feat_h = feat_sa_vi.shape[2]
        feat_w = feat_sa_vi.shape[3]
        # Predefined spatial grid
        xs = torch.linspace(0, feat_h - 1, feat_h)
        ys = torch.linspace(0, feat_w - 1, feat_w)
        xs = xs / (feat_h - 1)
        ys = ys / (feat_w - 1)

        grid = torch.stack(torch.meshgrid(xs, ys, indexing='ij'), dim=-1)
        grid = grid.unsqueeze(0).repeat(feat_sa_vi.shape[0], 1, 1, 1).to(self.device, dtype=self.fp)

        h = grid.shape[1]
        w = grid.shape[2]
        grid = n_h_w_c_2_n_c_h_w(grid)

        # Embedding position information into a high-dimensional space
        grid = self.grid_embedding(grid)

        # Embedding semantic information
        semantic_grid_vi = grid * feat_sa_vi
        semantic_grid_ir = grid * feat_sa_ir

        semantic_grid_vi = self.semantic_embedding(semantic_grid_vi)
        semantic_grid_ir = self.semantic_embedding(semantic_grid_ir)

        semantic_grid_vi = n_c_h_w_2_n_hw_c(semantic_grid_vi)
        semantic_grid_ir = n_c_h_w_2_n_hw_c(semantic_grid_ir)

        semantic_grid_vi = self.attention(semantic_grid_vi)
        semantic_grid_ir = self.attention(semantic_grid_ir)

        semantic_grid_vi = n_hw_c_2_n_c_h_w(semantic_grid_vi, h=h, w=w)
        semantic_grid_ir = n_hw_c_2_n_c_h_w(semantic_grid_ir, h=h, w=w)

        return semantic_grid_vi, semantic_grid_ir
