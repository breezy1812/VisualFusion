import torch
import torch.nn as nn
import torch.nn.functional as F
from .reg import SemLA_Reg
from .utils import n_c_h_w_2_n_hw_c
torch.set_printoptions(sci_mode=False)

class SemLA(nn.Module):

    def __init__(self, device, fp=torch.float32):
        super().__init__()
        self.backbone = SemLA_Reg(device, fp)
        
    def forward(self, img_vi, img_ir):
        # ===== 🔧 TensorRT 8.4 完全相容版本：移除所有不支援的運算符 =====
        # 不支援: NonZero, Mod, Where, 動態索引
        # 策略：使用固定形狀，完全靜態的操作
        
        feat_reg_vi_final, feat_reg_ir_final, feat_sa_vi, feat_sa_ir = self.backbone(
            torch.cat((img_vi, img_ir), dim=0)
        )
        
        # 固定尺寸常數
        batch_size = 1
        height = 30  # 240 / 8
        width = 40   # 320 / 8
        fixed_num_points = 1200  # 固定輸出 1200 個特徵點
        
        # 使用所有特徵點（不做閾值篩選，避免 NonZero）
        # feat_reg_vi/ir: [1, 1200, 256]
        feat_reg_vi = n_c_h_w_2_n_hw_c(feat_reg_vi_final)
        feat_reg_ir = n_c_h_w_2_n_hw_c(feat_reg_ir_final)
        
        # 正規化
        feat_reg_vi = feat_reg_vi / (feat_reg_vi.shape[-1] ** 0.5)
        feat_reg_ir = feat_reg_ir / (feat_reg_ir.shape[-1] ** 0.5)
        
        # 計算相似度矩陣 [1, 1200, 1200]
        conf = torch.einsum("nlc,nsc->nls", feat_reg_vi, feat_reg_ir) / 0.1
        
        # 找到每個 vi 特徵對應的最佳 ir 特徵（避免 where）
        conf_max_val, conf_max_idx = conf.max(dim=2)  # [1, 1200]
        
        # 雙向最大值檢查（mutual nearest neighbor）
        # 避免使用 where，改用 max + 比較
        mask_forward = (conf == conf.max(dim=2, keepdim=True)[0]).float()
        mask_backward = (conf == conf.max(dim=1, keepdim=True)[0]).float()
        mask_mutual = mask_forward * mask_backward  # [1, 1200, 1200]
        
        # 取得每個 vi 點的最佳匹配 ir 點索引
        _, j_ids_all = mask_mutual.max(dim=2)  # [1, 1200]
        
        # 生成所有可能的座標（避免使用 % 和動態索引）
        # 使用完全靜態的座標生成
        y_coords = torch.arange(height, device=img_vi.device).view(-1, 1).repeat(1, width).view(-1)
        x_coords = torch.arange(width, device=img_vi.device).view(1, -1).repeat(height, 1).view(-1)
        
        # mkpts0: vi 的座標 [1200, 2]
        mkpts0 = torch.stack([x_coords, y_coords], dim=1).float() * 8.0
        
        # mkpts1: 根據 j_ids_all 取得對應的 ir 座標
        j_ids = j_ids_all[0]  # [1200]
        
        # 避免使用 % 運算符，改用除法和減法
        j_y = j_ids // width  # 整數除法
        j_x = j_ids - j_y * width  # 替代 j_ids % width
        mkpts1 = torch.stack([j_x.float(), j_y.float()], dim=1) * 8.0
        
        # 返回固定大小的輸出 [1200, 2]
        # 計算實際有效的匹配點數量（基於 mutual mask）
        # mask_mutual: [1, 1200, 1200]
        # 不使用 squeeze，直接索引取得 [1200] 的 tensor
        mask_valid = mask_mutual.sum(dim=2)[0]  # 直接索引取 batch 0，避免 squeeze 產生 If 節點
        mask_valid = (mask_valid > 0).float()  # 轉換為 0/1 mask，shape: [1200]
        
        # 將無效點的座標設為 (0, 0)
        # mask_valid: [1200]，1 表示有效，0 表示無效
        mkpts0_final = mkpts0 * mask_valid.unsqueeze(1)  # [1200, 2]
        mkpts1_final = mkpts1 * mask_valid.unsqueeze(1)  # [1200, 2]
        
        # ⚠️ 重要：轉換為 int32，與 C++ 代碼匹配
        # C++ 期望 int32_t 類型的座標
        mkpts0_final = mkpts0_final.to(torch.int32)
        mkpts1_final = mkpts1_final.to(torch.int32)
        
        # 固定輸出 1200 個點
        # C++ 代碼需要遍歷所有點，跳過座標為 (0, 0) 的點
        return mkpts0_final, mkpts1_final
