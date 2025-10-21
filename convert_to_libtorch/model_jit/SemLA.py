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
        feat_reg_vi_final, feat_reg_ir_final, feat_sa_vi, feat_sa_ir = self.backbone(
            torch.cat((img_vi, img_ir), dim=0)
        )
        batch_size = 1
        height = 30
        width = 40
        fixed_num_points = 1200
        feat_reg_vi = n_c_h_w_2_n_hw_c(feat_reg_vi_final)
        feat_reg_ir = n_c_h_w_2_n_hw_c(feat_reg_ir_final)
        feat_reg_vi = feat_reg_vi / (feat_reg_vi.shape[-1] ** 0.5)
        feat_reg_ir = feat_reg_ir / (feat_reg_ir.shape[-1] ** 0.5)
        conf = torch.einsum("nlc,nsc->nls", feat_reg_vi, feat_reg_ir) / 0.1
        conf_max_val, conf_max_idx = conf.max(dim=2)
        mask_forward = (conf == conf.max(dim=2, keepdim=True)[0]).float()
        mask_backward = (conf == conf.max(dim=1, keepdim=True)[0]).float()
        mask_mutual = mask_forward * mask_backward
        _, j_ids_all = mask_mutual.max(dim=2)
        y_coords = torch.arange(height, device=img_vi.device).view(-1, 1).repeat(1, width).view(-1)
        x_coords = torch.arange(width, device=img_vi.device).view(1, -1).repeat(height, 1).view(-1)
        mkpts0 = torch.stack([x_coords, y_coords], dim=1).float() * 8.0
        j_ids = j_ids_all[0]
        j_y = j_ids // width
        j_x = j_ids - j_y * width
        mkpts1 = torch.stack([j_x.float(), j_y.float()], dim=1) * 8.0
        mask_valid = mask_mutual.sum(dim=2)[0]
        mask_valid = (mask_valid > 0).float()
        mkpts0_final = mkpts0 * mask_valid.unsqueeze(1)
        mkpts1_final = mkpts1 * mask_valid.unsqueeze(1)
        mkpts0_final = mkpts0_final.to(torch.int32)
        mkpts1_final = mkpts1_final.to(torch.int32)
        return mkpts0_final, mkpts1_final
