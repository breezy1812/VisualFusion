class SemLA(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  backbone : __torch__.model_jit.reg.SemLA_Reg
  def forward(self: __torch__.model_jit.SemLA.SemLA,
    img_vi: Tensor,
    img_ir: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    _0 = __torch__.model_jit.utils.n_c_h_w_2_n_hw_c
    _1 = __torch__.torch.nn.functional.___torch_mangle_70.interpolate
    backbone = self.backbone
    _2 = (backbone).forward(torch.cat([img_vi, img_ir]), )
    feat_reg_vi_final, feat_reg_ir_final, feat_sa_vi, feat_sa_ir, = _2
    sa_vi = torch.reshape(feat_sa_vi, [-1])
    sa_ir = torch.reshape(feat_sa_ir, [-1])
    sa_vi0 = (torch.where(torch.gt(sa_vi, 0)))[0]
    sa_ir0 = (torch.where(torch.gt(sa_ir, 0)))[0]
    feat_reg_vi = _0(feat_reg_vi_final, )
    feat_reg_ir = _0(feat_reg_ir_final, )
    _3 = torch.slice(feat_reg_vi)
    _4 = annotate(List[Optional[Tensor]], [None, sa_vi0])
    feat_reg_vi0 = torch.index(_3, _4)
    _5 = torch.slice(feat_reg_ir)
    _6 = annotate(List[Optional[Tensor]], [None, sa_ir0])
    feat_reg_ir0 = torch.index(_5, _6)
    _7 = torch.pow((torch.size(feat_reg_vi0))[-1], 0.5)
    feat_reg_vi1 = torch.div(feat_reg_vi0, _7)
    _8 = torch.pow((torch.size(feat_reg_ir0))[-1], 0.5)
    feat_reg_ir1 = torch.div(feat_reg_ir0, _8)
    _9 = torch.einsum("nlc,nsc->nls", [feat_reg_vi1, feat_reg_ir1])
    conf = torch.div(_9, 0.10000000000000001)
    _10 = torch.ones_like(conf)
    _11, _12 = torch.max(conf, 2, True)
    _13 = torch.mul(_10, torch.eq(conf, _11))
    _14, _15 = torch.max(conf, 1, True)
    ones = torch.mul(_13, torch.eq(conf, _14))
    zeros = torch.zeros_like(conf)
    mask = torch.where(torch.gt(conf, 0), ones, zeros)
    mask_v, all_j_ids = torch.max(mask, 2)
    b_ids, i_ids, = torch.where(mask_v)
    _16 = annotate(List[Optional[Tensor]], [b_ids, i_ids])
    j_ids = torch.index(all_j_ids, _16)
    _17 = annotate(List[Optional[Tensor]], [i_ids])
    i_ids0 = torch.index(sa_vi0, _17)
    _18 = annotate(List[Optional[Tensor]], [j_ids])
    j_ids0 = torch.index(sa_ir0, _18)
    _19 = torch.remainder(i_ids0, (torch.size(feat_sa_vi))[3])
    _20 = torch.div(i_ids0, (torch.size(feat_sa_vi))[3], rounding_mode="trunc")
    mkpts0 = torch.mul(torch.stack([_19, _20], 1), 8)
    _21 = torch.remainder(j_ids0, (torch.size(feat_sa_vi))[3])
    _22 = torch.div(j_ids0, (torch.size(feat_sa_vi))[3], rounding_mode="trunc")
    mkpts1 = torch.mul(torch.stack([_21, _22], 1), 8)
    sa_ir1 = _1(feat_sa_ir, None, 8., "bilinear", True, None, )
    _23 = (mkpts0, mkpts1, feat_sa_vi, feat_sa_ir, sa_ir1)
    return _23
