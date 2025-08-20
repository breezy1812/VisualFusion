class SemLA_Reg(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  fp : int
  dwt : __torch__.model_jit.utils.DWT_2D
  reg0 : __torch__.model_jit.reg.JConv
  reg1 : __torch__.model_jit.reg.___torch_mangle_13.JConv
  reg2 : __torch__.model_jit.reg.___torch_mangle_23.JConv
  reg3 : __torch__.model_jit.reg.___torch_mangle_33.JConv
  pred_reg : __torch__.torch.nn.modules.container.Sequential
  sa0 : __torch__.model_jit.reg.___torch_mangle_36.JConv
  sa1 : __torch__.model_jit.reg.___torch_mangle_46.JConv
  sa2 : __torch__.model_jit.reg.___torch_mangle_49.JConv
  sa3 : __torch__.model_jit.reg.___torch_mangle_59.JConv
  pred_sa : __torch__.torch.nn.modules.activation.Sigmoid
  csc0 : __torch__.model_jit.reg.CrossModalAttention
  csc1 : __torch__.model_jit.reg.CrossModalAttention
  ssr : __torch__.model_jit.reg.SemanticStructureRepresentation
  def forward(self: __torch__.model_jit.reg.SemLA_Reg,
    x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    _0 = __torch__.model_jit.utils.n_c_h_w_2_n_c_hw
    _1 = __torch__.model_jit.utils.n_c_h_w_2_n_hw_c
    _2 = __torch__.model_jit.utils.n_hw_c_2_n_c_h_w
    reg0 = self.reg0
    x0 = (reg0).forward(x, )
    reg1 = self.reg1
    dwt = self.dwt
    x1 = (reg1).forward((dwt).forward(x0, ), )
    reg2 = self.reg2
    dwt0 = self.dwt
    x2 = (reg2).forward((dwt0).forward(x1, ), )
    reg3 = self.reg3
    dwt1 = self.dwt
    x3 = (reg3).forward((dwt1).forward(x2, ), )
    pred_reg = self.pred_reg
    feat_reg = (pred_reg).forward(x3, )
    bs2 = (torch.size(feat_reg))[0]
    _3 = torch.split(feat_reg, torch.floordiv(bs2, 2))
    feat_reg_vi, feat_reg_ir, = _3
    h = (torch.size(feat_reg))[2]
    w = (torch.size(feat_reg))[3]
    sa0 = self.sa0
    feat_sa_ir = (sa0).forward(feat_reg_ir, )
    sa1 = self.sa1
    feat_sa_ir0 = (sa1).forward(feat_sa_ir, )
    sa2 = self.sa2
    feat_sa_ir1 = (sa2).forward(feat_sa_ir0, )
    sa3 = self.sa3
    feat_sa_ir2 = (sa3).forward(feat_sa_ir1, )
    pred_sa = self.pred_sa
    feat_sa_ir3 = (pred_sa).forward(feat_sa_ir2, )
    _4 = _0(feat_sa_ir3, )
    fp = self.fp
    feat_sa_ir_flatten = torch.to(_4, fp)
    _5 = _1(feat_reg_vi, )
    fp0 = self.fp
    feat_reg_vi_flatten_ = torch.to(_5, fp0)
    _6 = _1(feat_reg_ir, )
    fp1 = self.fp
    feat_reg_ir_flatten = torch.to(_6, fp1)
    _7 = (torch.size(feat_reg_vi_flatten_))[-1]
    feat_reg_vi_flatten = torch.div(feat_reg_vi_flatten_, torch.pow(_7, 0.5))
    _8 = (torch.size(feat_reg_ir_flatten))[-1]
    feat_reg_ir_flatten0 = torch.div(feat_reg_ir_flatten, torch.pow(_8, 0.5))
    _9 = [feat_reg_vi_flatten, feat_reg_ir_flatten0]
    attention = torch.div(torch.einsum("nlc,nsc->nls", _9), 0.10000000000000001)
    _10 = torch.softmax(attention, 1)
    fp2 = self.fp
    attention0 = torch.to(_10, fp2)
    attention1 = torch.einsum("nls,ncs->nls", [attention0, feat_sa_ir_flatten])
    _11 = torch.sum(attention1, [2])
    fp3 = self.fp
    attention2 = torch.to(_11, fp3)
    csc0 = self.csc0
    feat_reg_vi_ca = (csc0).forward(feat_reg_vi_flatten_, torch.mul(attention2, 1.5), )
    csc1 = self.csc1
    feat_reg_vi_ca0 = (csc1).forward(feat_reg_vi_ca, torch.mul(attention2, 1.5), )
    feat_reg_vi_ca1 = _2(feat_reg_vi_ca0, h, w, )
    sa00 = self.sa0
    feat_sa_vi = (sa00).forward(feat_reg_vi_ca1, )
    sa10 = self.sa1
    feat_sa_vi0 = (sa10).forward(feat_sa_vi, )
    sa20 = self.sa2
    feat_sa_vi1 = (sa20).forward(feat_sa_vi0, )
    sa30 = self.sa3
    feat_sa_vi2 = (sa30).forward(feat_sa_vi1, )
    pred_sa0 = self.pred_sa
    feat_sa_vi3 = (pred_sa0).forward(feat_sa_vi2, )
    ssr = self.ssr
    _12 = (ssr).forward(feat_sa_vi3, feat_sa_ir3, )
    feat_reg_vi_str, feat_reg_ir_str, = _12
    feat_reg_vi_final = torch.add(feat_reg_vi, feat_reg_vi_str)
    feat_reg_ir_final = torch.add(feat_reg_ir, feat_reg_ir_str)
    _13 = (feat_reg_vi_final, feat_reg_ir_final, feat_sa_vi3, feat_sa_ir3)
    return _13
class JConv(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  feat_trans : __torch__.model_jit.utils.CBR
  dwconv : __torch__.model_jit.utils.DWConv
  norm : __torch__.torch.nn.modules.batchnorm.BatchNorm2d
  mlp : __torch__.model_jit.utils.MLP
  def forward(self: __torch__.model_jit.reg.JConv,
    x: Tensor) -> Tensor:
    feat_trans = self.feat_trans
    x0 = (feat_trans).forward(x, )
    dwconv = self.dwconv
    x1 = torch.add(x0, (dwconv).forward(x0, ))
    norm = self.norm
    out = (norm).forward(x1, )
    mlp = self.mlp
    return torch.add(x1, (mlp).forward(out, ))
class CrossModalAttention(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  qkv : __torch__.torch.nn.modules.linear.Linear
  proj_out : __torch__.torch.nn.modules.linear.___torch_mangle_60.Linear
  norm1 : __torch__.torch.nn.modules.normalization.LayerNorm
  norm2 : __torch__.torch.nn.modules.normalization.LayerNorm
  mlp : __torch__.model_jit.utils.MLP2
  def forward(self: __torch__.model_jit.reg.CrossModalAttention,
    feat: Tensor,
    attention: Tensor) -> Tensor:
    norm1 = self.norm1
    feat0 = (norm1).forward(feat, )
    qkv = self.qkv
    feat1 = (qkv).forward(feat0, )
    x = torch.einsum("nl, nlc -> nlc", [attention, feat1])
    proj_out = self.proj_out
    x2 = (proj_out).forward(x, )
    x3 = torch.add(x2, feat)
    mlp = self.mlp
    norm2 = self.norm2
    _14 = (mlp).forward((norm2).forward(x3, ), )
    return torch.add(x3, _14)
class SemanticStructureRepresentation(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  device : Device
  fp : int
  grid_embedding : __torch__.model_jit.reg.___torch_mangle_66.JConv
  semantic_embedding : __torch__.model_jit.reg.___torch_mangle_36.JConv
  attention : __torch__.model_jit.utils.StructureAttention
  def forward(self: __torch__.model_jit.reg.SemanticStructureRepresentation,
    feat_sa_vi: Tensor,
    feat_sa_ir: Tensor) -> Tuple[Tensor, Tensor]:
    _15 = __torch__.model_jit.utils.n_h_w_c_2_n_c_h_w
    _16 = __torch__.model_jit.utils.n_c_h_w_2_n_hw_c
    _17 = __torch__.model_jit.utils.n_hw_c_2_n_c_h_w
    feat_h = (torch.size(feat_sa_vi))[2]
    feat_w = (torch.size(feat_sa_vi))[3]
    xs = torch.linspace(0, torch.sub(feat_h, 1), feat_h)
    ys = torch.linspace(0, torch.sub(feat_w, 1), feat_w)
    xs0 = torch.div(xs, torch.sub(feat_h, 1))
    ys0 = torch.div(ys, torch.sub(feat_w, 1))
    _18 = torch.meshgrid([xs0, ys0], indexing="ij")
    _19 = torch.unsqueeze(torch.stack(_18, -1), 0)
    _20 = [(torch.size(feat_sa_vi))[0], 1, 1, 1]
    _21 = torch.repeat(_19, _20)
    device = self.device
    fp = self.fp
    grid = torch.to(_21, device, fp)
    h = (torch.size(grid))[1]
    w = (torch.size(grid))[2]
    grid0 = _15(grid, )
    grid_embedding = self.grid_embedding
    grid1 = (grid_embedding).forward(grid0, )
    semantic_grid_vi = torch.mul(grid1, feat_sa_vi)
    semantic_grid_ir = torch.mul(grid1, feat_sa_ir)
    semantic_embedding = self.semantic_embedding
    semantic_grid_vi0 = (semantic_embedding).forward(semantic_grid_vi, )
    semantic_embedding0 = self.semantic_embedding
    semantic_grid_ir0 = (semantic_embedding0).forward(semantic_grid_ir, )
    semantic_grid_vi1 = _16(semantic_grid_vi0, )
    semantic_grid_ir1 = _16(semantic_grid_ir0, )
    attention = self.attention
    semantic_grid_vi2 = (attention).forward(semantic_grid_vi1, )
    attention3 = self.attention
    semantic_grid_ir2 = (attention3).forward(semantic_grid_ir1, )
    semantic_grid_vi3 = _17(semantic_grid_vi2, h, w, )
    semantic_grid_ir3 = _17(semantic_grid_ir2, h, w, )
    _22 = (semantic_grid_vi3, semantic_grid_ir3)
    return _22
