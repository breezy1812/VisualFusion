class JConv(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  feat_trans : __torch__.model_jit.utils.___torch_mangle_16.CBR
  dwconv : __torch__.model_jit.utils.___torch_mangle_19.DWConv
  norm : __torch__.torch.nn.modules.batchnorm.___torch_mangle_15.BatchNorm2d
  mlp : __torch__.model_jit.utils.___torch_mangle_22.MLP
  def forward(self: __torch__.model_jit.reg.___torch_mangle_23.JConv,
    x: Tensor) -> Tensor:
    feat_trans = self.feat_trans
    x0 = (feat_trans).forward(x, )
    dwconv = self.dwconv
    x1 = torch.add(x0, (dwconv).forward(x0, ))
    norm = self.norm
    out = (norm).forward(x1, )
    mlp = self.mlp
    return torch.add(x1, (mlp).forward(out, ))
