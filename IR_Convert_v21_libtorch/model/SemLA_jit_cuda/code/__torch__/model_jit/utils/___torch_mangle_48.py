class CBR(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  conv : __torch__.torch.nn.modules.conv.___torch_mangle_47.Conv2d
  bn : __torch__.torch.nn.modules.batchnorm.___torch_mangle_15.BatchNorm2d
  relu : __torch__.torch.nn.modules.activation.ReLU
  def forward(self: __torch__.model_jit.utils.___torch_mangle_48.CBR,
    x: Tensor) -> Tensor:
    relu = self.relu
    bn = self.bn
    conv = self.conv
    _0 = (relu).forward((bn).forward((conv).forward(x, ), ), )
    return _0
