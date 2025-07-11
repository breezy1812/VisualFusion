class DWConv(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  group_conv3x3 : __torch__.torch.nn.modules.conv.___torch_mangle_53.Conv2d
  norm : __torch__.torch.nn.modules.batchnorm.___torch_mangle_51.BatchNorm2d
  act : __torch__.torch.nn.modules.activation.ReLU
  projection : __torch__.torch.nn.modules.conv.___torch_mangle_54.Conv2d
  def forward(self: __torch__.model_jit.utils.___torch_mangle_55.DWConv,
    x: Tensor) -> Tensor:
    group_conv3x3 = self.group_conv3x3
    out = (group_conv3x3).forward(x, )
    norm = self.norm
    out0 = (norm).forward(out, )
    act = self.act
    out1 = (act).forward(out0, )
    projection = self.projection
    return (projection).forward(out1, )
