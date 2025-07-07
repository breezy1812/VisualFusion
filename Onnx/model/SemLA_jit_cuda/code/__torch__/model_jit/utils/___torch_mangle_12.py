class MLP(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  conv1 : __torch__.torch.nn.modules.conv.___torch_mangle_10.Conv2d
  act : __torch__.torch.nn.modules.activation.ReLU
  conv2 : __torch__.torch.nn.modules.conv.___torch_mangle_11.Conv2d
  def forward(self: __torch__.model_jit.utils.___torch_mangle_12.MLP,
    x: Tensor) -> Tensor:
    conv1 = self.conv1
    x0 = (conv1).forward(x, )
    act = self.act
    x1 = (act).forward(x0, )
    conv2 = self.conv2
    return (conv2).forward(x1, )
