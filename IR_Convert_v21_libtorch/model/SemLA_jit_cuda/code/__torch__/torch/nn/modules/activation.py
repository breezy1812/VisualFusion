class ReLU(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : NoneType
  inplace : Final[bool] = True
  def forward(self: __torch__.torch.nn.modules.activation.ReLU,
    input: Tensor) -> Tensor:
    _0 = __torch__.torch.nn.functional.relu(input, True, )
    return _0
class Sigmoid(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  def forward(self: __torch__.torch.nn.modules.activation.Sigmoid,
    input: Tensor) -> Tensor:
    return torch.sigmoid(input)
class GELU(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  def forward(self: __torch__.torch.nn.modules.activation.GELU,
    input: Tensor) -> Tensor:
    _1 = __torch__.torch.nn.functional.gelu(input, )
    return _1
