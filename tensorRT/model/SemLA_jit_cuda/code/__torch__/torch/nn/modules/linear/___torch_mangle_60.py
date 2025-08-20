class Linear(Module):
  __parameters__ = ["weight", "bias", ]
  __buffers__ = []
  weight : Tensor
  bias : Tensor
  training : bool
  _is_full_backward_hook : NoneType
  in_features : Final[int] = 256
  out_features : Final[int] = 256
  def forward(self: __torch__.torch.nn.modules.linear.___torch_mangle_60.Linear,
    input: Tensor) -> Tensor:
    _0 = __torch__.torch.nn.functional.linear
    weight = self.weight
    bias = self.bias
    return _0(input, weight, bias, )
