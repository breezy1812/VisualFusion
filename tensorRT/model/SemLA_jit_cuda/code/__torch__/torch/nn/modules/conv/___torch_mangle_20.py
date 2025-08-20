class Conv2d(Module):
  __parameters__ = ["weight", "bias", ]
  __buffers__ = []
  weight : Tensor
  bias : Optional[Tensor]
  training : bool
  _is_full_backward_hook : NoneType
  transposed : bool
  _reversed_padding_repeated_twice : List[int]
  dilation : Final[Tuple[int, int]] = (1, 1)
  stride : Final[Tuple[int, int]] = (1, 1)
  padding : Final[Tuple[int, int]] = (0, 0)
  out_channels : Final[int] = 64
  in_channels : Final[int] = 32
  output_padding : Final[Tuple[int, int]] = (0, 0)
  kernel_size : Final[Tuple[int, int]] = (1, 1)
  groups : Final[int] = 1
  padding_mode : Final[str] = "zeros"
  def forward(self: __torch__.torch.nn.modules.conv.___torch_mangle_20.Conv2d,
    input: Tensor) -> Tensor:
    weight = self.weight
    bias = self.bias
    _0 = (self)._conv_forward(input, weight, bias, )
    return _0
  def _conv_forward(self: __torch__.torch.nn.modules.conv.___torch_mangle_20.Conv2d,
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor]) -> Tensor:
    _1 = torch.conv2d(input, weight, bias, [1, 1], [0, 0], [1, 1])
    return _1
