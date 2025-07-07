def batch_norm(input: Tensor,
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
    weight: Optional[Tensor]=None,
    bias: Optional[Tensor]=None,
    training: bool=False,
    momentum: float=0.10000000000000001,
    eps: float=1.0000000000000001e-05) -> Tensor:
  _0 = __torch__.torch.nn.functional._verify_batch_size
  if training:
    _1 = _0(torch.size(input), )
  else:
    pass
  _2 = torch.batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps, True)
  return _2
def relu(input: Tensor,
    inplace: bool=False) -> Tensor:
  if inplace:
    result = torch.relu_(input)
  else:
    result = torch.relu(input)
  return result
def linear(input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor]=None) -> Tensor:
  return torch.linear(input, weight, bias)
def layer_norm(input: Tensor,
    normalized_shape: List[int],
    weight: Optional[Tensor]=None,
    bias: Optional[Tensor]=None,
    eps: float=1.0000000000000001e-05) -> Tensor:
  _3 = torch.layer_norm(input, normalized_shape, weight, bias, eps)
  return _3
def gelu(input: Tensor) -> Tensor:
  return torch.gelu(input)
def adaptive_avg_pool2d(input: Tensor,
    output_size: List[int]) -> Tensor:
  _4 = torch.gt(torch.len(torch.size(input)), torch.len(output_size))
  if _4:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  _5 = torch.adaptive_avg_pool2d(input, output_size)
  return _5
def adaptive_avg_pool3d(input: Tensor,
    output_size: List[int]) -> Tensor:
  _6 = torch.gt(torch.len(torch.size(input)), torch.len(output_size))
  if _6:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  _7 = torch.adaptive_avg_pool3d(input, output_size)
  return _7
def _verify_batch_size(size: List[int]) -> NoneType:
  _8 = "Expected more than 1 value per channel when training, got input size {}"
  size_prods = size[0]
  size_prods0 = size_prods
  for i in range(torch.sub(torch.len(size), 2)):
    size_prods1 = torch.mul(size_prods0, size[torch.add(i, 2)])
    size_prods0 = size_prods1
  if torch.eq(size_prods0, 1):
    ops.prim.RaiseException(torch.format(_8, size))
  else:
    pass
  return None
def elu(input: Tensor,
    alpha: float=1.,
    inplace: bool=False) -> Tensor:
  if inplace:
    result = torch.elu_(input, alpha)
  else:
    result = torch.elu(input, alpha)
  return result
