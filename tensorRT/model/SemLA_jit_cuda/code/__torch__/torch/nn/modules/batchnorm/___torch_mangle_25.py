class BatchNorm2d(Module):
  __parameters__ = ["weight", "bias", ]
  __buffers__ = ["running_mean", "running_var", "num_batches_tracked", ]
  weight : Tensor
  bias : Tensor
  running_mean : Tensor
  running_var : Tensor
  num_batches_tracked : Tensor
  training : bool
  _is_full_backward_hook : NoneType
  eps : Final[float] = 1.0000000000000001e-05
  track_running_stats : Final[bool] = True
  momentum : Final[float] = 0.10000000000000001
  affine : Final[bool] = True
  num_features : Final[int] = 256
  def forward(self: __torch__.torch.nn.modules.batchnorm.___torch_mangle_25.BatchNorm2d,
    input: Tensor) -> Tensor:
    _0 = __torch__.torch.nn.functional.batch_norm
    _1 = (self)._check_input_dim(input, )
    training = self.training
    if training:
      num_batches_tracked = self.num_batches_tracked
      self.num_batches_tracked = torch.add(num_batches_tracked, 1)
    else:
      pass
    training0 = self.training
    if training0:
      bn_training = True
    else:
      bn_training = False
    running_mean = self.running_mean
    running_var = self.running_var
    weight = self.weight
    bias = self.bias
    _2 = _0(input, running_mean, running_var, weight, bias, bn_training, 0.10000000000000001, 1.0000000000000001e-05, )
    return _2
  def _check_input_dim(self: __torch__.torch.nn.modules.batchnorm.___torch_mangle_25.BatchNorm2d,
    input: Tensor) -> NoneType:
    if torch.ne(torch.dim(input), 4):
      _3 = torch.format("expected 4D input (got {}D input)", torch.dim(input))
      ops.prim.RaiseException(_3)
    else:
      pass
    return None
