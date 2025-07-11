class DWT_2D(Module):
  __parameters__ = []
  __buffers__ = ["w_ll", "w_lh", "w_hl", "w_hh", ]
  w_ll : Tensor
  w_lh : Tensor
  w_hl : Tensor
  w_hh : Tensor
  training : bool
  _is_full_backward_hook : Optional[bool]
  def forward(self: __torch__.model_jit.utils.DWT_2D,
    x: Tensor) -> Tensor:
    x0 = torch.contiguous(x)
    dim = (torch.size(x0))[1]
    w_ll = self.w_ll
    _0 = torch.expand(w_ll, [dim, -1, -1, -1])
    x_ll = torch.conv2d(x0, _0, None, [2, 2], [0, 0], [1, 1], dim)
    w_lh = self.w_lh
    _1 = torch.expand(w_lh, [dim, -1, -1, -1])
    x_lh = torch.conv2d(x0, _1, None, [2, 2], [0, 0], [1, 1], dim)
    w_hl = self.w_hl
    _2 = torch.expand(w_hl, [dim, -1, -1, -1])
    x_hl = torch.conv2d(x0, _2, None, [2, 2], [0, 0], [1, 1], dim)
    w_hh = self.w_hh
    _3 = torch.expand(w_hh, [dim, -1, -1, -1])
    x_hh = torch.conv2d(x0, _3, None, [2, 2], [0, 0], [1, 1], dim)
    x1 = torch.cat([x_ll, x_lh, x_hl, x_hh], 1)
    return x1
class CBR(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  conv : __torch__.torch.nn.modules.conv.Conv2d
  bn : __torch__.torch.nn.modules.batchnorm.BatchNorm2d
  relu : __torch__.torch.nn.modules.activation.ReLU
  def forward(self: __torch__.model_jit.utils.CBR,
    x: Tensor) -> Tensor:
    relu = self.relu
    bn = self.bn
    conv = self.conv
    _4 = (relu).forward((bn).forward((conv).forward(x, ), ), )
    return _4
class DWConv(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  group_conv3x3 : __torch__.torch.nn.modules.conv.___torch_mangle_0.Conv2d
  norm : __torch__.torch.nn.modules.batchnorm.BatchNorm2d
  act : __torch__.torch.nn.modules.activation.ReLU
  projection : __torch__.torch.nn.modules.conv.___torch_mangle_1.Conv2d
  def forward(self: __torch__.model_jit.utils.DWConv,
    x: Tensor) -> Tensor:
    group_conv3x3 = self.group_conv3x3
    out = (group_conv3x3).forward(x, )
    norm = self.norm
    out0 = (norm).forward(out, )
    act = self.act
    out1 = (act).forward(out0, )
    projection = self.projection
    return (projection).forward(out1, )
class MLP(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  conv1 : __torch__.torch.nn.modules.conv.___torch_mangle_2.Conv2d
  act : __torch__.torch.nn.modules.activation.ReLU
  conv2 : __torch__.torch.nn.modules.conv.___torch_mangle_3.Conv2d
  def forward(self: __torch__.model_jit.utils.MLP,
    x: Tensor) -> Tensor:
    conv1 = self.conv1
    x2 = (conv1).forward(x, )
    act = self.act
    x3 = (act).forward(x2, )
    conv2 = self.conv2
    return (conv2).forward(x3, )
class MLP2(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  fc : __torch__.torch.nn.modules.container.___torch_mangle_63.Sequential
  def forward(self: __torch__.model_jit.utils.MLP2,
    x: Tensor) -> Tensor:
    fc = self.fc
    return (fc).forward(x, )
class StructureAttention(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  dim : int
  nhead : int
  q_proj : __torch__.torch.nn.modules.linear.Linear
  k_proj : __torch__.torch.nn.modules.linear.Linear
  v_proj : __torch__.torch.nn.modules.linear.Linear
  attention : __torch__.model_jit.utils.LinearAttention
  merge : __torch__.torch.nn.modules.linear.Linear
  mlp : __torch__.torch.nn.modules.container.___torch_mangle_68.Sequential
  norm1 : __torch__.torch.nn.modules.normalization.LayerNorm
  norm2 : __torch__.torch.nn.modules.normalization.LayerNorm
  def forward(self: __torch__.model_jit.utils.StructureAttention,
    x: Tensor) -> Tensor:
    bs = torch.size(x, 0)
    q_proj = self.q_proj
    _5 = (q_proj).forward(x, )
    nhead = self.nhead
    dim = self.dim
    query = torch.view(_5, [bs, -1, nhead, dim])
    k_proj = self.k_proj
    _6 = (k_proj).forward(x, )
    nhead0 = self.nhead
    dim0 = self.dim
    key = torch.view(_6, [bs, -1, nhead0, dim0])
    v_proj = self.v_proj
    _7 = (v_proj).forward(x, )
    nhead1 = self.nhead
    dim1 = self.dim
    value = torch.view(_7, [bs, -1, nhead1, dim1])
    attention = self.attention
    message = (attention).forward(query, key, value, )
    merge = self.merge
    nhead2 = self.nhead
    dim2 = self.dim
    _8 = torch.view(message, [bs, -1, torch.mul(nhead2, dim2)])
    message0 = (merge).forward(_8, )
    norm1 = self.norm1
    message1 = (norm1).forward(message0, )
    mlp = self.mlp
    message2 = (mlp).forward(torch.cat([x, message1], 2), )
    norm2 = self.norm2
    message3 = (norm2).forward(message2, )
    return torch.add(x, message3)
class LinearAttention(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  eps : float
  fp : int
  def forward(self: __torch__.model_jit.utils.LinearAttention,
    queries: Tensor,
    keys: Tensor,
    values: Tensor) -> Tensor:
    _9 = __torch__.model_jit.utils.elu_feature_map
    _10 = _9(queries, )
    fp = self.fp
    Q = torch.to(_10, fp)
    _11 = _9(keys, )
    fp0 = self.fp
    K = torch.to(_11, fp0)
    v_length = torch.size(values, 1)
    values0 = torch.div(values, v_length)
    KV = torch.einsum("nshd,nshv->nhdv", [K, values0])
    _12 = torch.einsum("nlhd,nhd->nlh", [Q, torch.sum(K, [1])])
    eps = self.eps
    _13 = torch.reciprocal(torch.add(_12, eps))
    Z = torch.mul(_13, 1)
    fp1 = self.fp
    Z0 = torch.to(Z, fp1)
    fp2 = self.fp
    KV0 = torch.to(KV, fp2)
    _14 = torch.einsum("nlhd,nhdv,nlh->nlhv", [Q, KV0, Z0])
    queried_values = torch.mul(_14, v_length)
    return torch.contiguous(queried_values)
def n_c_h_w_2_n_hw_c(tensor: Tensor) -> Tensor:
  tensor0 = torch.permute(tensor, [0, 2, 3, 1])
  _15 = torch.contiguous(tensor0)
  _16 = [torch.size(tensor0, 0), -1, torch.size(tensor0, -1)]
  return torch.view(_15, _16)
def n_c_h_w_2_n_c_hw(tensor: Tensor) -> Tensor:
  _17 = [torch.size(tensor, 0), torch.size(tensor, 1), -1]
  return torch.view(tensor, _17)
def n_hw_c_2_n_c_h_w(tensor: Tensor,
    h: int,
    w: int) -> Tensor:
  n, _18, c, = torch.size(tensor)
  tensor1 = torch.view(tensor, [n, h, w, c])
  return torch.permute(tensor1, [0, 3, 1, 2])
def n_h_w_c_2_n_c_h_w(tensor: Tensor) -> Tensor:
  return torch.permute(tensor, [0, 3, 1, 2])
def elu_feature_map(x: Tensor) -> Tensor:
  _19 = __torch__.torch.nn.functional.elu(x, 1., False, )
  return torch.add(_19, 1)
