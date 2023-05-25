import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
import math
import copy

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")

def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0

def c2_xavier_fill(module):
    # Caffe2 implementation of XavierFill in fact
    nn.init.kaiming_uniform_(module.weight, a=1)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)

def with_pos_embed(x, pos):
    return x if pos is None else x + pos

class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=256, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        h, w = not_mask.shape[-2:]
        minlen = min(h, w)
        h_embed = not_mask.cumsum(1, dtype=torch.float32)
        w_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            h_embed = (h_embed - h/2) / (minlen + eps) * self.scale
            w_embed = (w_embed - w/2) / (minlen + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_w = w_embed[:, :, :, None] / dim_t
        pos_h = h_embed[:, :, :, None] / dim_t
        pos_w = torch.stack(
            (pos_w[:, :, :, 0::2].sin(), pos_w[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_h = torch.stack(
            (pos_h[:, :, :, 0::2].sin(), pos_h[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_h, pos_w), dim=3).permute(0, 3, 1, 2)
        return pos
    
    def __repr__(self, _repr_indent=4):
        head = "Positional encoding " + self.__class__.__name__
        body = [
            "num_pos_feats: {}".format(self.num_pos_feats),
            "temperature: {}".format(self.temperature),
            "normalize: {}".format(self.normalize),
            "scale: {}".format(self.scale),
        ]
        # _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)

class Conv2d_Convenience(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)
        self.norm = norm
        self.activation = activation

    def forward(self, x):
        if not torch.jit.is_scripting():
            if x.numel() == 0 and self.training:
                assert not isinstance(
                    self.norm, torch.nn.SyncBatchNorm
                ), "SyncBatchNorm does not support empty inputs!"
        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

