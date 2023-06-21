import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from lib.model_zoo.common.get_model import get_model, register

symbol = 'seecoder'

###########
# helpers #
###########

def with_pos_embed(x, pos):
    return x if pos is None else x + pos

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

def c2_xavier_fill(module):
    # Caffe2 implementation of XavierFill in fact
    nn.init.kaiming_uniform_(module.weight, a=1)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)

def with_pos_embed(x, pos):
    return x if pos is None else x + pos

###########
# Modules #
###########

class Conv2d_Convenience(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)
        self.norm = norm
        self.activation = activation

    def forward(self, x):
        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self,
                 dim=256, 
                 feedforward_dim=1024,
                 dropout=0.1, 
                 activation="relu",
                 n_heads=8,):

        super().__init__()

        self.self_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)

        self.linear1 = nn.Linear(dim, feedforward_dim)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feedforward_dim, dim)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        h = x
        h1 = self.self_attn(x, x, x, attn_mask=None)[0]
        h = h + self.dropout1(h1)
        h = self.norm1(h)

        h2 = self.linear2(self.dropout2(self.activation(self.linear1(h))))
        h = h + self.dropout3(h2)
        h = self.norm2(h)
        return h

class DecoderLayerStacked(nn.Module):
    def __init__(self, layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, x):
        h = x
        for _, layer in enumerate(self.layers):
            h = layer(h)
        if self.norm is not None:
            h = self.norm(h)
        return h

class SelfAttentionLayer(nn.Module):
    def __init__(self, channels, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(channels, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward_post(self, 
                     qkv,
                     qk_pos = None,
                     mask = None,):
        h = qkv
        qk = with_pos_embed(qkv, qk_pos).transpose(0, 1)
        v = qkv.transpose(0, 1)
        h1 = self.self_attn(qk, qk, v, attn_mask=mask)[0]
        h1 = h1.transpose(0, 1)
        h = h + self.dropout(h1)
        h = self.norm(h)
        return h

    def forward_pre(self, tgt,
                    tgt_mask = None,
                    tgt_key_padding_mask = None,
                    query_pos = None):
        # deprecated
        assert False
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, *args, **kwargs):
        if self.normalize_before:
            return self.forward_pre(*args, **kwargs)
        return self.forward_post(*args, **kwargs)

class CrossAttentionLayer(nn.Module):
    def __init__(self, channels, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(channels, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward_post(self, 
                     q, 
                     kv,
                     q_pos = None, 
                     k_pos = None,
                     mask = None,):
        h = q
        q = with_pos_embed(q, q_pos).transpose(0, 1)
        k = with_pos_embed(kv, k_pos).transpose(0, 1)
        v = kv.transpose(0, 1)
        h1 = self.multihead_attn(q, k, v, attn_mask=mask)[0]
        h1 = h1.transpose(0, 1)
        h = h + self.dropout(h1)
        h = self.norm(h)
        return h

    def forward_pre(self, tgt, memory,
                    memory_mask = None,
                    memory_key_padding_mask = None,
                    pos = None,
                    query_pos = None):
        # Deprecated
        assert False
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, *args, **kwargs):
        if self.normalize_before:
            return self.forward_pre(*args, **kwargs)
        return self.forward_post(*args, **kwargs)

class FeedForwardLayer(nn.Module):
    def __init__(self, channels, hidden_channels=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.linear1 = nn.Linear(channels, hidden_channels)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_channels, channels)
        self.norm = nn.LayerNorm(channels)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward_post(self, x):
        h = x
        h1 = self.linear2(self.dropout(self.activation(self.linear1(h))))
        h = h + self.dropout(h1)
        h = self.norm(h)
        return h

    def forward_pre(self, x):
        xn = self.norm(x)
        h = x
        h1 = self.linear2(self.dropout(self.activation(self.linear1(xn))))
        h = h + self.dropout(h1)
        return h

    def forward(self, *args, **kwargs):
        if self.normalize_before:
            return self.forward_pre(*args, **kwargs)
        return self.forward_post(*args, **kwargs)

class MLP(nn.Module):
    def __init__(self, in_channels, channels, out_channels, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [channels] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) 
                for n, k in zip([in_channels]+h, h+[out_channels]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class PPE_MLP(nn.Module):
    def __init__(self, freq_num=20, freq_max=None, out_channel=768, mlp_layer=3):
        import math
        super().__init__()
        self.freq_num = freq_num
        self.freq_max = freq_max
        self.out_channel = out_channel
        self.mlp_layer = mlp_layer
        self.twopi = 2 * math.pi

        mlp = []
        in_channel = freq_num*4
        for idx in range(mlp_layer):
            linear = nn.Linear(in_channel, out_channel, bias=True)
            nn.init.xavier_normal_(linear.weight)
            nn.init.constant_(linear.bias, 0)
            mlp.append(linear)
            if idx != mlp_layer-1:
                mlp.append(nn.SiLU())
            in_channel = out_channel
        self.mlp = nn.Sequential(*mlp)
        nn.init.constant_(self.mlp[-1].weight, 0)

    def forward(self, x, mask=None):
        assert mask is None, "Mask not implemented"
        h, w = x.shape[-2:]
        minlen = min(h, w)

        h_embed, w_embed = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        if self.training:
            import numpy.random as npr
            pertube_h, pertube_w = npr.uniform(-0.5, 0.5), npr.uniform(-0.5, 0.5)
        else:
            pertube_h, pertube_w = 0, 0

        h_embed = (h_embed+0.5 - h/2 + pertube_h) / (minlen) * self.twopi
        w_embed = (w_embed+0.5 - w/2 + pertube_w) / (minlen) * self.twopi
        h_embed, w_embed = h_embed.to(x.device).to(x.dtype), w_embed.to(x.device).to(x.dtype)

        dim_t = torch.linspace(0, 1, self.freq_num, dtype=torch.float32, device=x.device)
        freq_max = self.freq_max if self.freq_max is not None else minlen/2
        dim_t = freq_max ** dim_t.to(x.dtype)

        pos_h = h_embed[:, :, None] * dim_t
        pos_w = w_embed[:, :, None] * dim_t
        pos = torch.cat((pos_h.sin(), pos_h.cos(), pos_w.sin(), pos_w.cos()), dim=-1)
        pos = self.mlp(pos)
        pos = pos.permute(2, 0, 1)[None]
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

###########
# Decoder #
###########

@register('seecoder_decoder')
class Decoder(nn.Module):
    def __init__(
            self,
            inchannels,
            trans_input_tags,
            trans_num_layers,
            trans_dim,
            trans_nheads,
            trans_dropout,
            trans_feedforward_dim,):

        super().__init__()
        trans_inchannels = {
            k: v for k, v in inchannels.items() if k in trans_input_tags}
        fpn_inchannels = {
            k: v for k, v in inchannels.items() if k not in trans_input_tags}

        self.trans_tags = sorted(list(trans_inchannels.keys()))
        self.fpn_tags   = sorted(list(fpn_inchannels.keys()))
        self.all_tags   = sorted(list(inchannels.keys()))

        if len(self.trans_tags)==0: 
            assert False # Not allowed

        self.num_trans_lvls = len(self.trans_tags)

        self.inproj_layers = nn.ModuleDict()
        for tagi in self.trans_tags:
            layeri = nn.Sequential(
                nn.Conv2d(trans_inchannels[tagi], trans_dim, kernel_size=1),
                nn.GroupNorm(32, trans_dim),)
            nn.init.xavier_uniform_(layeri[0].weight, gain=1)
            nn.init.constant_(layeri[0].bias, 0)
            self.inproj_layers[tagi] = layeri

        tlayer = DecoderLayer(
            dim     = trans_dim,
            n_heads = trans_nheads,
            dropout = trans_dropout,
            feedforward_dim = trans_feedforward_dim,
            activation = 'relu',)

        self.transformer = DecoderLayerStacked(tlayer, trans_num_layers)
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.level_embed = nn.Parameter(torch.Tensor(len(self.trans_tags), trans_dim))
        nn.init.normal_(self.level_embed)

        self.lateral_layers = nn.ModuleDict()
        self.output_layers = nn.ModuleDict()
        for tagi in self.all_tags:
            lateral_conv = Conv2d_Convenience(
                inchannels[tagi], trans_dim, kernel_size=1, 
                bias=False, norm=nn.GroupNorm(32, trans_dim))
            c2_xavier_fill(lateral_conv)
            self.lateral_layers[tagi] = lateral_conv

        for tagi in self.fpn_tags:
            output_conv = Conv2d_Convenience(
                trans_dim, trans_dim, kernel_size=3, stride=1, padding=1,
                bias=False, norm=nn.GroupNorm(32, trans_dim), activation=F.relu,)
            c2_xavier_fill(output_conv)
            self.output_layers[tagi] = output_conv

    def forward(self, features):
        x = []
        spatial_shapes = {}
        for idx, tagi in enumerate(self.trans_tags[::-1]):
            xi = features[tagi]
            xi = self.inproj_layers[tagi](xi)
            bs, _, h, w = xi.shape
            spatial_shapes[tagi] = (h, w)
            xi = xi.flatten(2).transpose(1, 2) + self.level_embed[idx].view(1, 1, -1)
            x.append(xi)

        x_length = [xi.shape[1] for xi in x]
        x_concat = torch.cat(x, 1)
        y_concat = self.transformer(x_concat)
        y = torch.split(y_concat, x_length, dim=1)

        out = {}
        for idx, tagi in enumerate(self.trans_tags[::-1]):
            h, w = spatial_shapes[tagi]
            yi = y[idx].transpose(1, 2).view(bs, -1, h, w)
            out[tagi] = yi

        for idx, tagi in enumerate(self.all_tags[::-1]):
            lconv = self.lateral_layers[tagi]
            if tagi in self.trans_tags:
                out[tagi] = out[tagi] + lconv(features[tagi])
                tag_save = tagi
            else:
                oconv = self.output_layers[tagi]
                h = lconv(features[tagi])
                oprev = out[tag_save]
                h = h + F.interpolate(oconv(oprev), size=h.shape[-2:], mode="bilinear", align_corners=False)
                out[tagi] = h

        return out

#####################
# Query Transformer #
#####################

@register('seecoder_query_transformer')
class QueryTransformer(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_dim,
                 num_queries = [8, 144],
                 nheads = 8,
                 num_layers = 9,
                 feedforward_dim = 2048,
                 mask_dim = 256,
                 pre_norm = False,
                 num_feature_levels = 3,
                 enforce_input_project = False, 
                 with_fea2d_pos = True):

        super().__init__()

        if with_fea2d_pos:
            self.pe_layer = PPE_MLP(freq_num=20, freq_max=None, out_channel=hidden_dim, mlp_layer=3)
        else:
            self.pe_layer = None

        if in_channels!=hidden_dim or enforce_input_project:
            self.input_proj = nn.ModuleList()
            for _ in range(num_feature_levels):
                self.input_proj.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1))
                c2_xavier_fill(self.input_proj[-1])
        else:
            self.input_proj = None

        self.num_heads = nheads
        self.num_layers = num_layers
        self.transformer_selfatt_layers = nn.ModuleList()
        self.transformer_crossatt_layers = nn.ModuleList()
        self.transformer_feedforward_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_selfatt_layers.append(
                SelfAttentionLayer(
                    channels=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm, ))

            self.transformer_crossatt_layers.append(
                CrossAttentionLayer(
                    channels=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm, ))

            self.transformer_feedforward_layers.append(
                FeedForwardLayer(
                    channels=hidden_dim,
                    hidden_channels=feedforward_dim,
                    dropout=0.0,
                    normalize_before=pre_norm, ))

        self.num_queries = num_queries
        num_gq, num_lq = self.num_queries
        self.init_query = nn.Embedding(num_gq+num_lq, hidden_dim)
        self.query_pos_embedding = nn.Embedding(num_gq+num_lq, hidden_dim)

        self.num_feature_levels = num_feature_levels
        self.level_embed = nn.Embedding(num_feature_levels, hidden_dim)

    def forward(self, x):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        fea2d = []
        fea2d_pos = []
        size_list = []

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            if self.pe_layer is not None:
                pi = self.pe_layer(x[i], None).flatten(2)
                pi = pi.transpose(1, 2)
            else:
                pi = None
            xi = self.input_proj[i](x[i]) if self.input_proj is not None else x[i]
            xi = xi.flatten(2) + self.level_embed.weight[i][None, :, None]
            xi = xi.transpose(1, 2)
            fea2d.append(xi)
            fea2d_pos.append(pi)

        bs, _, _ = fea2d[0].shape
        num_gq, num_lq = self.num_queries
        gquery = self.init_query.weight[:num_gq].unsqueeze(0).repeat(bs, 1, 1)
        lquery = self.init_query.weight[num_gq:].unsqueeze(0).repeat(bs, 1, 1)

        gquery_pos = self.query_pos_embedding.weight[:num_gq].unsqueeze(0).repeat(bs, 1, 1)
        lquery_pos = self.query_pos_embedding.weight[num_gq:].unsqueeze(0).repeat(bs, 1, 1)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels

            qout = self.transformer_crossatt_layers[i](
                q = lquery, 
                kv = fea2d[level_index],
                q_pos = lquery_pos, 
                k_pos = fea2d_pos[level_index], 
                mask = None,)
            lquery = qout

            qout = self.transformer_selfatt_layers[i](
                qkv = torch.cat([gquery, lquery], dim=1),
                qk_pos = torch.cat([gquery_pos, lquery_pos], dim=1),)
            
            qout = self.transformer_feedforward_layers[i](qout)

            gquery = qout[:, :num_gq]
            lquery = qout[:, num_gq:]

        output = torch.cat([gquery, lquery], dim=1)

        return output

##################
# Main structure #
##################

@register('seecoder')
class SemanticContextEncoder(nn.Module):
    def __init__(self, 
                 imencoder_cfg, 
                 imdecoder_cfg,
                 qtransformer_cfg):
        super().__init__()
        self.imencoder = get_model()(imencoder_cfg)
        self.imdecoder = get_model()(imdecoder_cfg)
        self.qtransformer = get_model()(qtransformer_cfg)

    def forward(self, x):
        fea = self.imencoder(x)
        hs = {'res3' : fea['res3'], 
              'res4' : fea['res4'], 
              'res5' : fea['res5'], }
        hs = self.imdecoder(hs)
        hs = [hs['res3'], hs['res4'], hs['res5']]
        q = self.qtransformer(hs)
        return q

    def encode(self, x):
        return self(x)
