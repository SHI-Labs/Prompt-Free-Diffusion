import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .msdeformattn import PositionEmbeddingSine, _get_clones, _get_activation_fn
from lib.model_zoo.common.get_model import get_model, register

##########
# helper #
##########

def with_pos_embed(x, pos):
    return x if pos is None else x + pos

##############
# One Former #
##############

class Transformer(nn.Module):
    def __init__(self,
                 d_model=512,
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False,
                 return_intermediate_dec=False,):

        super().__init__()
        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed, task_token=None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        if mask is not None:
            mask = mask.flatten(1)
            
        if task_token is None:
            tgt = torch.zeros_like(query_embed)
        else:
            tgt = task_token.repeat(query_embed.shape[0], 1, 1)
   
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed) # src = memory
        hs = self.decoder(
            tgt, memory, memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed
        )
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None, pos=None,):
        output = src
        for layer in self.layers:
            output = layer(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos
            )
        if self.norm is not None:
            output = self.norm(output)
        return output

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
            self,
            tgt,
            memory,
            tgt_mask=None,
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None,
            pos=None,
            query_pos=None,):

        output = tgt
        intermediate = []
        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)

class TransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            d_model,
            nhead,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
            normalize_before=False, ):

        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, x, pos):
        return x if pos is None else x + pos

    def forward_post(
            self,
            src,
            src_mask = None,
            src_key_padding_mask = None,
            pos = None,):

        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(
            self,
            src,
            src_mask = None,
            src_key_padding_mask = None,
            pos = None,):

        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
            self,
            src,
            src_mask = None,
            src_key_padding_mask = None,
            pos = None,):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class TransformerDecoderLayer(nn.Module):
    def __init__(
            self,
            d_model,
            nhead,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
            normalize_before=False,):

        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, x, pos):
        return x if pos is None else x + pos

    def forward_post(
            self,
            tgt,
            memory,
            tgt_mask = None,
            memory_mask = None,
            tgt_key_padding_mask = None,
            memory_key_padding_mask = None,
            pos = None,
            query_pos = None,):

        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
            self,
            tgt,
            memory,
            tgt_mask = None,
            memory_mask = None,
            tgt_key_padding_mask = None,
            memory_key_padding_mask = None,
            pos = None,
            query_pos = None,):

        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
            self,
            tgt,
            memory,
            tgt_mask = None,
            memory_mask = None,
            tgt_key_padding_mask = None,
            memory_key_padding_mask = None,
            pos = None,
            query_pos = None, ):

        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
                query_pos,)
        return self.forward_post(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,)

class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask = None,
                     tgt_key_padding_mask = None,
                     query_pos = None):
        q = k = self.with_pos_embed(tgt, query_pos).transpose(0 ,1)
        tgt2 = self.self_attn(q, k, value=tgt.transpose(0 ,1), attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2.transpose(0 ,1))
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask = None,
                    tgt_key_padding_mask = None,
                    query_pos = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        
        return tgt

    def forward(self, tgt,
                tgt_mask = None,
                tgt_key_padding_mask = None,
                query_pos = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)

class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask = None,
                     memory_key_padding_mask = None,
                     pos = None,
                     query_pos = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos).transpose(0, 1),
                                   key=self.with_pos_embed(memory, pos).transpose(0, 1),
                                   value=memory.transpose(0, 1), attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2.transpose(0, 1))
        tgt = self.norm(tgt)
        
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask = None,
                    memory_key_padding_mask = None,
                    pos = None,
                    query_pos = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask = None,
                memory_key_padding_mask = None,
                pos = None,
                query_pos = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)

class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

@register('seet_oneformer_tdecoder')
class Seet_OneFormer_TDecoder(nn.Module):
    def __init__(
            self,
            in_channels,
            mask_classification,
            num_classes,
            hidden_dim,
            num_queries,
            nheads,
            dropout,
            dim_feedforward,
            enc_layers,
            is_train,
            dec_layers,
            class_dec_layers,
            pre_norm,
            mask_dim,
            enforce_input_project,
            use_task_norm,):

        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification
        self.is_train = is_train
        self.use_task_norm = use_task_norm

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        self.class_transformer = Transformer(
            d_model=hidden_dim,
            dropout=dropout,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=enc_layers,
            num_decoder_layers=class_dec_layers,
            normalize_before=pre_norm,
            return_intermediate_dec=False,
        )

        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())
        
        self.class_input_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        weight_init.c2_xavier_fill(self.class_input_proj)

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    def forward(self, x, mask_features, tasks):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])
            pos[-1] = pos[-1].transpose(1, 2)
            src[-1] = src[-1].transpose(1, 2)

        bs, _, _ = src[0].shape

        query_embed = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)

        tasks = tasks.unsqueeze(0)
        if self.use_task_norm:
            tasks = self.decoder_norm(tasks)
        
        feats = self.pe_layer(mask_features, None)

        out_t, _ = self.class_transformer(
            feats, None, 
            self.query_embed.weight[:-1], 
            self.class_input_proj(mask_features),
            tasks if self.use_task_norm else None)
        out_t = out_t[0]
        
        out = torch.cat([out_t, tasks], dim=1)

        output = out.clone()

        predictions_class = []
        predictions_mask = []

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
            output, mask_features, attn_mask_target_size=size_list[0])
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,
                pos=pos[level_index], query_pos=query_embed, )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed, )
            
            # FFN
            output = self.transformer_ffn_layers[i](output)

            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
                output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            
        assert len(predictions_class) == self.num_layers + 1

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],}

        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask
