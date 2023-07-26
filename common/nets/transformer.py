# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
from base64 import encode
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import math
from einops import repeat
class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.smpl_tgt = nn.Parameter(torch.zeros(256))
        self.kp_tgt = nn.Parameter(torch.zeros(256))

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed, dense_feat, exponent):
        # L, B, C
        tgt = torch.zeros_like(query_embed)
        L, B, C = query_embed.shape
        smpl_tgt = repeat(self.smpl_tgt, 'c -> n b c',n=3, b=B)
        kp_tgt = repeat(self.kp_tgt, 'c ->n b c',n=15, b=B)
        tgt = torch.cat([smpl_tgt, kp_tgt], dim=0)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs, joint_img = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed, dense_feat=dense_feat, exponent=exponent)
        # hs [layers, length, batch, channels], memory [length, batch, channels]
        return hs, joint_img


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

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
        self.joint_img_regress = nn.ModuleList(
            [nn.Linear(256, 3)] + \
            [nn.Linear(256 + 3, 3) for _ in range(num_layers-1)]
        )
        # self.fc_pos_3d = nn.Linear(3, 256)
        self.fc_pos_3d = nn.Sequential(
            nn.Linear(3, 256), 
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256)
        )

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                dense_feat: Optional[Tensor] = None,
                exponent: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for idx, layer in enumerate(self.layers):
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.norm is not None:
                output = self.norm(output)
            joint_img_hs = output[3:].transpose(0, 1).contiguous()
            intermediate.append(output[:3])
            if idx == 0:
                joint_img = self.joint_img_regress[idx](joint_img_hs).sigmoid()
            else:
                inverse_sigmoid_joint_img = inverse_sigmoid(joint_img)
                offset = self.joint_img_regress[idx](torch.cat([joint_img_hs, inverse_sigmoid_joint_img], dim=-1))
                joint_img = torch.sigmoid(inverse_sigmoid_joint_img + offset)
            # sample_grid = joint_img.mul(2).add(-1).unsqueeze(1).unsqueeze(1)
            xy_sample_grid = joint_img[...,:2].mul(2).add(-1)
            z_sample_grid = torch.pow(joint_img[...,-1:], exponent).mul(2).add(-1)
            sample_grid = torch.cat([xy_sample_grid, z_sample_grid], dim=-1).unsqueeze(1).unsqueeze(1)
            sample_feat = F.grid_sample(dense_feat, sample_grid, mode='bilinear', align_corners=True).squeeze(2).squeeze(2).permute(2, 0, 1).contiguous() # N, B, C
            sample_feat_pos = self.fc_pos_3d(inverse_sigmoid(joint_img)).transpose(0, 1)
            if idx == 0:
                memory = torch.cat([memory, sample_feat], dim=0)
                pos = torch.cat([pos, sample_feat_pos], dim=0)
            else:
                memory = torch.cat([memory[:-15], sample_feat], dim=0)
                pos = torch.cat([pos[:-15], sample_feat_pos], dim=0)

        return torch.stack(intermediate), joint_img


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
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

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
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

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        not_mask = torch.ones_like(x[:,0]).bool()
        # x_embed = torch.arange(x.size(2), device=x.device, dtype=torch.float32)
        # y_embed = torch.arange(x.size(3), device=x.device, dtype=torch.float32)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

def bulid_transformer_decoder(
        d_model=256,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        num_decoder_layers=6,
        normalize_before=False,
        return_intermediate_dec=False,
    ):
    decoder_layer = TransformerDecoderLayer(
        d_model, 
        nhead, 
        dim_feedforward,
        dropout, 
        activation, 
        normalize_before
    )
    decoder_norm = nn.LayerNorm(d_model)
    decoder = TransformerDecoder(
        decoder_layer, 
        num_decoder_layers, 
        decoder_norm,
        return_intermediate=return_intermediate_dec
    )
    return decoder

def bulid_transformer_encoder(
        d_model=256,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        num_encoder_layers=6,
        normalize_before=False,
    ):
    encoder_layer = TransformerEncoderLayer(
        d_model, 
        nhead, 
        dim_feedforward,
        dropout, 
        activation, 
        normalize_before
    )
    encoder_norm = nn.LayerNorm(d_model)
    encoder = TransformerEncoder(
        encoder_layer, 
        num_encoder_layers, 
        encoder_norm
    )
    return encoder

def bulid_transformer(
        d_model=256, 
        nhead=8, 
        num_encoder_layers=6,
        num_decoder_layers=6, 
        dim_feedforward=2048, 
        dropout=0.1,
        activation="relu", 
        normalize_before=False,
        return_intermediate_dec=True
    ):
    return Transformer(
        d_model=d_model, 
        nhead=nhead, 
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers, 
        dim_feedforward=dim_feedforward, 
        dropout=dropout,
        activation=activation, 
        normalize_before=normalize_before,
        return_intermediate_dec=return_intermediate_dec
    )


if __name__ == '__main__':
    net = bulid_transformer().cuda()
    src = torch.randn(10, 2, 256).cuda()
    mask = torch.zeros(2,10).bool().cuda()
    query_embed = torch.randn(12, 2, 256).cuda()
    pos_embed = torch.randn(10, 2, 256).cuda()
    
    hs, mem = net(src, mask, query_embed, pos_embed)
    print(hs.shape, mem.shape)
    