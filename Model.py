from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import math
from torch.nn import Dropout, Softmax, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
import torch.nn as nn
import torch
import torch.nn.functional as F
import ml_collections
from einops import rearrange
import numbers


def get_CTranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 480  
    config.transformer.num_heads = 4
    config.transformer.num_layers = 4
    config.patch_sizes = [16, 8, 4, 2]
    config.base_channel = 32 
    config.n_classes = 1
    return config


class Channel_Embeddings(nn.Module):
    def __init__(self, config, patchsize, img_size, in_channels):
        super().__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patchsize)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])  # 14 * 14 = 196

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=in_channels,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, in_channels))
        self.dropout = Dropout(config.transformer["embeddings_dropout_rate"])

    def forward(self, x):
        if x is None:
            return None
        x = self.patch_embeddings(x)
        return x


class Reconstruct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Reconstruct, self).__init__()
        if kernel_size == 3:
            padding = 1
        else:
            padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    # def forward(self, x, h, w):
    def forward(self, x):
        if x is None:
            return None

        x = nn.Upsample(scale_factor=self.scale_factor, mode='bilinear')(x)

        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out




class CDConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, groups=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride,
            padding=dilation, dilation=dilation,
            groups=groups, bias=bias
        )

    def forward(self, x):
        w = self.conv.weight
        w_c = w.sum(dim=(2, 3), keepdim=True)
        yc = F.conv2d(
            x, w_c, bias=None,
            stride=self.conv.stride, padding=0,
            dilation=1, groups=self.conv.groups
        )
        y = self.conv(x)
        return y - yc


class Attention_diffv2(nn.Module):
    def __init__(self, config, vis, channel_num):
        super().__init__()
        self.vis = vis
        self.KV_size = config.KV_size
        self.channel_num = channel_num
        self.num_attention_heads = 1

        self.psi = nn.InstanceNorm2d(self.num_attention_heads)
        self.softmax = Softmax(dim=3)

        # Q path, output channels doubled for DiffV2
        self.mhead1 = nn.Conv2d(channel_num[0], channel_num[0] * 2 * self.num_attention_heads, 1, bias=False)
        self.mhead2 = nn.Conv2d(channel_num[1], channel_num[1] * 2 * self.num_attention_heads, 1, bias=False)
        self.mhead3 = nn.Conv2d(channel_num[2], channel_num[2] * 2 * self.num_attention_heads, 1, bias=False)
        self.mhead4 = nn.Conv2d(channel_num[3], channel_num[3] * 2 * self.num_attention_heads, 1, bias=False)

        self.mheadk = nn.Conv2d(self.KV_size, self.KV_size * self.num_attention_heads, 1, bias=False)
        self.mheadv = nn.Conv2d(self.KV_size, self.KV_size * self.num_attention_heads, 1, bias=False)

        C1 = channel_num[0] * self.num_attention_heads
        C2 = channel_num[1] * self.num_attention_heads
        C3 = channel_num[2] * self.num_attention_heads
        C4 = channel_num[3] * self.num_attention_heads

        # scale 1,2: qa uses learnable CDConv (depthwise)
        self.q1a = CDConv2d(C1, C1, stride=1, dilation=1, groups=C1, bias=False)
        self.q2a = CDConv2d(C2, C2, stride=1, dilation=1, groups=C2, bias=False)

        # scale 3,4: same as scale 1,2 (qa CDConv, qb fixed mean depthwise)
        self.q3a = CDConv2d(C3, C3, stride=1, dilation=1, groups=C3, bias=False)
        self.q4a = CDConv2d(C4, C4, stride=1, dilation=1, groups=C4, bias=False)

        # fixed depthwise mean 3x3 for qb on all scales
        mean3 = torch.ones(3, 3, dtype=torch.float32) / 9.0
        self.register_buffer("mean_w1", mean3.view(1, 1, 3, 3).repeat(C1, 1, 1, 1))
        self.register_buffer("mean_w2", mean3.view(1, 1, 3, 3).repeat(C2, 1, 1, 1))
        self.register_buffer("mean_w3", mean3.view(1, 1, 3, 3).repeat(C3, 1, 1, 1))
        self.register_buffer("mean_w4", mean3.view(1, 1, 3, 3).repeat(C4, 1, 1, 1))

        self.k = nn.Conv2d(
            self.KV_size * self.num_attention_heads,
            self.KV_size * self.num_attention_heads,
            kernel_size=3, stride=1, padding=1,
            groups=self.KV_size * self.num_attention_heads,
            bias=False
        )
        self.v = nn.Conv2d(
            self.KV_size * self.num_attention_heads,
            self.KV_size * self.num_attention_heads,
            kernel_size=3, stride=1, padding=1,
            groups=self.KV_size * self.num_attention_heads,
            bias=False
        )

        self.project_out1 = nn.Conv2d(channel_num[0], channel_num[0], 1, bias=False)
        self.project_out2 = nn.Conv2d(channel_num[1], channel_num[1], 1, bias=False)
        self.project_out3 = nn.Conv2d(channel_num[2], channel_num[2], 1, bias=False)
        self.project_out4 = nn.Conv2d(channel_num[3], channel_num[3], 1, bias=False)

        self.gate1 = nn.Conv2d(channel_num[0], channel_num[0], 1, bias=True)
        self.gate2 = nn.Conv2d(channel_num[1], channel_num[1], 1, bias=True)
        self.gate3 = nn.Conv2d(channel_num[2], channel_num[2], 1, bias=True)
        self.gate4 = nn.Conv2d(channel_num[3], channel_num[3], 1, bias=True)

        self.out_norm1 = LayerNorm3d(channel_num[0], LayerNorm_type='BiasFree')
        self.out_norm2 = LayerNorm3d(channel_num[1], LayerNorm_type='BiasFree')
        self.out_norm3 = LayerNorm3d(channel_num[2], LayerNorm_type='BiasFree')
        self.out_norm4 = LayerNorm3d(channel_num[3], LayerNorm_type='BiasFree')

        self.pool = nn.AdaptiveAvgPool2d(1)

        nn.init.constant_(self.gate1.bias, -2.0)
        nn.init.constant_(self.gate2.bias, -2.0)
        nn.init.constant_(self.gate3.bias, -2.0)
        nn.init.constant_(self.gate4.bias, -2.0)

    def _diff_attn_one(self, emb_q, emb_for_gate, k, v, C, h, w, gate_conv, out_norm):
        q = rearrange(emb_q, 'b (head c2) h w -> b head c2 (h w)', head=self.num_attention_heads)
        qa, qb = q.chunk(2, dim=2)

        qa = F.normalize(qa, dim=-1)
        qb = F.normalize(qb, dim=-1)
        k = F.normalize(k, dim=-1)

        attna = (qa @ k.transpose(-2, -1)) / math.sqrt(self.KV_size)
        attnb = (qb @ k.transpose(-2, -1)) / math.sqrt(self.KV_size)

        pa = self.softmax(self.psi(attna))
        pb = self.softmax(self.psi(attnb))

        outa = (pa @ v).mean(dim=1)
        outb = (pb @ v).mean(dim=1)

        g = torch.sigmoid(gate_conv(self.pool(emb_for_gate))).view(emb_for_gate.size(0), C, 1)

        out = outa - outb * g
        out = rearrange(out, 'b c (h w) -> b c h w', h=h, w=w)
        out = out_norm(out)
        return out

    def forward(self, emb1, emb2, emb3, emb4, emb_all):
        b, c1, h, w = emb1.shape

        # scale 1: qa CDConv, qb fixed mean
        q1_2c = self.mhead1(emb1)
        q1a, q1b = q1_2c.chunk(2, dim=1)
        q1a = self.q1a(q1a)
        w1 = self.mean_w1.to(dtype=q1b.dtype, device=q1b.device)
        q1b = F.conv2d(q1b, w1, bias=None, stride=1, padding=1, groups=w1.size(0))
        q1 = torch.cat([q1a, q1b], dim=1)

        # scale 2: qa CDConv, qb fixed mean
        q2_2c = self.mhead2(emb2)
        q2a, q2b = q2_2c.chunk(2, dim=1)
        q2a = self.q2a(q2a)
        w2 = self.mean_w2.to(dtype=q2b.dtype, device=q2b.device)
        q2b = F.conv2d(q2b, w2, bias=None, stride=1, padding=1, groups=w2.size(0))
        q2 = torch.cat([q2a, q2b], dim=1)

        # scale 3: qa CDConv, qb fixed mean
        q3_2c = self.mhead3(emb3)
        q3a, q3b = q3_2c.chunk(2, dim=1)
        q3a = self.q3a(q3a)
        w3 = self.mean_w3.to(dtype=q3b.dtype, device=q3b.device)
        q3b = F.conv2d(q3b, w3, bias=None, stride=1, padding=1, groups=w3.size(0))
        q3 = torch.cat([q3a, q3b], dim=1)

        # scale 4: qa CDConv, qb fixed mean
        q4_2c = self.mhead4(emb4)
        q4a, q4b = q4_2c.chunk(2, dim=1)
        q4a = self.q4a(q4a)
        w4 = self.mean_w4.to(dtype=q4b.dtype, device=q4b.device)
        q4b = F.conv2d(q4b, w4, bias=None, stride=1, padding=1, groups=w4.size(0))
        q4 = torch.cat([q4a, q4b], dim=1)


        k = self.k(self.mheadk(emb_all))
        v = self.v(self.mheadv(emb_all))

        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)

        O1 = self._diff_attn_one(q1, emb1, k, v, self.channel_num[0], h, w, self.gate1, self.out_norm1)
        O2 = self._diff_attn_one(q2, emb2, k, v, self.channel_num[1], h, w, self.gate2, self.out_norm2)
        O3 = self._diff_attn_one(q3, emb3, k, v, self.channel_num[2], h, w, self.gate3, self.out_norm3)
        O4 = self._diff_attn_one(q4, emb4, k, v, self.channel_num[3], h, w, self.gate4, self.out_norm4)

        O1 = self.project_out1(O1)
        O2 = self.project_out2(O2)
        O3 = self.project_out3(O3)
        O4 = self.project_out4(O4)

        return O1, O2, O3, O4, None

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm3d(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm3d, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class Block_ViT(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(Block_ViT, self).__init__()
        self.attn_norm1 = LayerNorm3d(channel_num[0], LayerNorm_type='WithBias')
        self.attn_norm2 = LayerNorm3d(channel_num[1], LayerNorm_type='WithBias')
        self.attn_norm3 = LayerNorm3d(channel_num[2], LayerNorm_type='WithBias')
        self.attn_norm4 = LayerNorm3d(channel_num[3], LayerNorm_type='WithBias')
        self.attn_norm = LayerNorm3d(config.KV_size, LayerNorm_type='WithBias')
 
        self.channel_attn = Attention_diffv2(config, vis, channel_num)  #Attention_org 


    def forward(self, emb1, emb2, emb3, emb4):
        embcat = []
        org1 = emb1
        org2 = emb2
        org3 = emb3
        org4 = emb4
        for i in range(4):
            var_name = "emb" + str(i + 1)
            tmp_var = locals()[var_name]
            if tmp_var is not None:
                embcat.append(tmp_var)
        emb_all = torch.cat(embcat, dim=1)
        cx1 = self.attn_norm1(emb1) if emb1 is not None else None
        cx2 = self.attn_norm2(emb2) if emb2 is not None else None
        cx3 = self.attn_norm3(emb3) if emb3 is not None else None
        cx4 = self.attn_norm4(emb4) if emb4 is not None else None
        emb_all = self.attn_norm(emb_all)  # 1 196 960
        cx1, cx2, cx3, cx4, weights = self.channel_attn(cx1, cx2, cx3, cx4, emb_all)
        cx1 = org1 + cx1 if emb1 is not None else None
        cx2 = org2 + cx2 if emb2 is not None else None
        cx3 = org3 + cx3 if emb3 is not None else None
        cx4 = org4 + cx4 if emb4 is not None else None

        return cx1, cx2, cx3, cx4, weights


class Encoder(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm1 = LayerNorm3d(channel_num[0], LayerNorm_type='WithBias')
        self.encoder_norm2 = LayerNorm3d(channel_num[1], LayerNorm_type='WithBias')
        self.encoder_norm3 = LayerNorm3d(channel_num[2], LayerNorm_type='WithBias')
        self.encoder_norm4 = LayerNorm3d(channel_num[3], LayerNorm_type='WithBias')
        for _ in range(config.transformer["num_layers"]):
            layer = Block_ViT(config, vis, channel_num)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, emb1, emb2, emb3, emb4):
        attn_weights = []
        for layer_block in self.layer:
            emb1, emb2, emb3, emb4, weights = layer_block(emb1, emb2, emb3, emb4)
            if self.vis:
                attn_weights.append(weights)
        emb1 = self.encoder_norm1(emb1) if emb1 is not None else None
        emb2 = self.encoder_norm2(emb2) if emb2 is not None else None
        emb3 = self.encoder_norm3(emb3) if emb3 is not None else None
        emb4 = self.encoder_norm4(emb4) if emb4 is not None else None
        return emb1, emb2, emb3, emb4, attn_weights


class ChannelTransformer(nn.Module):
    def __init__(self, config, vis, img_size, channel_num=[64, 128, 256, 512], patchSize=[32, 16, 8, 4]):
        super().__init__()

        self.patchSize_1 = patchSize[0]
        self.patchSize_2 = patchSize[1]
        self.patchSize_3 = patchSize[2]
        self.patchSize_4 = patchSize[3]
        self.embeddings_1 = Channel_Embeddings(config, self.patchSize_1, img_size=img_size, in_channels=channel_num[0])
        self.embeddings_2 = Channel_Embeddings(config, self.patchSize_2, img_size=img_size // 2, in_channels=channel_num[1])
        self.embeddings_3 = Channel_Embeddings(config, self.patchSize_3, img_size=img_size // 4, in_channels=channel_num[2])
        self.embeddings_4 = Channel_Embeddings(config, self.patchSize_4, img_size=img_size // 8, in_channels=channel_num[3])
        self.encoder = Encoder(config, vis, channel_num)

        self.reconstruct_1 = Reconstruct(channel_num[0], channel_num[0], kernel_size=1, scale_factor=(self.patchSize_1, self.patchSize_1))
        self.reconstruct_2 = Reconstruct(channel_num[1], channel_num[1], kernel_size=1, scale_factor=(self.patchSize_2, self.patchSize_2))
        self.reconstruct_3 = Reconstruct(channel_num[2], channel_num[2], kernel_size=1, scale_factor=(self.patchSize_3, self.patchSize_3))
        self.reconstruct_4 = Reconstruct(channel_num[3], channel_num[3], kernel_size=1, scale_factor=(self.patchSize_4, self.patchSize_4))

    def forward(self, en1, en2, en3, en4):
        emb1 = self.embeddings_1(en1)
        emb2 = self.embeddings_2(en2)
        emb3 = self.embeddings_3(en3)
        emb4 = self.embeddings_4(en4)

        encoded1, encoded2, encoded3, encoded4, attn_weights = self.encoder(emb1, emb2, emb3, emb4)  # (B, n_patch, hidden)

        x1 = self.reconstruct_1(encoded1) if en1 is not None else None
        x2 = self.reconstruct_2(encoded2) if en2 is not None else None
        x3 = self.reconstruct_3(encoded3) if en3 is not None else None
        x4 = self.reconstruct_4(encoded4) if en4 is not None else None

        x1 = x1 + en1 if en1 is not None else None
        x2 = x2 + en2 if en2 is not None else None
        x3 = x3 + en3 if en3 is not None else None
        x4 = x4 + en4 if en4 is not None else None

        return x1, x2, x3, x4, attn_weights


def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()


def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(CBN(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(CBN(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class CBN(nn.Module):
    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(CBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super().__init__()
        self.block = _make_nConv(in_channels, out_channels, nb_Conv=2, activation=activation)

    def forward(self, x):
        return self.block(x)


class DirMSGate(nn.Module):
    def __init__(self, dim, group_kernel_sizes=(3, 5, 7, 9), gate_layer="sigmoid"):
        super().__init__()
        assert dim % 4 == 0, "dim must be divisible by 4"
        gch = dim // 4

        self.local_dwc = nn.Conv1d(
            gch, gch, kernel_size=group_kernel_sizes[0],
            padding=group_kernel_sizes[0] // 2, groups=gch, bias=False
        )
        self.global_dwc_s = nn.Conv1d(
            gch, gch, kernel_size=group_kernel_sizes[1],
            padding=group_kernel_sizes[1] // 2, groups=gch, bias=False
        )
        self.global_dwc_m = nn.Conv1d(
            gch, gch, kernel_size=group_kernel_sizes[2],
            padding=group_kernel_sizes[2] // 2, groups=gch, bias=False
        )
        self.global_dwc_l = nn.Conv1d(
            gch, gch, kernel_size=group_kernel_sizes[3],
            padding=group_kernel_sizes[3] // 2, groups=gch, bias=False
        )

        # GN is stable for small batch
        self.norm_h = nn.GroupNorm(4, dim)
        self.norm_w = nn.GroupNorm(4, dim)

        if gate_layer == "softmax":
            self.act = nn.Softmax(dim=2)
        else:
            self.act = nn.Sigmoid()

        # (B,C,H,W) -> (B,1,H,W)
        self.proj = nn.Conv2d(dim, 1, kernel_size=1, bias=True)
        nn.init.constant_(self.proj.bias, 0.0)

    def _dir_attn(self, x, is_h: bool):
        # x: (B,C,H,W)
        if is_h:
            t_mean = x.mean(dim=3)     # (B,C,H)
            t_max = x.max(dim=3)[0]    # (B,C,H)
        else:
            t_mean = x.mean(dim=2)     # (B,C,W)
            t_max = x.max(dim=2)[0]    # (B,C,W)

        t = t_mean + t_max

        l, s, m, lrg = torch.chunk(t, 4, dim=1)  # each: (B,C/4,L)

        y = torch.cat(
            [
                self.local_dwc(l),
                self.global_dwc_s(s),
                self.global_dwc_m(m),
                self.global_dwc_l(lrg),
            ],
            dim=1,
        )  # (B,C,L)

        y = self.norm_h(y) if is_h else self.norm_w(y)
        y = self.act(y)
        return y

    def forward(self, x):
        # x: (B,C,H,W)
        b, c, h, w = x.shape
        ah = self._dir_attn(x, is_h=True).view(b, c, h, 1)   # (B,C,H,1)
        aw = self._dir_attn(x, is_h=False).view(b, c, 1, w)  # (B,C,1,W)
        a = ah * aw                                           # (B,C,H,W)
        g = torch.sigmoid(self.proj(a))                       # (B,1,H,W)
        return g


class JointDualGate(nn.Module):
    def __init__(self, dim_cat, group_kernel_sizes=(3, 5, 7, 9), gate_layer="sigmoid"):
        super().__init__()
        self.a = DirMSGate(
            dim_cat,
            group_kernel_sizes=group_kernel_sizes,
            gate_layer=gate_layer
        )
        self.to2 = nn.Conv2d(1, 2, kernel_size=1, bias=True)
        nn.init.constant_(self.to2.bias, 0.0)

    def forward(self, z):
        # z: (B, Cskip+Cup, H, W)
        g = self.a(z)                    # (B,1,H,W)
        g2 = torch.sigmoid(self.to2(g))  # (B,2,H,W)
        g_skip = g2[:, 0:1, :, :]        # (B,1,H,W)
        g_up = g2[:, 1:2, :, :]          # (B,1,H,W)
        return g_skip, g_up


class JointChannelGate(nn.Module):
    def __init__(self, c_skip, c_up, reduction=16):
        super().__init__()
        self.c_skip = c_skip
        self.c_up = c_up
        dim_cat = c_skip + c_up
        hidden = max(dim_cat // reduction, 8)

        self.mlp = nn.Sequential(
            nn.Conv2d(dim_cat, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, dim_cat, kernel_size=1, bias=False)
        )

    def forward(self, z):
        # z: (B, C_skip + C_up, H, W)
        z_avg = F.adaptive_avg_pool2d(z, 1)   # (B,C,1,1)
        z_max = F.adaptive_max_pool2d(z, 1)   # (B,C,1,1)

        c = self.mlp(z_avg) + self.mlp(z_max) # (B,C,1,1)
        c = torch.sigmoid(c)

        c_skip, c_up = torch.split(c, [self.c_skip, self.c_up], dim=1)
        return c_skip, c_up


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_rbs=False,
        activation="ReLU",
        use_joint_gate=True,
        use_channel_gate=True,
        gate_residual=True,
        group_kernel_sizes=(3, 5, 7, 9),
        gate_layer="sigmoid",
        ch_reduction=16,
    ):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.use_joint_gate = use_joint_gate
        self.use_channel_gate = use_channel_gate
        self.gate_residual = gate_residual

        if self.use_joint_gate:
            self.joint_gate = JointDualGate(
                dim_cat=in_channels + skip_channels,
                group_kernel_sizes=group_kernel_sizes,
                gate_layer=gate_layer,
            )

        if self.use_channel_gate:
            self.channel_gate = JointChannelGate(
                c_skip=skip_channels,
                c_up=in_channels,
                reduction=ch_reduction,
            )

        fuse_in_channels = in_channels + skip_channels
        if use_rbs:
            self.fuse = Res_block(fuse_in_channels, out_channels)
        else:
            self.fuse = _make_nConv(
                fuse_in_channels,
                out_channels,
                nb_Conv=2,
                activation=activation
            )

    def _apply_gate(self, x, g):
        # x: (B,C,H,W)
        # g: (B,1,H,W) or (B,C,1,1)
        if self.gate_residual:
            return x * (1.0 + g)
        else:
            return x * g

    def forward(self, x, skip_x):
        # x: deeper feature
        # skip_x: shallower feature
        up = self.up(x)

        # ===== 1) spatial dual gate on original joint feature =====
        z = torch.cat([skip_x, up], dim=1)  # (B, C_skip+C_up, H, W)

        if self.use_joint_gate:
            g_skip, g_up = self.joint_gate(z)
            skip_x = self._apply_gate(skip_x, g_skip)   # spatially gated skip
            up = self._apply_gate(up, g_up)             # spatially gated up

        # ===== 2) channel gate on gated joint feature z_g =====
        if self.use_channel_gate:
            z_g = torch.cat([skip_x, up], dim=1)        
            c_skip, c_up = self.channel_gate(z_g)
            skip_x = self._apply_gate(skip_x, c_skip)   # channel reweight for skip
            up = self._apply_gate(up, c_up)             # channel reweight for up

        # ===== 3) fuse =====
        out = torch.cat([skip_x, up], dim=1)
        out = self.fuse(out)
        return out



class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Res_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # self.fca = FCA_Layer(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out


class DifTransNet(nn.Module):
    def __init__(
        self,
        num_classes=1,
        input_channels=1,
        img_size=256,
        deep_supervision=True,
        deepsuper=None,
        mode='test',
        use_rbs=True,
        use_transformer=True,
        base_channels=None,
    ):
        if deepsuper is not None:
            deep_supervision = deepsuper

        config = get_CTranS_config()
        if base_channels is not None:
            config.base_channel = base_channels

        n_channels = input_channels
        n_classes = num_classes
        vis = False

        super().__init__()
        self.vis = vis
        self.deepsuper = deep_supervision
        self.use_rbs = use_rbs
        self.use_transformer = use_transformer
        self.mode = mode
        self.n_channels = n_channels
        self.n_classes = n_classes

        c1 = config.base_channel
        c2 = c1 * 2
        c3 = c2 * 2
        c4 = c3 * 2
        c5 = c4

        block = Res_block if self.use_rbs else ConvBlock
        self.pool = nn.MaxPool2d(2, 2)
        self.inc = self._make_layer(block, n_channels, c1)
        self.down_encoder1 = self._make_layer(block, c1, c2, 1)
        self.down_encoder2 = self._make_layer(block, c2, c3, 1)
        self.down_encoder3 = self._make_layer(block, c3, c4, 1)
        self.down_encoder4 = self._make_layer(block, c4, c5, 1)

        self.mtc = None
        if self.use_transformer:
            self.mtc = ChannelTransformer(
                config,
                vis,
                img_size,
                channel_num=[c1, c2, c3, c4],
                patchSize=config.patch_sizes,
            )

        gate_ks = (1,3,5,7)
        print(f"[Gate Config] group_kernel_sizes = {gate_ks}")

        self.up_decoder4 = UpBlock(c5, c4, c3, use_rbs=self.use_rbs, group_kernel_sizes=gate_ks)
        self.up_decoder3 = UpBlock(c3, c3, c2, use_rbs=self.use_rbs, group_kernel_sizes=gate_ks)
        self.up_decoder2 = UpBlock(c2, c2, c1, use_rbs=self.use_rbs, group_kernel_sizes=gate_ks)
        self.up_decoder1 = UpBlock(c1, c1, c1, use_rbs=self.use_rbs, group_kernel_sizes=gate_ks)
        self.outc = nn.Conv2d(c1, n_classes, kernel_size=1, stride=1)

        if self.deepsuper:
            self.gt_conv5 = nn.Conv2d(c5, n_classes, 1)
            self.gt_conv4 = nn.Conv2d(c3, n_classes, 1)
            self.gt_conv3 = nn.Conv2d(c2, n_classes, 1)
            self.gt_conv2 = nn.Conv2d(c1, n_classes, 1)
            self.outconv = nn.Conv2d(5 * n_classes, n_classes, 1)

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for _ in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down_encoder1(self.pool(x1))
        x3 = self.down_encoder2(self.pool(x2))
        x4 = self.down_encoder3(self.pool(x3))
        d5 = self.down_encoder4(self.pool(x4))

        if self.use_transformer:
            f1, f2, f3, f4 = x1, x2, x3, x4
            x1, x2, x3, x4, _ = self.mtc(x1, x2, x3, x4)
            x1 = x1 + f1
            x2 = x2 + f2
            x3 = x3 + f3
            x4 = x4 + f4

        d4 = self.up_decoder4(d5, x4)
        d3 = self.up_decoder3(d4, x3)
        d2 = self.up_decoder2(d3, x2)
        d1 = self.up_decoder1(d2, x1)
        out = self.outc(d1)

        if self.deepsuper:
            gt_5 = self.gt_conv5(d5)
            gt_4 = self.gt_conv4(d4)
            gt_3 = self.gt_conv3(d3)
            gt_2 = self.gt_conv2(d2)

            out_size = out.shape[-2:]
            gt5 = F.interpolate(gt_5, size=out_size, mode='bilinear', align_corners=True)
            gt4 = F.interpolate(gt_4, size=out_size, mode='bilinear', align_corners=True)
            gt3 = F.interpolate(gt_3, size=out_size, mode='bilinear', align_corners=True)
            gt2 = F.interpolate(gt_2, size=out_size, mode='bilinear', align_corners=True)
            d0 = self.outconv(torch.cat((gt2, gt3, gt4, gt5, out), 1))

            if self.mode == 'train':
                return (
                    torch.sigmoid(gt5),
                    torch.sigmoid(gt4),
                    torch.sigmoid(gt3),
                    torch.sigmoid(gt2),
                    torch.sigmoid(d0),
                    torch.sigmoid(out),
                )
            else:
                return torch.sigmoid(out)
        else:
            return torch.sigmoid(out)
