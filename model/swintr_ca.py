""" Swin Transformer
A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`
    - https://arxiv.org/pdf/2103.14030

Code/weights from https://github.com/microsoft/Swin-Transformer

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional


device = torch.device("cuda:0")
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

# 定义交叉注意力模块

import torch
import torch.nn as nn


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x





class CA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CA, self).__init__()

        self.out_channels = out_channels
        self.conv1 = DoubleConv(in_channels[0], out_channels)
        self.conv2 = DoubleConv(in_channels[1], out_channels)
        self.up_sam = nn.ConvTranspose2d(in_channels[1],in_channels[1], 2, stride=2)
        self.conv3 = DoubleConv(2*in_channels[1],in_channels[0])
        # channel attention
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.mlp = nn.Sequential(
            # Conv2d比Linear
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(2*in_channels[1], 4*in_channels[1], 1, bias=False),
            # inplace=True
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(4*in_channels[1], in_channels[0], 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

        self.relu = nn.ReLU()



    def forward(self, x1, x2): #(unet,swintransformer)


        b, c1, h1, w1 = x1.shape
        b, c2, h2, w2 = x2.shape
        # print(b,c1,h1,w1)
        # print(b,c2,h2,w2)


        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x2 = self.up_sam(x2)

        me1 = torch.cat([x1,x2],dim=1)

        max_out = self.mlp(self.max_pool(me1))
        avg_out = self.mlp(self.avg_pool(me1))
        # print('me1:', me1.shape, 'avg:', avg_out.shape, 'max:', max_out.shape)
        channel_out = self.sigmoid(max_out + avg_out)
        x = self.conv3(me1)
        x = channel_out * x
        return x


class Csp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Csp, self).__init__()

        self.out_channels = out_channels
        self.conv1 = DoubleConv(in_channels[0], out_channels)
        self.conv2 = DoubleConv(in_channels[1], out_channels)
        self.up_sam = nn.ConvTranspose2d(in_channels[1], in_channels[1], 2, stride=2)
        self.conv3 = DoubleConv(2 * in_channels[1], in_channels[0])
        # channel attention


        self.conv = nn.Conv2d(2, 1, kernel_size=7,
                              padding=3, bias=False)



        self.sigmoid = nn.Sigmoid()

        # self.relu = nn.ReLU()

    def forward(self, x1, x2):
        b, c1, h1, w1 = x1.shape
        b, c2, h2, w2 = x2.shape
        # print(b, c1, h1, w1)
        # print(b, c2, h2, w2)

        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x2 = self.up_sam(x2)

        m1 = torch.cat([x1, x2], dim=1)

        max_out, _ = torch.max(m1, dim=1, keepdim=True)
        avg_out = torch.mean(m1, dim=1, keepdim=True)
        # print('m1:',m1.shape,'avg:',avg_out.shape,'max:',max_out.shape)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        m1 = self.conv3(m1)
        m1 = spatial_out * m1
        return m1


def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


def window_partition(x, window_size: int):
    """
    将feature map按照window_size划分成一个个没有重叠的window
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    将一个个window还原成一个feature map
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, patch_size=4, in_c=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_chans = in_c
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape

        # padding
        # 如果输入图片的H，W不是patch_size的整数倍，需要进行padding
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            # to pad the last 3 dimensions,
            # (W_left, W_right, H_top,H_bottom, C_front, C_back)
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],
                          0, self.patch_size[0] - H % self.patch_size[0],
                          0, 0))

        # 下采样patch_size倍
        x = self.proj(x)
        _, _, H, W = x.shape
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        # 如果输入feature map的H，W不是2的整数倍，需要进行padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            # to pad the last 3 dimensions, starting from the last dimension and moving forward.
            # (C_front, C_back, W_left, W_right, H_top, H_bottom)
            # 注意这里的Tensor通道是[B, H, W, C]，所以会和官方文档有些不同
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
        x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]

        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2*W/2, 2*C]

        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # [Mh, Mw]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2*Mh-1 * 2*Mw-1, nH]

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # [2, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw]
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Mh*Mw, Mh*Mw]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, Mh*Mw, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # [batch_size*num_windows, Mh*Mw, total_embed_dim]
        B_, N, C = x.shape
        # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # transpose: -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask):
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        # 把feature map给pad到window size的整数倍
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, Mh, Mw, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [nW*B, Mh*Mw, C]

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, Mh*Mw, C]

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  # [nW*B, Mh, Mw, C]
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # [B, H', W', C]

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            # 把前面pad的数据移除掉
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicLayer(nn.Module):
    """
    A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.shift_size = window_size // 2

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def create_mask(self, x, H, W):
        # calculate attention mask for SW-MSA
        # 保证Hp和Wp是window_size的整数倍
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        # 拥有和feature map一样的通道排列顺序，方便后续window_partition
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, H, W):
        attn_mask = self.create_mask(x, H, W)  # [nW, Mh*Mw, Mh*Mw]
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        real_x = x
        if self.downsample is not None:
            x = self.downsample(x, H, W)
            H, W = (H + 1) // 2, (W + 1) // 2

        return real_x,x, H, W

# 定义一个函数，用于计算每个token的信噪比
# def snr_tokens(g, v):
#     # g: 灰度值，形状为[B, C, H, W]
#     # v: 方差，形状为[B, C, H, W]
#     # 返回值: 每个token的信噪比，形状为[B, N]，其中N是token的数量
#     B, C, H, W = g.shape
#     P = int(H ** 0.5) # token的大小，假设是平方数
#     g = split_tokens(g, P) # 划分token，形状为[B, N, C, P, P]
#     v = split_tokens(v, P) # 划分token，形状为[B, N, C, P, P]
#     g = torch.mean(g, dim=(2, 3, 4)) # 计算每个token的平均灰度值，形状为[B, N]
#     v = torch.mean(v, dim=(2, 3, 4)) # 计算每个token的平均方差，形状为[B, N]
#     snr = g / (v + 1e-8) # 计算每个token的信噪比，形状为[B, N]，加一个小的常数防止除零错误
#     return snr
#
# # 定义一个函数，用于对token进行排序，并选择前K个最大的token，其他的token则被mask掉
# def sort_tokens(x, k):
#     # x: 输入的token，形状为[B, N, C, P, P]
#     # k: 要选择的token的个数，一个整数
#     # 返回值: 经过mask的token，形状为[B, N, C, P, P]
#     B, N, C, P, P = x.shape
#     x = x.reshape(B, N, -1) # 将token展平，形状为[B, N, C * P * P]
#     g = torch.mean(x, dim=-1) # 计算每个token的平均值，形状为[B, N]
#     v = torch.var(x, dim=-1) # 计算每个token的方差，形状为[B, N]
#     snr = snr_tokens(g, v) # 计算每个token的信噪比，形状为[B, N]
#     _, indices = torch.topk(snr, k, dim=-1) # 对信噪比进行排序，并选择前K个最大的token的索引，形状为[B, K]
#     mask = torch.zeros(B, N, device=x.device) # 创建一个mask矩阵，形状为[B, N]
#     mask.scatter_(1, indices, 1) # 将被选择的token的mask置为1，形状为[B, N]
#     mask = mask.unsqueeze(-1).repeat(1, 1, C * P * P) # 复制mask，形状为[B, N, C * P * P]
#     x = x * mask # 将被mask掉的token的值置为0，形状为[B, N, C * P * P]
#     x = x.reshape(B, N, C, P, P) # 将token恢复原来的形状，形状为[B, N, C, P, P]
#     return x


class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, patch_size=4, base_chanel = 16,in_chans=3, num_classes=1000,
                 embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.base_chanel = base_chanel
        self.patch_norm = patch_norm
        # stage4输出特征矩阵的channels
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # 注意这里构建的stage和论文图中有些差异
            # 这里的stage不包含该stage的patch_merging层，包含的是下个stage的
            layers = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                use_checkpoint=use_checkpoint)
            self.layers.append(layers)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)


        #unet _ encoder

        self.uconv1 = DoubleConv(3, 2*self.base_chanel)
        self.pool1 = nn.MaxPool2d(2)
        self.uconv2 = DoubleConv(2*self.base_chanel, 4*self.base_chanel)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(4*self.base_chanel, 8*self.base_chanel)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(8*self.base_chanel, 16*self.base_chanel)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(16*self.base_chanel, 32*self.base_chanel)

        #gdff
        # self.gdff_up1 = nn.ConvTranspose2d(self.embed_dim * 8, self.embed_dim * 4, 2, stride=2)
        # self.gdff_up2 = nn.ConvTranspose2d(self.embed_dim * 4, self.embed_dim * 2, 2, stride=2)
        # self.gdff_up3 = nn.ConvTranspose2d(self.embed_dim * 2, self.embed_dim * 1, 2, stride=2)
        # self.gdff_up4 = nn.ConvTranspose2d(self.embed_dim , int(self.embed_dim /2), 2, stride=2)
        #
        # self.gd_conv1 = DoubleConv(self.embed_dim * 4,32*self.base_chanel)
        # self.gd_conv2 = DoubleConv(self.embed_dim * 2, 16*self.base_chanel)
        # self.gd_conv3 = DoubleConv(self.embed_dim * 1, 8*self.base_chanel)
        # self.gd_conv4 = DoubleConv(int(self.embed_dim /2), 4*self.base_chanel)
        #
        # self.ff_conv1 = DoubleConv(64*self.base_chanel, 32*self.base_chanel)
        # self.ff_conv2 = DoubleConv(32*self.base_chanel, 16*self.base_chanel)
        # self.ff_conv3 = DoubleConv(16*self.base_chanel, 8*self.base_chanel)
        # self.ff_conv4 = DoubleConv(8*self.base_chanel, 4*self.base_chanel)

        #swin u decoder


        # self.trconv0 = DoubleConv(self.embed_dim*4,1024)
        # self.trconv = DoubleConv(2048, 1024)
        self.outconv4 = DoubleConv(32*self.base_chanel,2)
        # self.trup6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.up6 = nn.ConvTranspose2d(32*self.base_chanel, 16*self.base_chanel, 2, stride=2)
        self.conv6 = DoubleConv(32*self.base_chanel, 16*self.base_chanel)
        self.outconv3 = DoubleConv(16*self.base_chanel,2)
        # self.trup7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up7 = nn.ConvTranspose2d(16*self.base_chanel, 8*self.base_chanel, 2, stride=2)
        self.conv7 = DoubleConv(16*self.base_chanel, 8*self.base_chanel)
        self.outconv2 = DoubleConv(8*self.base_chanel,2)
        # self.trup8 = nn.ConvTranspose2d(64, 48, 2, stride=2)
        self.up8 = nn.ConvTranspose2d(8*self.base_chanel, 4*self.base_chanel, 2, stride=2)
        self.conv8 = DoubleConv(8*self.base_chanel, 4*self.base_chanel)
        self.outconv1 = DoubleConv(4*self.base_chanel,2)
        # self.trup9 = nn.ConvTranspose2d(48, 32, 2, stride=2)
        self.up9 = nn.ConvTranspose2d(4*self.base_chanel, 2*self.base_chanel, 2, stride=2)
        self.conv9 = DoubleConv(4*self.base_chanel, 2*self.base_chanel)
        self.conv10 = nn.Conv2d(2*self.base_chanel, self.num_classes, 1)

        #CA 通道注意力特征融合
        # CSP 空间注意力特征融合
        #cross_attention
        self.ca1 = CA([32*self.base_chanel,8*self.embed_dim],8*self.embed_dim)
        self.ca2 = CA([16*self.base_chanel,4*self.embed_dim] ,4 * self.embed_dim)
        self.ca3 = Csp([8 * self.base_chanel,2 * self.embed_dim], 2 * self.embed_dim)
        self.ca4 = Csp([4 * self.base_chanel,self.embed_dim ], self.embed_dim)




        self.conv1 = nn.Conv2d(
            in_channels=self.embed_dim*8,
            out_channels=self.embed_dim*8,
            kernel_size=1,
            stride=1,
            padding=self._get_padding('VALID', (1, 1), ),
        )
        self.bn1 = nn.BatchNorm2d(self.embed_dim*8)
        self.bn2 = nn.BatchNorm2d(self.embed_dim * 4)
        self.bn3 = nn.BatchNorm2d(self.embed_dim * 2)
        self.bn4 = nn.BatchNorm2d(self.embed_dim * 1)
        self.act1 = nn.ReLU()


    def unet_encoder(self,x):
        #unet encoder
        c1 = self.uconv1(x)
        p1 = self.pool1(c1)
        c2 = self.uconv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        return c1,c2,c3,c4,c5

    def snr_order(self, x):
        # x: 输入的token，形状为[B, N, D]
        # 返回值: token的snr，形状为[B, N]
        mean = torch.mean(x, dim=-1)  # 计算token的均值，形状为[B, N]
        std = torch.std(x, dim=-1)  # 计算token的标准差，形状为[B, N]
        snr = mean / (std + 1e-6)  # 计算token的snr，形状为[B, N]
        return snr

    # 定义一个函数，用于对token进行排序，并保留前p%的token，其他的token被mask掉
    def sort_token(self, x, p):
        # x: 输入的token，形状为[B, N, D]
        # p: 要保留的token的比例，一个浮点数
        # 返回值: 排序后的token，形状为[B, N, D]
        B, N, D = x.shape
        snr = self.snr_order(x)  # 计算token的snr，形状为[B, N]
        k = int(p * N)  # 计算要保留的token的个数，一个整数
        values, indices = torch.topk(snr, k=k, dim=-1, largest=True,
                                     sorted=True)  # 对snr进行topk操作，得到最大的k个元素的值和索引，形状为[B, k]
        mask = torch.zeros(B, N, device=x.device)  # 创建一个mask矩阵，形状为[B, N]
        mask.scatter_(1, indices, 1)  # 将被选择的token的mask置为1，形状为[B, N]
        x = x * mask.unsqueeze(-1)  # 将被mask掉的token的值置为0，形状为[B, N, D]
        return x



    def _reshape_output(self, x_list):

        # create an empty list to store the reshaped tensors
        y_list = []
        # loop through the x_list and apply the reshape operation to each tensor
        for x in x_list:
            # get the square root of the second dimension
            n = int(x.size(1) ** 0.5)
            # check if the second dimension is a perfect square
            if n * n != x.size(1):
                raise ValueError("The second dimension must be a perfect square")
            # reshape the tensor to four dimensions
            x = x.reshape(
                x.size(0),
                n,
                n,
                x.size(2),
            )
            # permute the dimensions to match the original order




            x = x.permute(0, 3, 1, 2).contiguous()
            # append the reshaped tensor to the y_list
            # print(x.shape,x.size(1))

            if x.size(1) == self.embed_dim*8:
                x = self.bn1(x)
            elif x.size(1) == self.embed_dim*4:
                x = self.bn2(x)
            elif x.size(1) == self.embed_dim*2:
                x = self.bn3(x)
            else:

                x = self.bn4(x)
            # print(x.shape)
            x = self.act1(x)
            y_list.append(x)
        # return the y_list
        return y_list



    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)

        # print(high_feature.shape,low_feature.shape)
        # print(ff.shape)

    # def GDFF(self,uf,swf):
    #     if swf.size(1) ==self.embed_dim * 8:
    #
    #         ff = self.ca1(uf,swf)
    #     elif swf.size(1) == self.embed_dim * 4:
    #         # print('ca2_uf:',uf.shape,'ca2_swf:',swf.shape)
    #         ff = self.ca2(uf, swf)
    #     elif swf.size(1) == self.embed_dim *2:
    #         ff = self.ca3(uf,swf)
    #     else:
    #         ff = self.ca4(uf,swf)
    #     return ff

        # print('uf:', uf.shape, 'swf', swf.shape,'ff',ff.shape)



    # def GDFF(self,low_feature,high_feature):
    #     if high_feature.size(1) == self.embed_dim*8:
    #         high_feature = self.gdff_up1(high_feature)
    #         high_feature = self.gd_conv1(high_feature)
    #         ff = torch.cat([high_feature,low_feature],dim=1)
    #
    #         ff = self.ff_conv1(ff)
    #     elif high_feature.size(1) == self.embed_dim*4:
    #         high_feature = self.gdff_up2(high_feature)
    #         high_feature = self.gd_conv2(high_feature)
    #         ff = torch.cat([high_feature,low_feature],dim=1)
    #         ff = self.ff_conv2(ff)
    #     elif high_feature.size(1) == self.embed_dim*2:
    #         high_feature = self.gdff_up3(high_feature)
    #         high_feature = self.gd_conv3(high_feature)
    #         ff = torch.cat([high_feature,low_feature],dim=1)
    #         ff = self.ff_conv3(ff)
    #     else:
    #         high_feature = self.gdff_up4(high_feature)
    #         high_feature = self.gd_conv4(high_feature)
    #         ff = torch.cat([high_feature,low_feature],dim=1)
    #
    #         ff = self.ff_conv4(ff)

    def swin_unet_decoder(self,swin_encoder_layer_x,c1,c2,c3,c4,c5):
        # print('c1,c2,c3,c4,c5',c1.shape,c2.shape,c3.shape,c4.shape,c5.shape)
        # print(swin_encoder_layer_x[0].shape)
        x = self._reshape_output(swin_encoder_layer_x)


         # x.shape = 7*7*768
        x4 = self.ca1(c5,x[3])
        out4 = self.outconv4(x4)

        # print(x.shape)
        up_6 = self.up6(x4)

        # x4 = self.trup6(x4)
        x3 = self.ca2(c4, x[2])

        merge6 = torch.cat([up_6,x3], dim=1)




        c6 = self.conv6(merge6)
        out3 = self.outconv3(c6)
        # x3 = self.trup7(x3)
        up_7 = self.up7(c6)
        x2 = self.ca3(c3, x[1])
        merge7 = torch.cat([up_7,x2], dim=1)
        c7 = self.conv7(merge7)
        out2 = self.outconv2(c7)


        up_8 = self.up8(c7)
        x1 = self.ca4(c2, x[0])
        merge8 = torch.cat([up_8,x1], dim=1)
        # print(up_8.shape,x1.shape)
        # print(merge8.shape)
        c8 = self.conv8(merge8)
        out1 = self.outconv1(c8)
        # x = self.trup9(x)

        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        # out = nn.Sigmoid()(c10)
        out = c10
        return out,out1,out2,out3,out4

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x: [B, L, C]
        # print(x.shape)
        c1,c2,c3,c4,c5 = self.unet_encoder(x)
        x, H, W = self.patch_embed(x)
        # print(x.shape,H,W)

        x = self.pos_drop(x)
        # zero_count = torch.sum(torch.all(x == 0, dim=2), dim=1)
        # print('zero_number00:', zero_count)

        swin_encoder_layer_x = []
        # print('x:',x)
        x = self.sort_token(x, 0.8)

        # zero_count = torch.sum(torch.all(x == 0, dim=2), dim=1)
        # print('zero_number0:',zero_count)
        # print('x_select:',x)
        # i = 1
        for layer in self.layers:
            r_x,x, H, W = layer(x, H, W)
            # x = self.sort_token(x, 0.5)
            # zero_count = torch.sum(torch.all(x == 0, dim=2), dim=1)
            # print('zero_number:', i ,zero_count)
            # i+=1

            swin_encoder_layer_x.append(r_x)
            # print('stage: ', x.shape)
        #print('sf_swin',x.shape,H,W)
        # print('swin_encoder_layer_x:', swin_encoder_layer_x[0].shape , swin_encoder_layer_x[1].shape , swin_encoder_layer_x[2].shape , swin_encoder_layer_x[3].shape)
        # x = self.norm(x)  # [B, L, C]
        # x = x.transpose(1, 2)  # [B, C, L]
        #print(x.shape)
        x = self.swin_unet_decoder(swin_encoder_layer_x,c1,c2,c3,c4,c5)

        return x


def swin_tiny_patch4_window7_224(num_classes: int = 2,embed_dim=64,base_chanel= 24, **kwargs):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth


    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=48,
                            base_chanel= 16,
                            depths=(2, 2, 6, 2),
                            num_heads=(3, 6, 12, 24),
                            num_classes=num_classes,
                            **kwargs)
    return model




def swin_small_patch4_window7_224(num_classes: int = 1000, **kwargs):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=96,
                            base_chanel= 32,
                            depths=(2, 2, 18, 2),
                            num_heads=(3, 6, 12, 24),
                            num_classes=num_classes,
                            **kwargs)
    return model


def swin_base_patch4_window7_224(num_classes: int = 1000, **kwargs):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=128,
                            depths=(2, 2, 18, 2),
                            num_heads=(4, 8, 16, 32),
                            num_classes=num_classes,
                            **kwargs)
    return model


def swin_base_patch4_window12_384(num_classes: int = 1000, **kwargs):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=12,
                            embed_dim=128,
                            base_chanel= 32,
                            depths=(2, 2, 18, 2),
                            num_heads=(4, 8, 16, 32),
                            num_classes=num_classes,
                            **kwargs)
    return model


def swin_base_patch4_window7_224_in22k(num_classes: int = 21841,embed_dim=64,base_chanel= 24, **kwargs):
    # trained ImageNet-22K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=embed_dim,
                            base_chanel= base_chanel,
                            depths=(2, 2, 18, 2),
                            num_heads=(4, 8, 16, 32),
                            num_classes=num_classes,
                            **kwargs)
    return model


def swin_base_patch4_window12_384_in22k(num_classes: int = 21841, **kwargs):
    # trained ImageNet-22K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=12,
                            embed_dim=128,
                            depths=(2, 2, 18, 2),
                            num_heads=(4, 8, 16, 32),
                            num_classes=num_classes,
                            **kwargs)
    return model


def swin_large_patch4_window7_224_in22k(num_classes: int = 21841, **kwargs):
    # trained ImageNet-22K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=192,
                            depths=(2, 2, 18, 2),
                            num_heads=(6, 12, 24, 48),
                            num_classes=num_classes,
                            **kwargs)
    return model


def swin_large_patch4_window12_384_in22k(num_classes: int = 21841,embed_dim=64,base_chanel= 24, **kwargs):
    # trained ImageNet-22K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=12,
                            embed_dim=192,
                            base_chanel= 64,
                            depths=(2, 2, 18, 2),
                            num_heads=(6, 12, 24, 48),
                            num_classes=num_classes,
                            **kwargs)
    return model