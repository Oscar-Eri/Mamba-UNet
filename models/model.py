from thop import profile
from config import cfg_Ideatormer
import torch.profiler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from mamba_ssm import Mamba
import numbers
from einops import rearrange
import math




class WindowLocalAttention(nn.Module):
    """
    轻量化窗口局部注意力
    支持：在固定大小的窗口内执行多头点积注意力
    实现：先通过1×1卷积降维生成 q/k/v，再划分窗口、计算点积注意力并重组
    怎么操作：输入特征图，自动 Padding 到窗口尺寸倍数，窗口内自注意力，最后裁剪回原始大小
    """
    def __init__(self, dim, num_heads,bias,window_size=7, reduction=4):
        super().__init__()
        self.dim = dim
        self.ws = window_size
        self.num_heads = num_heads
        self.head_dim = (dim // reduction) // num_heads
        self.scale = self.head_dim ** -0.5

        # 降维生成 qkv：保持计算轻量
        hidden_dim = (dim // reduction)
        self.qkv = nn.Conv2d(dim, hidden_dim * 3, kernel_size=1, bias=bias)
        self.proj = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        pad_h = (self.ws - H % self.ws) % self.ws
        pad_w = (self.ws - W % self.ws) % self.ws
        x_pad = F.pad(x, (0, pad_w, 0, pad_h))  # 右下补齐

        Hp, Wp = H + pad_h, W + pad_w
        # 1. 生成 qkv 并划分多头
        qkv = self.qkv(x_pad)  # B, 3*hidden_dim, Hp, Wp
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, Hp, Wp)
        q, k, v = qkv[:,0], qkv[:,1], qkv[:,2]  # 每项：B, heads, head_dim, Hp, Wp

        # 2. 窗口切分 -> (num_windows*B, heads, head_dim, ws, ws)
        q_windows = q.unfold(3, self.ws, self.ws).unfold(4, self.ws, self.ws)  # B, heads, head_dim, nWh, nWw, ws, ws
        k_windows = k.unfold(3, self.ws, self.ws).unfold(4, self.ws, self.ws)
        v_windows = v.unfold(3, self.ws, self.ws).unfold(4, self.ws, self.ws)

        B_, Hn, Wn, ws, _ = q_windows.shape[0], q_windows.shape[3], q_windows.shape[4], self.ws, self.ws
        # 重排到 (B_*Hn*Wn, heads, head_dim, ws*ws)
        q_flat = q_windows.permute(0,3,4,1,2,5,6).reshape(-1, self.num_heads, self.head_dim, ws*ws)
        k_flat = k_windows.permute(0,3,4,1,2,5,6).reshape(-1, self.num_heads, self.head_dim, ws*ws)
        v_flat = v_windows.permute(0,3,4,1,2,5,6).reshape(-1, self.num_heads, self.head_dim, ws*ws)

        # 3. 点积注意力
        attn = torch.einsum('bhcN,bhcM->b h N M', q_flat * self.scale, k_flat)  # b,heads,N,N
        attn = F.softmax(attn, dim=-1)
        out_flat = torch.einsum('b h N M,bhcM->bhcN', attn, v_flat)  # b,heads,head_dim,ws*ws

        # 4. 重组窗口 -> 恢复空间
        out_windows = out_flat.reshape(B, Hn, Wn, self.num_heads, self.head_dim, ws, ws)
        out_windows = out_windows.permute(0,3,4,1,5,2,6)  # B,hidden_dim,nWh,ws,nWw,ws
        out_windows = out_windows.reshape(B, -1, Hp, Wp)  # B, hidden_dim, Hp, Wp

        # 5. 投影回原通道 + 裁剪回原始 H×W
        out = self.proj(out_windows)
        return out[:, :, :H, :W]
##################################################################low
class WGCA_Lite(nn.Module):
    """
    Wavelet‑Guided Channel Attention (Lite)
    - 输入：x_LL, x_HL, x_LH, x_HH
    - 核心：融合高频后，用全局平均池化 + 小型 MLP 生成通道注意力，再加权 x_LL
    """

    def __init__(self, channels, reduction=8):
        super().__init__()
        # 高频融合：1x1 降维
        self.hf_fuse = nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False)
        # 通道注意力 MLP
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        # 最终融合 1x1
        self.out_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x_LL, x_HL, x_LH, x_HH):
        # 1. 高频拼接融合
        hf = torch.cat([x_HL, x_LH, x_HH], dim=1)  # [B, 3C, H, W]
        hf_feat = self.hf_fuse(hf)  # [B,   C, H, W]

        # 2. 通道注意力生成
        attn = self.avgpool(hf_feat)  # [B, C, 1, 1]
        attn = self.mlp(attn)  # [B, C, 1, 1]

        # 3. 加权 LL 并残差融合
        ll_weighted = x_LL * attn  # [B, C, H, W]
        out = self.out_conv(ll_weighted) + x_LL  # [B, C, H, W]
        return out

##########################################################################

### Hybrid Module 曼巴块
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)

        self.mam = MambaLayer(dim)  # 全局建模

        self.use_attn = True
        if self.use_attn:
            self.attn = WindowLocalAttention(dim, num_heads=2, bias=bias)

        #  融合权重模块（动态 alpha）
        self.alpha_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        self.fuse = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.ffn = GFFN(dim, ffn_expansion_factor, bias)

        #  深度卷积 + SE 注意力增强
        self.smooth = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=bias)
        self.smooth_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 8, dim, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x, pos, order = x[0], x[1], x[2]
        x_spatial = x

        # 全局建模
        x_m = self.mam(self.norm1(x), pos, order)

        # 空间注意力
        if self.use_attn:
            x_sa = self.attn(self.norm1(x))
            alpha = self.alpha_conv(x)  # shape: [B, C, 1, 1]
            x_fused = alpha * x_m + (1 - alpha) * x_sa
        else:
            x_fused = x_m

        # 融合 + FFN
        x = x + self.fuse(x_fused)
        x = x + self.ffn(self.norm2(x), x_spatial)

        #  平滑卷积 + 通道注意力
        smooth_out = self.smooth(x)
        smooth_out = smooth_out * self.smooth_se(x)
        x = x + smooth_out

        return x


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


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


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim

        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        # 可学习扰动型位置编码
        self.learnable_pe = nn.Parameter(torch.randn(1, dim, 1, 1))  # 可扩展为 HxW 大小

        # 低频引导卷积
        self.lowpass = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, bias=False)

        # 动态融合权重
        self.alpha = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, 1, 1),
            nn.Sigmoid()
        )

    @autocast(enabled=False)
    def forward(self, x, pe, order):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.dim

        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]

        # 展平并加入扰动位置编码
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        pe = pe[:n_tokens, :] if pe is not None else 0
        x_flat = x_flat + pe + self.learnable_pe.flatten(2).transpose(1, 2)

        # Mamba 编码
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)

        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)

        # 引导融合
        low_freq = self.lowpass(x)
        a = self.alpha(x)  # [B, 1, 1, 1]
        out = a * out + (1 - a) * low_freq

        return out


##########################################################################
## SRSA
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.qkv2 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv2 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.pool = nn.AdaptiveAvgPool2d(8)

        #### SA attention
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True),
            LayerNorm(dim, 'BiasFree'),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True),
            LayerNorm(dim, 'BiasFree'),
            nn.ReLU(inplace=True)
        )
        self.upsample = nn.Upsample(scale_factor=2)
        ##########
        self.lce = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        #### Spatial Attention
        qkv2 = self.qkv_dwconv2(self.qkv2(x))
        q2, k2, v2 = qkv2.chunk(3, dim=1)

        v_lce = v2

        v2 = self.pool(v2)
        k2 = self.pool(k2)

        b2, c2, h2, w2 = v2.shape

        q2 = rearrange(q2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        k2 = rearrange(k2, 'b (head c) h2 w2 -> b head c (h2 w2)', head=self.num_heads)
        v2 = rearrange(v2, 'b (head c) h2 w2 -> b head c (h2 w2)', head=self.num_heads)

        v2 = torch.nn.functional.normalize(v2, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)

        attn_s = (q2.transpose(-2, -1) @ k2) * self.temperature2
        attn_s = attn_s.softmax(dim=-1)

        out2 = (v2 @ attn_s.transpose(-2, -1))

        out2 = rearrange(out2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out2 = self.lce(v_lce) + out2
        #######################

        out = self.project_out(out2)
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.fusion = nn.Conv2d(hidden_features + dim, hidden_features, kernel_size=1, bias=bias)
        self.dwconv_afterfusion = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                            groups=hidden_features, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, groups=dim, dilation=1, bias=True),
            LayerNorm(dim, 'BiasFree'),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(2),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, groups=dim, dilation=2, bias=True),
            LayerNorm(dim, 'BiasFree'),
            nn.ReLU(inplace=True)
        )
        self.upsample = nn.Upsample(scale_factor=2)

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, spatial):
        x = self.project_in(x)

        x1, x2 = self.dwconv(x).chunk(2, dim=1)

        x = F.gelu(x1) * x2
        x = self.project_out(x)

        x_c = self.pool(x)

        x_bc = x_c.mean(dim=1, keepdim=True)
        x = x + x_bc

        return x


class LayerNormffn(nn.Module):
    def __init__(self, dim, mode='BiasFree'):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=(mode == 'BiasFree'))

    def forward(self, x):
        # x: (B,C,H,W) -> (B,H,W,C) 归一化后还原
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


###############################################################GFFN

class GatedFeedForward(nn.Module):
    """
    Gated-FFN (GFFN)：适用于 2D 特征图
      1）1×1 卷积扩张通道至 hidden*2（包含门控和特征两路）
      2）分成两路：一路做 3×3 深度可分离卷积（保留空间信息），一路直接保留
      3）激活 + 门控相乘
      4）1×1 卷积投影回原通道数
    """

    def __init__(self, dim, expansion=4, dropout=0.0):
        super().__init__()
        hidden = dim * expansion
        self.project_in = nn.Conv2d(dim, hidden * 2, kernel_size=1)
        self.dwconv = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1)
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(dropout)

    def forward(self, x):
        x2 = self.project_in(x)
        u, v = x2.chunk(2, dim=1)
        u = self.dwconv(u)
        u = self.act(u)
        x = u * v
        x = self.project_out(x)
        return self.drop(x)


class GFFN(nn.Module):
    def __init__(self, dim, expansion_factor, bias=True, dropout=0.0):
        super().__init__()
        self.ffn = GatedFeedForward(dim, expansion=expansion_factor, dropout=dropout)
        self.norm = LayerNormffn(dim, 'BiasFree')

    def forward(self, x, spatial):
        residual = spatial
        x = self.ffn(x)
        x = x + residual
        x = self.norm(x)
        return x


################################################


class SpectralConv2D(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.kernel_real = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.kernel_imag = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, F):
        real = self.kernel_real(F.real)
        imag = self.kernel_imag(F.imag)
        return torch.complex(real, imag)


class EnhancedSpectralSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, alpha=0.5, mid_channels=4):
        super().__init__()
        self.alpha = alpha
        self.spatial = nn.Sequential(
            nn.Conv2d(1, mid_channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.ReLU(),
            nn.Conv2d(mid_channels, 1, kernel_size=1, bias=False)
        )
        self.spectral = nn.Sequential(
            nn.InstanceNorm2d(1, affine=True),
            nn.Conv2d(1, mid_channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.ReLU(),
            nn.Conv2d(mid_channels, 1, kernel_size=1, bias=False)
        )

    def forward(self, x, x_fft):
        spatial_map = x.mean(dim=1, keepdim=True)
        spec_mag = torch.abs(x_fft).mean(dim=1, keepdim=True)
        w_spatial = self.spatial(spatial_map).sigmoid()
        w_spectral = self.spectral(spec_mag).sigmoid()
        w = self.alpha * w_spatial + (1 - self.alpha) * w_spectral
        return x * w


class LiteSpectralSubbandEnhance(nn.Module):
    def __init__(self, channels, bottleneck_ratio=4, sa_kernel=7):
        super().__init__()
        hidden = channels // bottleneck_ratio
        self.proj1 = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU()
        )
        self.proj2 = nn.Sequential(
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

        self.spec_conv = SpectralConv2D(hidden, kernel_size=3)

        self.spatial_attn_hl = EnhancedSpectralSpatialAttention(kernel_size=sa_kernel)
        self.spatial_attn_lh = EnhancedSpectralSpatialAttention(kernel_size=sa_kernel)
        self.spatial_attn_hh = EnhancedSpectralSpatialAttention(kernel_size=sa_kernel)

    def forward(self, x_hl, x_lh, x_hh):
        B, C, H, W = x_hl.shape

        # 1) 频域变换
        F_hl = torch.fft.fft2(x_hl, norm='ortho')
        F_lh = torch.fft.fft2(x_lh, norm='ortho')
        F_hh = torch.fft.fft2(x_hh, norm='ortho')

        # 2) Bottleneck 实部虚部
        def bottleneck(F):
            real = self.proj1(F.real)
            imag = self.proj1(F.imag)
            return torch.complex(real, imag)

        F_hl_proj = self.spec_conv(bottleneck(F_hl))
        F_lh_proj = self.spec_conv(bottleneck(F_lh))
        F_hh_proj = self.spec_conv(bottleneck(F_hh))

        # 3) 重建
        def recon(F_proj, orig):
            real = self.proj2(F_proj.real)
            imag = self.proj2(F_proj.imag)
            F_recon = torch.complex(real, imag)
            return torch.fft.ifft2(F_recon, norm='ortho').real + orig

        y_hl_freq = recon(F_hl_proj, x_hl)
        y_lh_freq = recon(F_lh_proj, x_lh)
        y_hh_freq = recon(F_hh_proj, x_hh)

        # 4) 空域注意力
        y_hl_spat = self.spatial_attn_hl(x_hl, F_hl)
        y_lh_spat = self.spatial_attn_lh(x_lh, F_lh)
        y_hh_spat = self.spatial_attn_hh(x_hh, F_hh)

        # 5) 融合输出
        y_hl = y_hl_freq + y_hl_spat
        y_lh = y_lh_freq + y_lh_spat
        y_hh = y_hh_freq + y_hh_spat

        return y_hl, y_lh, y_hh


##############################################
def init_weights(m):
    """
    weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        # nn.init.kaiming_normal_(m.weight, mode="fan_out")
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


########################################################
from einops.layers.torch import Rearrange


class PixelAttention(nn.Module):
    def __init__(self, in_channels):
        super(PixelAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.conv1(x)
        attention = self.relu(attention)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)
        return x * attention


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        # 原始CAP结构
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cap = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        # 后接的Conv-ReLU-Conv-Sigmoid结构
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 原始CAP模块
        cap_attention = self.cap(self.avg_pool(x))
        # 后续卷积增强模块
        refined_attention = self.conv_block(cap_attention)
        # 与原输入相乘
        return x * refined_attention


class FusionModule(nn.Module):
    def __init__(self, in_channels):
        super(FusionModule, self).__init__()
        # 初步特征提取
        self.conv_ft = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv_fw = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

        # 融合 & 注意力增强
        self.pixel_attention = PixelAttention(in_channels * 2)
        self.channel_attention = ChannelAttention(in_channels * 2)
        self.conv_attention = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3, padding=1),
            nn.PReLU()
        )

        # 显著性图引导分支（增强关键区域）
        self.saliency = nn.Sequential(
            nn.Conv2d(in_channels * 2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # 权重图生成
        self.weight_t = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.weight_w = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # 结构细节重建引导模块
        self.structure_refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        )

        # 输出卷积
        self.output_conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)

    def forward(self, FT, FW):
        Ft = self.conv_ft(FT)
        Fw = self.conv_fw(FW)

        # 特征拼接
        Fc = torch.cat([Ft, Fw], dim=1)

        # 注意力增强
        Fcp = self.pixel_attention(Fc)
        Fcp = self.channel_attention(Fcp)
        Fcp = self.conv_attention(Fcp)

        # 显著性图增强
        saliency_map = self.saliency(Fcp)
        Fcp = Fcp * saliency_map

        # 权重融合
        Wt = self.weight_t(Fcp)
        Ww = self.weight_w(Fcp)
        fused = Ft * Wt + Fw * Ww

        # 结构细节恢复增强
        fused = fused + self.structure_refine(fused)

        # 输出卷积降维
        out = self.output_conv(fused)
        return out


########################################################
class UpSample(nn.Module):
    def __init__(self, channels, rate=2):
        super(UpSample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels * rate, kernel_size=3, padding=1, bias=False),
                                  nn.PixelShuffle(rate))

    def forward(self, x):
        return self.body(x)


class SeparableConv2D(nn.Module):
    """
    深度可分离卷积3*3
    "Depthwise conv + Pointwise conv"
    """

    def __init__(self, in_channels, out_channels):
        super(SeparableConv2D, self).__init__()
        # Depthwise 过程
        self.DWconv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False)  # 深度可分离卷积
        # Pointwise 过程
        self.PWconv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # x_input = x.cpu().detach().numpy()
        x = self.DWconv(x)
        x = self.PWconv(x)
        # x_2 = x.cpu().detach().numpy()
        return x


###################################图上的注意力机制


##########################################
def PositionalEncoding(d_model, max_len):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe.cuda()


def generate_desired_spiral_order(H, W):
    visited = torch.zeros((H, W), dtype=torch.bool)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
    direction = 0  # Initial direction: right
    x, y = 0, 0  # Starting position
    spiral_order = []

    for _ in range(H * W):
        spiral_order.append(x * W + y)  # Add the current position in flattened index form
        visited[x, y] = True
        next_x, next_y = x + directions[direction][0], y + directions[direction][1]

        if next_x < 0 or next_x >= H or next_y < 0 or next_y >= W or visited[next_x, next_y]:
            direction = (direction + 1) % 4  # Change direction
            next_x, next_y = x + directions[direction][0], y + directions[direction][1]

        x, y = next_x, next_y

    return torch.tensor(spiral_order).cuda()


def prepare_input_keep_4d(x, pe, order):
    B, C, H, W = x.shape
    HW = H * W
    pos = pe[:HW].unsqueeze(0).expand(B, -1, -1)  # [B, HW, C]
    ord = order[:HW].unsqueeze(0).expand(B, -1)  # [B, HW]
    return {0: x, 1: pos, 2: ord}


############################################
class DAVT_fomer(nn.Module):
    # def __init__(self,num_blocks=[1, 2, 3, 4], num_heads=[2,4,8,16], channels=[8, 16, 32, 64], num_refinement=2,patchsizes=[32, 16, 8, 4]):
    def __init__(self, num_blocks=cfg_Ideatormer["num_blocks"], num_heads=cfg_Ideatormer["num_heads"], channels=cfg_Ideatormer["channels"],
                 patchsizes=cfg_Ideatormer["patchsizes"]):
        super(DAVT_fomer, self).__init__()
        self.embed_conv = nn.Conv2d(3, channels[0], kernel_size=3, padding=1, bias=False)

        self.encoder_level1 = TransformerBlock(dim=16, num_heads=2, ffn_expansion_factor=2, bias=False, LayerNorm_type='BiasFree')

        self.encoder_level2 = TransformerBlock(dim=int(16 * 2 ** 1), num_heads=4, ffn_expansion_factor=3, bias=False, LayerNorm_type='BiasFree')

        self.encoder_level3 = TransformerBlock(dim=int(16 * 2 ** 2), num_heads=8, ffn_expansion_factor=4, bias=False, LayerNorm_type='BiasFree')

        self.latent = TransformerBlock(dim=int(16 * 2 ** 3), num_heads=16, ffn_expansion_factor=6, bias=False, LayerNorm_type='BiasFree')

        self.decoder_level3 = TransformerBlock(dim=int(16 * 2 ** 2), num_heads=8, ffn_expansion_factor=4, bias=False, LayerNorm_type='BiasFree')

        self.decoder_level2 = TransformerBlock(dim=int(16 * 2 ** 1), num_heads=4, ffn_expansion_factor=3, bias=False, LayerNorm_type='BiasFree')

        self.decoder_level1 = TransformerBlock(dim=16, num_heads=2, ffn_expansion_factor=2, bias=False, LayerNorm_type='BiasFree')

        self.ups = nn.ModuleList([UpSample(num_ch) for num_ch in channels[1:]])

        self.reduces = nn.ModuleList([nn.Conv2d(channels[i], channels[i - 1], kernel_size=1, bias=False) for i in [1, 2, 3]])

        # 输出层
        self.output1 = SeparableConv2D(channels[0] * 2, 3)
        self.tanh = nn.Tanh()

        # out_dec2 上采样，深度可分离卷积还原
        self.Up2 = UpSample(channels[1])
        self.output2 = SeparableConv2D(channels[0] * 2, 3)

        # out_dec3 上采样，深度可分离卷积还原
        self.Up3 = UpSample(channels[2], rate=4)
        self.output3 = SeparableConv2D(channels[0] * 2, 3)
        # 小波变换
        self.dwt = DWT()
        self.idwt = IDWT()
        self.attentionLow1 = WGCA_Lite(channels=16)
        self.attentionLow2 = WGCA_Lite(channels=32)
        self.attentionLow3 = WGCA_Lite(channels=64)
        self.attentionHigh1 = LiteSpectralSubbandEnhance(channels=16)
        self.attentionHigh2 = LiteSpectralSubbandEnhance(channels=32)
        self.attentionHigh3 = LiteSpectralSubbandEnhance(channels=64)
        self.fusion1 = FusionModule(32)
        self.fusion2 = FusionModule(64)
        self.fusion3 = FusionModule(128)
        # self.attention1 = DWT_Attention(16)
        # self.attention2 = DWT_Attention(32)
        # self.attention3 = DWT_Attention(64)

        self.pe1 = PositionalEncoding(16, 1100000)  # 从48改为16
        self.pe2 = PositionalEncoding(32, 550000)
        self.pe3 = PositionalEncoding(64, 225000)
        self.pe4 = PositionalEncoding(128, 112500)

        self.order256 = generate_desired_spiral_order(256, 256)
        self.order128 = generate_desired_spiral_order(128, 128)
        self.order64 = generate_desired_spiral_order(64, 64)
        self.order32 = generate_desired_spiral_order(32, 32)

        self.conv_expand1 = nn.Conv2d(16, 32, kernel_size=1)
        self.conv_expand2 = nn.Conv2d(32, 64, kernel_size=1)
        self.conv_expand3 = nn.Conv2d(64, 128, kernel_size=1)

    def forward(self, x):
        pos1 = self.pe1
        pos2 = self.pe2
        pos3 = self.pe3
        pos4 = self.pe4

        order256 = self.order256
        order128 = self.order128
        order64 = self.order64
        order32 = self.order32

        fo = self.embed_conv(x)  # C,H,W

        '''
        encoder
        '''

        # encoder stage1 *4
        out_enc1 = self.encoder_level1(prepare_input_keep_4d(fo, pos1, order256))  # C,H,W#注意力，空间+通道16
        out_enc1 = self.encoder_level1(prepare_input_keep_4d(out_enc1, pos1, order256))  # C,H,W#注意力，空间+通道16
        out_enc1 = self.encoder_level1(prepare_input_keep_4d(out_enc1, pos1, order256))  # C,H,W#注意力，空间+通道16
        out_enc1 = self.encoder_level1(prepare_input_keep_4d(out_enc1, pos1, order256))  # C,H,W#注意力，空间+通道16

        x_LL1, x_HL1, x_LH1, x_HH1 = self.dwt(out_enc1)  # 16
        out1 = self.attentionLow1(x_LL1, x_HL1, x_LH1, x_HH1)  # C,H,W#FIM
        # 创建一个卷积层，将通道数从 in_channels 扩展到 in_channels
        ######################################
        # 使用卷积层对 out1 进行变换
        out1 = self.conv_expand1(out1)

        # encoder stage2 *6
        out_enc2 = self.encoder_level2(prepare_input_keep_4d(out1, pos2, order128))  # 2C,H//2,W//2s 32
        out_enc2 = self.encoder_level2(prepare_input_keep_4d(out_enc2, pos2, order128))  # 2C,H//2,W//2s 32
        out_enc2 = self.encoder_level2(prepare_input_keep_4d(out_enc2, pos2, order128))  # 2C,H//2,W//2s 32
        out_enc2 = self.encoder_level2(prepare_input_keep_4d(out_enc2, pos2, order128))  # 2C,H//2,W//2s 32
        out_enc2 = self.encoder_level2(prepare_input_keep_4d(out_enc2, pos2, order128))  # 2C,H//2,W//2s 32
        out_enc2 = self.encoder_level2(prepare_input_keep_4d(out_enc2, pos2, order128))  # 2C,H//2,W//2s 32

        x_LL2, x_HL2, x_LH2, x_HH2 = self.dwt(out_enc2)  # 32
        out2 = self.attentionLow2(x_LL2, x_HL2, x_LH2, x_HH2)  # 2C,sH//2,W//2
        out2 = self.conv_expand2(out2)

        # encoder stage3 *8
        out_enc3 = self.encoder_level3(prepare_input_keep_4d(out2, pos3, order64))  # 4C,H//4,W//4 64
        out_enc3 = self.encoder_level3(prepare_input_keep_4d(out_enc3, pos3, order64))  # 4C,H//4,W//4 64
        out_enc3 = self.encoder_level3(prepare_input_keep_4d(out_enc3, pos3, order64))  # 4C,H//4,W//4 64s
        out_enc3 = self.encoder_level3(prepare_input_keep_4d(out_enc3, pos3, order64))  # 4C,H//4,W//4 64
        out_enc3 = self.encoder_level3(prepare_input_keep_4d(out_enc3, pos3, order64))  # 4C,H//4,W//4 64
        out_enc3 = self.encoder_level3(prepare_input_keep_4d(out_enc3, pos3, order64))  # 4C,H//4,W//4 64
        out_enc3 = self.encoder_level3(prepare_input_keep_4d(out_enc3, pos3, order64))  # 4C,H//4,W//4 64
        out_enc3 = self.encoder_level3(prepare_input_keep_4d(out_enc3, pos3, order64))  # 4C,H//4,W//4 64

        x_LL3, x_HL3, x_LH3, x_HH3 = self.dwt(out_enc3)  # 64
        out3 = self.attentionLow3(x_LL3, x_HL3, x_LH3, x_HH3)  # 2C,sH//2,W//2
        out3 = self.conv_expand3(out3)

        # encoder stage4 *10

        out_enc4 = self.latent(prepare_input_keep_4d(out3, pos4, order32))  # 8C,H//8,W//8 128
        out_enc4 = self.latent(prepare_input_keep_4d(out_enc4, pos4, order32))  # 8C,H//8,W//8 128
        out_enc4 = self.latent(prepare_input_keep_4d(out_enc4, pos4, order32))  # 8C,H//8,W//8 128
        out_enc4 = self.latent(prepare_input_keep_4d(out_enc4, pos4, order32))  # 8C,H//8,W//8 128
        out_enc4 = self.latent(prepare_input_keep_4d(out_enc4, pos4, order32))  # 8C,H//8,W//8 128
        out_enc4 = self.latent(prepare_input_keep_4d(out_enc4, pos4, order32))  # 8C,H//8,W//8 128
        out_enc4 = self.latent(prepare_input_keep_4d(out_enc4, pos4, order32))  # 8C,H//8,W//8 128
        out_enc4 = self.latent(prepare_input_keep_4d(out_enc4, pos4, order32))  # 8C,H//8,W//8 128
        out_enc4 = self.latent(prepare_input_keep_4d(out_enc4, pos4, order32))  # 8C,H//8,W//8 128
        out_enc4 = self.latent(prepare_input_keep_4d(out_enc4, pos4, order32))  # 8C,H//8,W//8 128

        '''
        SkipConnectEnhence
        '''
        enx_HL1, enx_LH1, enx_HH1 = self.attentionHigh1(x_HL1, x_LH1, x_HH1)
        enx_HL2, enx_LH2, enx_HH2 = self.attentionHigh2(x_HL2, x_LH2, x_HH2)
        enx_HL3, enx_LH3, enx_HH3 = self.attentionHigh3(x_HL3, x_LH3, x_HH3)

        '''
        decoder
        '''
        out_enc3_resized = F.interpolate(out3, size=out_enc4.shape[2:], mode='bilinear', align_corners=False)
        out_enc4 = self.fusion3(out_enc4, out_enc3_resized)  # fusion中通道减半了
        if out_enc4.size() != enx_HL3.size():
            out_enc4 = F.interpolate(out_enc4, size=enx_HL3.shape[2:], mode='bilinear', align_corners=False)
        out_dec3up = self.idwt(out_enc4, enx_HL3, enx_LH3, enx_HH3)

        # *8
        fr1 = self.decoder_level3(prepare_input_keep_4d(out_dec3up, pos3, order64))
        fr1 = self.decoder_level3(prepare_input_keep_4d(fr1, pos3, order64))
        fr1 = self.decoder_level3(prepare_input_keep_4d(fr1, pos3, order64))
        fr1 = self.decoder_level3(prepare_input_keep_4d(fr1, pos3, order64))
        fr1 = self.decoder_level3(prepare_input_keep_4d(fr1, pos3, order64))
        fr1 = self.decoder_level3(prepare_input_keep_4d(fr1, pos3, order64))
        fr1 = self.decoder_level3(prepare_input_keep_4d(fr1, pos3, order64))
        fr1 = self.decoder_level3(prepare_input_keep_4d(fr1, pos3, order64))

        out_enc2_resized = F.interpolate(out2, size=fr1.shape[2:], mode='bilinear', align_corners=False)
        out_enc3 = self.fusion2(fr1, out_enc2_resized)
        if out_enc3.size() != enx_HL2.size():
            out_enc3 = F.interpolate(out_enc3, size=enx_HL2.shape[2:], mode='bilinear', align_corners=False)
        out_dec2up = self.idwt(out_enc3, enx_HL2, enx_LH2, enx_HH2)

        fr2 = self.decoder_level2(prepare_input_keep_4d(out_dec2up, pos2, order128))
        fr2 = self.decoder_level2(prepare_input_keep_4d(fr2, pos2, order128))
        fr2 = self.decoder_level2(prepare_input_keep_4d(fr2, pos2, order128))
        fr2 = self.decoder_level2(prepare_input_keep_4d(fr2, pos2, order128))
        fr2 = self.decoder_level2(prepare_input_keep_4d(fr2, pos2, order128))
        fr2 = self.decoder_level2(prepare_input_keep_4d(fr2, pos2, order128))
        fr2 = self.decoder_level2(prepare_input_keep_4d(fr2, pos2, order128))

        out_enc1_resized = F.interpolate(out1, size=fr2.shape[2:], mode='bilinear', align_corners=False)
        out_enc2 = self.fusion1(fr2, out_enc1_resized)
        if out_enc2.size() != enx_HL1.size():
            out_enc2 = F.interpolate(out_enc2, size=enx_HL1.shape[2:], mode='bilinear', align_corners=False)
        out_dec1up = self.idwt(out_enc2, enx_HL1, enx_LH1, enx_HH1)

        fr3 = self.decoder_level1(prepare_input_keep_4d(out_dec1up, pos1, order256))
        fr3 = self.decoder_level1(prepare_input_keep_4d(fr3, pos1, order256))
        fr3 = self.decoder_level1(prepare_input_keep_4d(fr3, pos1, order256))
        fr3 = self.decoder_level1(prepare_input_keep_4d(fr3, pos1, order256))

        '''
        output
        '''

        # output stage
        # fr_resized = F.interpolate(fr, size=fo.shape[2:], mode='bilinear', align_corners=False)

        # 现在 fr_resized 和 fo 尺寸一致，可以拼接
        # output stage1
        fr3_resized = F.interpolate(fr3, size=fo.shape[2:], mode='bilinear', align_corners=False)
        y1 = self.tanh(self.output1(torch.cat([fr3_resized, fo], dim=1))) + x  # 唯一改变就是将底层特征embed_conv给添加了回来

        y1 = torch.clamp(y1, -1, 1, out=None)

        return y1


if __name__ == '__main__':
    torch.manual_seed(123)
    input = torch.randn(1, 3, 160, 160).cuda()
    print("模型输入", input.size())
    net = DAVT_fomer()
    net.cuda()
    net.eval()
    print("#打印模型")
    # print(net)
    out = net(input)
    print("模型输出", out[0].size())
    print("统计模型参数量")
    net1 = DAVT_fomer()
    flops, params = profile(net1, inputs=(torch.randn(1, 3, 128, 128, )).unsqueeze(0))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f GFLOPs, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))

    # input = torch.randn(1, 48, 128*3, 128*4).cuda()
    # print("模型输入", input.size())
    # net = SMHA()
    # net.cuda()
    # print("#打印模型")
    # # print(net)
    # out = net(input)
    # print("模型输出", out.size())
    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameter: ", total)
    print("Number of parameter: M", total / 1e6)

    # summary.summary(net, [(3, 256, 256),(1, 256, 256)])

    # 性能瓶颈分析
    # https://www.jianshu.com/p/1d09e56abbf8
    # Warn-up

    # input = torch.randn(1, 3, 160, 160).cuda()
    # input1 = torch.randn(1, 1, 160, 160).cuda()
    # for _ in range(5):
    #     start = time.time()
    #     outputs = net(input,input1)
    #     torch.cuda.synchronize()
    #     end = time.time()
    #     print('Time:{}ms'.format((end - start) * 1000))
    #
    # with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=True,
    #                                      profile_memory=True) as prof:
    #     outputs = net(input,input1)
    # print(prof.table())
    # prof.export_chrome_trace('sss/Ideatormer_profile.json')
