"""
skip_pyramid.py

用"共享的多尺度像素编码器"替换 skip_gated.py 里"每层各自 interpolate + 两层卷积"的做法，
融合部分（零初始化 fuse_conv + 逐像素门控）保持不变。不修改 skip_gated.py / skip_modules.py。

与 GatedSkipFusion 的区别
--------------------------
    旧版(每层独立):  cond_feat_i = cond_conv_i( bilinear(hazy -> 该层分辨率) )
                     3x3 -> SiLU -> 3x3，感受野很小，低分辨率层还带无抗锯齿的下采样
    新版(共享金字塔): {f1, f2, f4} = PixelPyramid(hazy)         # 步长卷积逐级下采样，一次前向
                     cond_feat_i = proj_i( f_{该层分辨率} )     # 1x1 对齐到该层 decoder 通道数
    融合:            out = dec + gate * fuse_conv(cat[dec, cond_feat])    (同旧版)

分辨率对应关系
--------------
Flux2 VAE 的 decoder 有 4 个 up_block，前 3 个各上采样 2x，最后一个不上采样；latent 是图像的 1/8。
所以 up_block i 之后的特征相对原图的下采样倍数是  2 ** max(0, n_blocks - 2 - i)：
    level 0 -> 1/4,  level 1 -> 1/2,  level 2 -> 1/1,  level 3 -> 1/1
level 2 和 level 3 共用同一份全分辨率特征 f1。

checkpoint 键
-------------
    encoder.stem.* / encoder.down1.* / encoder.down2.*     共享编码器
    levels.<i>.proj / fuse_conv / gate_head.*              每层融合
有 "encoder." 前缀即视为金字塔版。编码器宽度、门控隐藏层宽度从权重形状自动推断，
所以推理/评测只需要 checkpoint 路径（见 load_skip_auto）。

旧版(per-level)的 checkpoint 不能热启动到这个结构：旧 fuse_conv 是按旧 cond_feat 的分布训练的，
换了特征提取器之后这些权重不再对应。金字塔版需要从零训练。
"""

import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from skip_gated import load_skip, patch_decoder_multi


def level_down_factor(level, n_blocks):
    """up_block `level` 之后的特征，相对原图的下采样倍数。"""
    return 2 ** max(0, n_blocks - 2 - level)


class PixelPyramid(nn.Module):
    """输入 [-1,1] 的雾图，输出 {下采样倍数: 特征}，倍数为 1 / 2 / 4。"""

    def __init__(self, in_ch=3, widths=(64, 128, 128)):
        super().__init__()
        c0, c1, c2 = widths
        self.widths = tuple(widths)
        self.stem = nn.Sequential(                                  # 1/1
            nn.Conv2d(in_ch, c0, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(c0, c0, 3, 1, 1), nn.SiLU(),
        )
        self.down1 = nn.Sequential(                                 # 1/2
            nn.Conv2d(c0, c1, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(c1, c1, 3, 1, 1), nn.SiLU(),
        )
        self.down2 = nn.Sequential(                                 # 1/4
            nn.Conv2d(c1, c2, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(c2, c2, 3, 1, 1), nn.SiLU(),
        )
        self.out_channels = {1: c0, 2: c1, 4: c2}

    def forward(self, x):
        f1 = self.stem(x)
        f2 = self.down1(f1)
        f4 = self.down2(f2)
        return {1: f1, 2: f2, 4: f4}


class PyramidGatedFusion(nn.Module):
    """与 GatedSkipFusion 相同的融合方式，只是 cond_feat 来自共享金字塔(1x1 对齐通道)。"""

    def __init__(self, decoder_channels, feat_channels, gate_hidden=32, gate_bias_init=0.0):
        super().__init__()
        self.proj = nn.Conv2d(feat_channels, decoder_channels, 1)
        self.fuse_conv = nn.Conv2d(decoder_channels * 2, decoder_channels, 3, padding=1)
        nn.init.zeros_(self.fuse_conv.weight)
        nn.init.zeros_(self.fuse_conv.bias)

        self.gate_head = nn.Sequential(
            nn.Conv2d(decoder_channels * 2, gate_hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(gate_hidden, 1, 3, padding=1),
        )
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(self.gate_head[-1].bias, float(gate_bias_init))

        self.last_gate = None  # 最近一次forward的gate(detach)，仅用于日志

    def forward(self, dec_feat, pyr_feat):
        if pyr_feat.shape[-2:] != dec_feat.shape[-2:]:  # 正常情况下尺寸一致；输入不是16的倍数时兜底
            pyr_feat = F.interpolate(pyr_feat, size=dec_feat.shape[-2:], mode="bilinear", align_corners=False)
        cond_feat = self.proj(pyr_feat)
        x = torch.cat([dec_feat, cond_feat], dim=1)
        delta = self.fuse_conv(x)
        gate = torch.sigmoid(self.gate_head(x))
        self.last_gate = gate.detach()
        return dec_feat + gate * delta


class PyramidSkip(nn.Module):
    """参数容器：共享编码器 + 各层融合。真正的调用在 patch_decoder_pyramid 里。"""

    def __init__(self, encoder, level_modules, n_blocks):
        super().__init__()
        self.encoder = encoder
        self.levels = nn.ModuleDict({str(i): m for i, m in sorted(level_modules.items())})
        self.n_blocks = n_blocks

    @property
    def level_ids(self):
        return [int(k) for k in self.levels.keys()]

    @property
    def last_gates(self):
        return {int(k): m.last_gate for k, m in self.levels.items() if getattr(m, "last_gate", None) is not None}


def build_pyramid_skip(levels, block_out_channels, widths=(64, 128, 128), gate_hidden=32, gate_bias_init=0.0):
    rev = list(reversed(block_out_channels))  # up_block i 的输出通道数
    n = len(rev)
    enc = PixelPyramid(widths=widths)
    mods = {}
    for i in levels:
        if not 0 <= i < n:
            raise ValueError(f"skip level {i} 超出范围，up_block 共 {n} 个(0..{n - 1})")
        d = level_down_factor(i, n)
        mods[i] = PyramidGatedFusion(rev[i], enc.out_channels[d], gate_hidden, gate_bias_init)
    return PyramidSkip(enc, mods, n)


def patch_decoder_pyramid(decoder, skip):
    """
    复刻 diffusers Decoder.forward：金字塔只算一次，然后在每个已配置 level 的 up_block 之后融合。
    保留 gradient checkpointing 分支(编码器本身不放进 checkpoint)；cond_img=None 时等价于原始 decoder。
    """
    levels = {int(k): m for k, m in skip.levels.items()}
    n_blocks = skip.n_blocks

    def patched_forward(self, sample, latent_embeds=None, cond_img=None):
        feats = skip.encoder(cond_img) if (cond_img is not None and levels) else None
        sample = self.conv_in(sample)
        use_ckpt = torch.is_grad_enabled() and self.gradient_checkpointing

        if use_ckpt:
            sample = self._gradient_checkpointing_func(self.mid_block, sample, latent_embeds)
        else:
            sample = self.mid_block(sample, latent_embeds)

        for i, up_block in enumerate(self.up_blocks):
            if use_ckpt:
                sample = self._gradient_checkpointing_func(up_block, sample, latent_embeds)
            else:
                sample = up_block(sample, latent_embeds)
            if feats is not None and i in levels:
                sample = levels[i](sample, feats[level_down_factor(i, n_blocks)])

        if latent_embeds is None:
            sample = self.conv_norm_out(sample)
        else:
            sample = self.conv_norm_out(sample, latent_embeds)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        return sample

    decoder.forward = types.MethodType(patched_forward, decoder)
    return decoder


# ---------------------------------------------------------------------------
# checkpoint 识别 / 加载 / 统一入口
# ---------------------------------------------------------------------------
def is_pyramid_state_dict(sd):
    return any(k.startswith("encoder.") for k in sd)


def pyramid_config_from_state_dict(sd):
    """从权重形状推断 (levels, widths, gate_hidden)。"""
    widths = (
        sd["encoder.stem.0.weight"].shape[0],
        sd["encoder.down1.0.weight"].shape[0],
        sd["encoder.down2.0.weight"].shape[0],
    )
    levels = sorted({int(k.split(".")[1]) for k in sd if k.startswith("levels.")})
    gate_hidden = sd[f"levels.{levels[0]}.gate_head.0.weight"].shape[0]
    return levels, widths, gate_hidden


def load_pyramid_skip(path, block_out_channels, device, dtype=torch.float32):
    sd = torch.load(path, map_location="cpu")
    levels, widths, gate_hidden = pyramid_config_from_state_dict(sd)
    skip = build_pyramid_skip(levels, list(block_out_channels), widths=widths, gate_hidden=gate_hidden)
    skip.load_state_dict(sd)  # 严格加载
    desc = f"pyramid(widths={list(widths)}) " + ", ".join(f"level{i}:gated" for i in levels)
    return skip.to(device, dtype).eval(), desc


def load_skip_auto(path, block_out_channels, device, dtype=torch.float32):
    """按 checkpoint 自动选择：金字塔版 / 旧版(per-level, plain 或 gated)。返回 (skip, 描述)。"""
    sd = torch.load(path, map_location="cpu")
    if is_pyramid_state_dict(sd):
        return load_pyramid_skip(path, block_out_channels, device, dtype)
    return load_skip(path, block_out_channels, device, dtype)


def patch_decoder_auto(decoder, skip):
    if isinstance(skip, PyramidSkip):
        return patch_decoder_pyramid(decoder, skip)
    return patch_decoder_multi(decoder, skip)