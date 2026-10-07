"""
skip_gated.py

逐像素门控版 skip 融合模块、多层 skip 容器、以及按 checkpoint 自动识别结构的加载函数。
不修改 skip_modules.py（里面的 LastLevelSkipFusion 照常使用）。

GatedSkipFusion 与 LastLevelSkipFusion 的区别：

    原版:   out = dec_feat + delta
    门控版: out = dec_feat + gate * delta        gate ∈ (0,1)，形状 (B,1,H,W)，逐像素

初始化：
  - fuse_conv 零初始化(同原版) -> delta=0 -> step 0 输出与无skip逐比特一致。
  - gate_head 最后一层权重零初始化，bias = gate_bias_init：
      从头训练:  0   (gate=0.5，delta 本来是0，门控此时没有影响)
      从已训练的skip热启动: 3 (gate≈0.95)，让已训练好的修正量起步时近乎直通。

多层 skip（MultiLevelSkip + patch_decoder_multi）：
  可以在任意几个 up_block 之后各挂一个独立的 skip，level i 表示第 i 个 up_block 之后。
  Flux2 VAE 的 block_out_channels=[128,256,512,512]，up_block 输出通道为 [512,512,256,128]，
  其中 level 2、3 的输出已经是全分辨率(1/1)，level 0、1 分别是 1/4、1/2 分辨率。
  level 3(最后一个) 与原来"只在最后加一次"的位置完全相同。

checkpoint 格式：
  新格式的键为 levels.<i>.<...>；旧版单点 checkpoint(键为 cond_conv.* / fuse_conv.* /
  gate_head.*)一律视为挂在最后一个 up_block 之后(level 3)，可直接用于热启动或推理。
"""

import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from skip_modules import LastLevelSkipFusion


class GatedSkipFusion(nn.Module):
    def __init__(self, decoder_channels=128, cond_channels=3, hidden=64, gate_hidden=32, gate_bias_init=0.0):
        super().__init__()
        self.cond_conv = nn.Sequential(
            nn.Conv2d(cond_channels, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, decoder_channels, 3, padding=1),
        )
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

        self.last_gate = None  # 最近一次forward的gate(detach)，仅用于日志/可视化

    def forward(self, dec_feat, cond_img):
        cond_resized = F.interpolate(cond_img, size=dec_feat.shape[-2:], mode="bilinear", align_corners=False)
        cond_feat = self.cond_conv(cond_resized)
        x = torch.cat([dec_feat, cond_feat], dim=1)
        delta = self.fuse_conv(x)
        gate = torch.sigmoid(self.gate_head(x))
        self.last_gate = gate.detach()
        return dec_feat + gate * delta


def build_skip(kind, decoder_channels, gate_bias_init=0.0):
    """kind: 'plain'(原版 LastLevelSkipFusion) 或 'gated'"""
    if kind == "gated":
        return GatedSkipFusion(decoder_channels=decoder_channels, gate_bias_init=gate_bias_init)
    if kind == "plain":
        return LastLevelSkipFusion(decoder_channels=decoder_channels)
    raise ValueError(f"未知的 skip 类型: {kind}")


# ---------------------------------------------------------------------------
# 多层 skip
# ---------------------------------------------------------------------------
class MultiLevelSkip(nn.Module):
    """只是参数容器(用于优化器/state_dict/to/eval)，真正的调用发生在 patch_decoder_multi 里。"""

    def __init__(self, level_modules):
        super().__init__()
        self.levels = nn.ModuleDict({str(i): m for i, m in sorted(level_modules.items())})

    @property
    def level_ids(self):
        return [int(k) for k in self.levels.keys()]

    @property
    def last_gates(self):
        """{level: 最近一次forward的gate}，只包含gated的层"""
        return {int(k): m.last_gate for k, m in self.levels.items() if getattr(m, "last_gate", None) is not None}


def build_multi_skip(kind, levels, block_out_channels, gate_bias_init=0.0):
    rev = list(reversed(block_out_channels))  # up_block i 的输出通道数
    mods = {}
    for i in levels:
        if not 0 <= i < len(rev):
            raise ValueError(f"skip level {i} 超出范围，up_block 共 {len(rev)} 个(0..{len(rev) - 1})")
        mods[i] = build_skip(kind, rev[i], gate_bias_init)
    return MultiLevelSkip(mods)


def patch_decoder_multi(decoder, skip):
    """
    复刻 diffusers Decoder.forward，在每个已配置 level 的 up_block 之后调用对应的 skip。
    保留原本的 gradient checkpointing 分支。cond_img=None 时等价于原始 decoder。
    """
    levels = {int(k): m for k, m in skip.levels.items()}

    def patched_forward(self, sample, latent_embeds=None, cond_img=None):
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
            if cond_img is not None and i in levels:
                sample = levels[i](sample, cond_img)

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
# checkpoint 格式处理（纯字典操作）
# ---------------------------------------------------------------------------
def to_multi_state_dict(sd, final_level):
    """旧版单点checkpoint(无levels.前缀)视为挂在最后一个up_block之后；新格式原样返回。"""
    if any(k.startswith("levels.") for k in sd):
        return sd
    return {f"levels.{final_level}.{k}": v for k, v in sd.items()}


def ckpt_levels(sd):
    return sorted({int(k.split(".")[1]) for k in sd if k.startswith("levels.")})


def skip_kind_from_state_dict(sd):
    return "gated" if any(k.startswith("gate_head.") or ".gate_head." in k for k in sd) else "plain"


def unexpected_load_issues(missing, unexpected, new_levels):
    """
    strict=False 加载之后，哪些缺失/多余的键是不该出现的。
    允许缺失的只有：(1)本次新增的 level 的全部参数；(2)门控头 gate_head.*(从plain权重热启动gated)。
    多余的键(checkpoint里有、当前结构里没有)一律视为问题，比如 checkpoint 带 level 3
    却只配置了 --skip_levels 2。
    """
    def allowed(k):
        return int(k.split(".")[1]) in set(new_levels) or ".gate_head." in k

    return [k for k in missing if not allowed(k)] + list(unexpected)


def load_skip(path, block_out_channels, device, dtype=torch.float32):
    """
    推理/评测用：按checkpoint自动识别 level 和每层的类型(plain/gated)，严格加载。
    返回 (MultiLevelSkip, 描述字符串)。旧版单点checkpoint自动当作 level=最后一层。
    """
    block_out_channels = list(block_out_channels)
    sd = to_multi_state_dict(torch.load(path, map_location="cpu"), len(block_out_channels) - 1)
    rev = list(reversed(block_out_channels))
    mods, kinds = {}, {}
    for i in ckpt_levels(sd):
        sub = {k for k in sd if k.startswith(f"levels.{i}.")}
        kinds[i] = "gated" if any(".gate_head." in k for k in sub) else "plain"
        mods[i] = build_skip(kinds[i], rev[i])
    skip = MultiLevelSkip(mods)
    skip.load_state_dict(sd)
    desc = ", ".join(f"level{i}:{kinds[i]}" for i in sorted(kinds))
    return skip.to(device, dtype).eval(), desc