"""
traj_guidance.py

轨迹侧(condition-side)像素引导，用于 FLUX.2-klein 的 img2img/编辑式条件。

klein 把参考图(雾图)编码成 latent 后，经过  vae.encode -> patchify(2x2) -> BN 归一化 -> pack
变成 token，与待去噪 token 在序列维上拼接，DiT 通过联合注意力读取。
这里在 "BN 归一化之后、pack 之前" 给参考 latent 加一个来自原始像素的门控残差：

    z_tilde = z + lam * sigmoid(g) * W_p( P(x_hz)[1/16] )

    z      : (B, 128, H/16, W/16)  已 patchify + BN 的参考 latent（klein 原有）
    P      : 可训练的像素金字塔编码器，输出与 token 网格(1/16)对齐的特征
    W_p    : 1x1 卷积，零初始化   -> 训练起点与原模型逐比特一致
    g      : 可学习标量门，初始 0 (sigmoid=0.5)
    lam    : 固定强度系数

训练和推理两条路径必须数学上完全等价：
    训练:  训练脚本里在 BN 之后直接调用 traj(z, pixel)
    推理:  attach_trajectory_guidance(pipe, traj) 包装 pipe._encode_vae_image，
           其收到的 image 就是预处理后 [-1,1] 的参考图，与训练时喂给 VAE 的是同一张

编码器的 stem/down1/down2 与 skip_pyramid.PixelPyramid 的层名一致（只多了 down3/down4），
所以以后可以用这里训练好的权重去初始化解码侧 skip 的共享编码器。

本文件不依赖仓库里的其他模块。
"""

import torch
import torch.nn as nn


class PixelPyramid16(nn.Module):
    """输入 [-1,1] 的图像，输出 {下采样倍数: 特征}，倍数为 1/2/4/8/16。"""

    def __init__(self, in_ch=3, widths=(64, 128, 128), c_tok=256):
        super().__init__()
        c0, c1, c2 = widths
        self.widths, self.c_tok = tuple(widths), c_tok

        def block(cin, cout, stride):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride, 1), nn.SiLU(),
                nn.Conv2d(cout, cout, 3, 1, 1), nn.SiLU(),
            )

        self.stem = block(in_ch, c0, 1)   # 1/1
        self.down1 = block(c0, c1, 2)     # 1/2
        self.down2 = block(c1, c2, 2)     # 1/4
        self.down3 = block(c2, c2, 2)     # 1/8
        self.down4 = block(c2, c_tok, 2)  # 1/16  <- 与 klein token 网格对齐

    def forward(self, x):
        f1 = self.stem(x)
        f2 = self.down1(f1)
        f4 = self.down2(f2)
        f8 = self.down3(f4)
        f16 = self.down4(f8)
        return {1: f1, 2: f2, 4: f4, 8: f8, 16: f16}


class TrajectoryGuidance(nn.Module):
    def __init__(self, z_ch=128, widths=(64, 128, 128), p_ch=256, lam=1.0):
        super().__init__()
        self.config = dict(z_ch=z_ch, widths=list(widths), p_ch=p_ch, lam=lam)
        self.lam = lam
        self.pyramid = PixelPyramid16(widths=widths, c_tok=p_ch)
        self.Wp = nn.Conv2d(p_ch, z_ch, 1)
        nn.init.zeros_(self.Wp.weight)
        nn.init.zeros_(self.Wp.bias)
        self.g = nn.Parameter(torch.zeros(1))

    def correction(self, x_hz):
        return self.lam * torch.sigmoid(self.g) * self.Wp(self.pyramid(x_hz)[16])

    def forward(self, z, x_hz, return_corr=False):
        """z: (B,128,H/16,W/16) patchify+BN 之后的参考 latent；x_hz: (B,3,H,W) in [-1,1]
        return_corr=True 时额外返回修正量本身（训练时用来记录 ‖修正‖/‖z‖）"""
        corr = self.correction(x_hz)
        out = z + corr.to(z.dtype)
        return (out, corr) if return_corr else out


def save_traj(traj, path):
    torch.save({"config": traj.config, "state_dict": traj.state_dict()}, path)


def load_traj(path, device="cpu"):
    ckpt = torch.load(path, map_location="cpu")
    traj = TrajectoryGuidance(**ckpt["config"])
    traj.load_state_dict(ckpt["state_dict"])  # 严格加载
    return traj.to(device).eval()


def attach_trajectory_guidance(pipe, traj):
    """
    包装 Flux2KleinPipeline._encode_vae_image：先走原实现(VAE 编码 + patchify + BN)，再加像素修正。
    pipe.prepare_image_latents 是以关键字参数 image=/generator= 调用它的，所以这里的形参名要保持一致。
    """
    if getattr(pipe, "_traj_guidance", None) is not None:
        raise RuntimeError("这个 pipeline 已经挂过 trajectory guidance")
    orig = pipe._encode_vae_image  # 绑定方法

    def wrapped(image, generator):
        z = orig(image=image, generator=generator)
        traj.to(device=image.device)  # 兼容 enable_model_cpu_offload：跟着 image 的执行设备走
        return traj(z.float(), image.float()).to(z.dtype)

    pipe._encode_vae_image = wrapped
    pipe._traj_guidance = traj
    return pipe