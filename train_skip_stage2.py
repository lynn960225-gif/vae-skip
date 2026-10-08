"""
train_skip_stage2.py

CUDA_VISIBLE_DEVICES=3 nohup python -u train_skip_stage2.py \
        --base_model black-forest-labs/FLUX.2-klein-4B \
        --hazy_dir /data/storage/users/yliu/datasets/dehazing/NH-HAZE/train/hazy --gt_dir /data/storage/users/yliu/datasets/dehazing/NH-HAZE/train/gt \
        --latent_dirs ./infer_out_nhhaze_skip_0/latents ./infer_out_nhhaze_skip_1/latents ./infer_out_nhhaze_skip_2/latents ./infer_out_nhhaze_skip_3/latents ./infer_out_nhhaze_skip_4/latents ./infer_out_nhhaze_skip_5/latents \
        --train_ids 1-45 --val_ids 46-50 \
        --skip_type gated \
        --gate_bias_init 0 \
        --skip_levels 1 2 3 \
        --init_ckpt /home/yliu/code/vae-skip-experiment/checkpoints_stage2_nhhaze/skip_stage2_best.pt \
        --output_dir ./checkpoints_stage2_nhhaze_skip_123_gate \
        > logs/train_stage2_nhhaze_skip_123_gate.log &


CUDA_VISIBLE_DEVICES=2 nohup python -u train_skip_stage2.py \
        --base_model black-forest-labs/FLUX.2-klein-4B \
        --hazy_dir /data/storage/users/yliu/datasets/dehazing/NH-HAZE/train/hazy --gt_dir /data/storage/users/yliu/datasets/dehazing/NH-HAZE/train/gt \
        --latent_dirs ./infer_out_0/latents ./infer_out_1/latents ./infer_out_2/latents ./infer_out_3/latents ./infer_out_4/latents ./infer_out_5/latents \
        --train_ids 1-45 --val_ids 46-50 \
        --skip_arch pyramid \
        --skip_levels 0 1 2 3 \
        --output_dir ./ckpt_pyr_0123 \
        > logs/train_stage2_nhhaze_pyr_0123.log &


# 新版
python -u train_skip_stage2.py ... --skip_arch pyramid --skip_levels 0 1 2 3 --output_dir ./ckpt_pyr_0123
# 对照：旧结构，同样从零训，同样的 steps 和 seed
python -u train_skip_stage2.py ... --skip_arch perlevel --skip_type gated --skip_levels 0 1 2 3 --output_dir ./ckpt_perlevel_0123


CUDA_VISIBLE_DEVICES=1 nohup python -u train_skip_stage2.py \
        --base_model black-forest-labs/FLUX.2-klein-4B \
        --hazy_dir /data/storage/users/yliu/datasets/RESIDE_Standard/ITS_train/hazy --gt_dir /data/storage/users/yliu/datasets/RESIDE_Standard/ITS_train/clear \
        --latent_dirs ./infer_out_its_0/latents \
        --train_ids 1-1380 --val_ids 1381-1399 \
        --output_dir ./checkpoints_stage2_its \
        > logs/train_stage2_its.log &


CUDA_VISIBLE_DEVICES=3 nohup python -u train_skip_stage2.py \
        --base_model black-forest-labs/FLUX.2-klein-4B \
        --hazy_dir /data/storage/users/yliu/datasets/dehazing/Dense_Haze/train/hazy --gt_dir /data/storage/users/yliu/datasets/dehazing/Dense_Haze/train/gt \
        --latent_dirs ./infer_out_dense_0/latents ./infer_out_dense_1/latents ./infer_out_dense_2/latents ./infer_out_dense_3/latents ./infer_out_dense_4/latents ./infer_out_dense_5/latents \
        --train_ids 1-45 --val_ids 46-50 \
        --skip_levels 0 1 2 3 \
        --skip_arch pyramid \
        --output_dir ./ckpt_pyr_densehaze_0123 \
        > logs/train_stage2_densehaze_pyr_0123.log &

"""

import argparse
import glob
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLFlux2
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
from PIL import Image

from eval_infer import gt_candidates, index_dir, leading_int, original_region, parse_ids
from infer_dehaze import preprocess_cond, vae_scale_factor_from_vae
from skip_gated import (build_multi_skip, ckpt_levels, load_skip, patch_decoder_multi,
                        skip_kind_from_state_dict, to_multi_state_dict, unexpected_load_issues)
from skip_pyramid import (build_pyramid_skip, is_pyramid_state_dict, load_skip_auto,
                          patch_decoder_auto, pyramid_config_from_state_dict)
from skimage.metrics import structural_similarity
from ssim_decompose import ssim_decompose


def to_u8(t):
    return ((t + 1) / 2 * 255).round().clamp(0, 255).to(torch.uint8)


def from_u8(u, device):
    return u.to(device=device, dtype=torch.float32) / 127.5 - 1


def load_samples(args, proc, scale, ids):
    """
    读取 (hazy, GT, latent) 样本。latent 不要求齐全：
      - 某张图在所有 latent_dirs 里都没有 .pt            -> 跳过这张图
      - 只在部分 seed 目录里有                           -> 只用存在的那几份
      - .pt 文件损坏/半截(生成被中断留下的)、无法读取    -> 跳过这一份，不会让进程崩掉
      - latent 尺寸与图像对不上                          -> 跳过这一份
    所有跳过只在最后汇总打印一次，不再每张图一行警告。
    """
    gt_index = index_dir(args.gt_dir)
    hazy_files = sorted(
        glob.glob(os.path.join(args.hazy_dir, "*.png")) + glob.glob(os.path.join(args.hazy_dir, "*.jpg"))
    )
    n_dirs = len(args.latent_dirs)
    samples = []
    n_in_range = 0
    n_partial = 0  # 只有部分seed有可用latent、但至少有一份的图片数
    skipped = {
        "所有seed都没有latent": [],
        "latent损坏/无法读取": [],
        "latent尺寸与图像不一致": [],
        "图片读取/预处理失败": [],
    }

    for path in hazy_files:
        stem = os.path.splitext(os.path.basename(path))[0]
        gt_path = next((gt_index[c] for c in gt_candidates(stem, args.dataset, "", "") if c in gt_index), None)
        if gt_path is None:
            continue
        gid = leading_int(os.path.splitext(os.path.basename(gt_path))[0])
        if gid not in ids:
            continue
        n_in_range += 1

        lat_paths = [p for p in (os.path.join(d, stem + ".pt") for d in args.latent_dirs) if os.path.exists(p)]
        if not lat_paths:
            skipped["所有seed都没有latent"].append(stem)
            continue

        # 先读latent(很小)，全部读不出来就不用做后面耗时的图片预处理
        zs = []
        for lp in lat_paths:
            try:
                zs.append(torch.load(lp, map_location="cpu"))
            except Exception as e:
                skipped["latent损坏/无法读取"].append(f"{lp} [{type(e).__name__}]")
        if not zs:
            continue

        try:
            hazy = to_u8(preprocess_cond(proc, scale, Image.open(path).convert("RGB")))
            gt = to_u8(preprocess_cond(proc, scale, Image.open(gt_path).convert("RGB")))

            # 原生分辨率GT：裁出pred(pipeline分辨率)对应GT原图里的居中区域，保留原生细节，
            # 转成[0,1] tensor缓存下来，供 --loss_res native 时直接用，不用每个step都重新读盘/裁剪
            pred_size_wh = (hazy.shape[-1], hazy.shape[-2])  # (w, h)
            gt_native_img = Image.open(gt_path).convert("RGB")
            if gt_native_img.size != pred_size_wh:
                gt_native_img = gt_native_img.crop(original_region(pred_size_wh, gt_native_img.size))
            gt_native_01 = torch.from_numpy(
                np.asarray(gt_native_img, dtype=np.float32) / 255.0
            ).permute(2, 0, 1).unsqueeze(0)
        except Exception as e:
            skipped["图片读取/预处理失败"].append(f"{stem} [{type(e).__name__}]")
            continue

        n_ok = 0
        for z in zs:
            if z.shape[-2] * scale != hazy.shape[-2] or z.shape[-1] * scale != hazy.shape[-1]:
                skipped["latent尺寸与图像不一致"].append(stem)
                continue
            samples.append({
                "name": stem, "z": z, "hazy": hazy, "gt": gt,
                "gt_path": gt_path, "gt_native_01": gt_native_01,
            })
            n_ok += 1
        if 0 < n_ok < n_dirs:
            n_partial += 1

    print(f"[load_samples] 编号范围内的图片 {n_in_range} 张 -> 有效 (图, latent) 样本 {len(samples)} 个"
          f"（latent_dirs 共 {n_dirs} 个，满配为 {n_in_range * n_dirs} 个）")
    if n_partial:
        print(f"  其中 {n_partial} 张图只有部分seed的latent，已只用存在的那几份")
    for reason, items in skipped.items():
        if items:
            ex = ", ".join(items[:5]) + (" ..." if len(items) > 5 else "")
            print(f"  跳过 [{reason}] {len(items)} 项，例如: {ex}")
    return samples


def run(vae, z, hazy_u8, use_skip, device):
    z = z.to(device, torch.float32)
    z_in = vae.post_quant_conv(z) if vae.post_quant_conv is not None else z
    cond = from_u8(hazy_u8, device) if use_skip else None
    return vae.decoder(z_in, cond_img=cond)


def sk_style_ssim(pred, gt, win_size=7, data_range=1.0, K1=0.01, K2=0.03):
    """
    可微分实现，严格对齐 skimage.metrics.structural_similarity 的默认行为：
      - 7x7均匀窗口(不是高斯)，每个通道独立算，用同一组卷积对所有通道一起算等价于
        "每通道算完再平均"(像素总数相同，联合平均=分通道平均再平均)
      - use_sample_covariance=True 的 N/(N-1) 协方差修正(skimage默认开启)
      - 裁掉窗口半径宽度的边框再平均，避免边界统计不可靠(skimage的做法)
    用 F.avg_pool2d(zero-pad) 算'same'尺寸的滑动窗口均值，再裁边——裁掉的内部区域
    数值跟skimage内部用reflect-pad再裁边完全一致（裁剩的像素，窗口没碰到过padding区域，
    跟padding用什么模式无关），是目前能做到的最接近skimage的可微分版本，不是近似凑合。
    pred, gt: (B,C,H,W)，建议传入[0,1]范围配合data_range=1.0。
    返回: 标量ssim值(越大越好)，可直接用 1 - sk_style_ssim(...) 当loss。
    """
    C1, C2 = (K1 * data_range) ** 2, (K2 * data_range) ** 2
    pad = win_size // 2
    cov_norm = win_size ** 2 / (win_size ** 2 - 1)

    def uf(x):
        return F.avg_pool2d(x, kernel_size=win_size, stride=1, padding=pad)

    mu_x, mu_y = uf(pred), uf(gt)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x2 = cov_norm * (uf(pred * pred) - mu_x2)
    sigma_y2 = cov_norm * (uf(gt * gt) - mu_y2)
    sigma_xy = cov_norm * (uf(pred * gt) - mu_xy)

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
    if pad > 0:
        ssim_map = ssim_map[..., pad:-pad, pad:-pad]
    return ssim_map.mean()


def make_batches(samples, batch_size):
    """按latent空间尺寸分组后再切batch，确保torch.cat时同一batch内形状一致
    （正常情况下同一数据集所有图裁剪后尺寸应该相同，分组只是防御性的，
    真遇到不同尺寸也不会崩，只是那一组内batch会分别成组，不会混在一起）"""
    buckets = {}
    for s in samples:
        buckets.setdefault(tuple(s["z"].shape[-2:]), []).append(s)
    batches = []
    for group in buckets.values():
        random.shuffle(group)
        batches.extend(group[i : i + batch_size] for i in range(0, len(group), batch_size))
    random.shuffle(batches)
    return batches


@torch.no_grad()
def evaluate(vae, proc, samples, device, use_skip, metrics=("psnr",)):
    """metrics 可含 psnr / ssim / L / C / S。
    L/C/S 复用 ssim_decompose.py（灰度、11x11局部窗口），衡量的是"结构对不对齐"这个维度，
    跟 PSNR/整体SSIM 是互补信息，不是同一件事的不同写法。

    对齐方式：GT 用 gt_path 重新以原生分辨率加载，通过 original_region() 反推 pred 对应
    GT 里的居中区域，裁出该区域后把 pred 放大到这个尺寸再比较——与 evaluate_batch.py /
    infer_dehaze.py(--eval_res original) 完全一致。之前这里直接拿 load_samples() 里被
    preprocess_cond 裁小过的 s["gt"] 跟 pred 比较，等价于"pipeline模式"：GT自己的细节被
    一起降采样掉了，局部错位/重影这类问题会被这个更粗糙的尺度"糊"掉，PSNR/SSIM偏乐观，
    跟其他脚本的评测口径不一致，是之前"train_skip_stage2内部显示提升、但batch_inference_
    skip+evaluate_batch显示下降"这个矛盾的根源。
    """
    need_np = any(m in metrics for m in ("ssim", "L", "C", "S"))
    acc = {m: [] for m in metrics}
    for s in samples:
        out = run(vae, s["z"], s["hazy"], use_skip, device)
        pred_img = proc.postprocess(out.float(), output_type="pil")[0]
        gt_img = Image.open(s["gt_path"]).convert("RGB")

        if pred_img.size != gt_img.size:
            region = original_region(pred_img.size, gt_img.size)
            gt_img = gt_img.crop(region)
            pred_img = pred_img.resize(gt_img.size, Image.BICUBIC)

        pred_np = np.asarray(pred_img)
        gt_np = np.asarray(gt_img)

        if "psnr" in metrics:
            mse = float(np.mean((pred_np.astype(np.float64) - gt_np.astype(np.float64)) ** 2))
            acc["psnr"].append(10 * np.log10(255.0 ** 2 / mse) if mse > 0 else float("inf"))

        if need_np:
            if "ssim" in metrics:
                acc["ssim"].append(structural_similarity(pred_np, gt_np, channel_axis=2, data_range=255))
            if any(m in metrics for m in ("L", "C", "S")):
                pred_gray = np.asarray(Image.fromarray(pred_np).convert("L"), dtype=np.float64)
                gt_gray = np.asarray(Image.fromarray(gt_np).convert("L"), dtype=np.float64)
                l_map, c_map, s_map, *_ = ssim_decompose(pred_gray, gt_gray)
                if "L" in metrics:
                    acc["L"].append(float(l_map.mean()))
                if "C" in metrics:
                    acc["C"].append(float(c_map.mean()))
                if "S" in metrics:
                    acc["S"].append(float(s_map.mean()))

    return {m: float(np.mean(v)) for m, v in acc.items()}


@torch.no_grad()
def evaluate_pipeline_res_ssim(vae, proc, samples, device, use_skip):
    """
    诊断用：在 pipeline 分辨率(跟训练时完全一致的尺寸，不做任何上采样/original_region对齐)下，
    对比 sk_style_ssim(训练loss实际用的，严格对齐skimage) 和真正调用skimage库算出来的值——
    两者理论上应该几乎完全一样(裁边等细节已经对齐)，这里留着是为了双重确认没有遗漏任何实现差异。
    另外对比 evaluate() 算出来的原生分辨率skimage ssim，能看出"分辨率/对齐方式"这个因素还剩多少。
    """
    vals_diff, vals_sk = [], []
    for s in samples:
        out = run(vae, s["z"], s["hazy"], use_skip, device)
        tgt = from_u8(s["gt"], device)
        out_01 = (out.clamp(-1, 1) + 1) / 2
        tgt_01 = (tgt.clamp(-1, 1) + 1) / 2
        vals_diff.append(sk_style_ssim(out_01, tgt_01, data_range=1.0).item())

        pred_np = np.asarray(proc.postprocess(out.float(), output_type="pil")[0])
        gt_np = np.asarray(proc.postprocess(tgt.float(), output_type="pil")[0])
        vals_sk.append(structural_similarity(pred_np, gt_np, channel_axis=2, data_range=255))

    return {"loss_ssim": float(np.mean(vals_diff)), "sk_ssim_pipeline": float(np.mean(vals_sk))}


@torch.no_grad()
def save_outputs(vae, proc, samples, device, out_dir, skip=None):
    """
    保存 pipeline 原生分辨率的解码图，不做任何resize——跟 infer_dehaze.py / batch_inference_skip.py
    的约定一致，这样存出来的图可以直接喂给 evaluate_batch.py，不会重新引入"保存前拉伸失真"的问题。
    """
    baseline_dir = os.path.join(out_dir, "baseline")
    skip_dir = os.path.join(out_dir, "skip")
    os.makedirs(baseline_dir, exist_ok=True)
    os.makedirs(skip_dir, exist_ok=True)

    for s in samples:
        out_b = run(vae, s["z"], s["hazy"], use_skip=False, device=device)
        out_s = run(vae, s["z"], s["hazy"], use_skip=True, device=device)
        for lvl, gate in (getattr(skip, "last_gates", None) or {}).items():
            # gated版：逐像素门控图存成灰度图(白=信任hazy修正量, 黑=不信任)，每个level一个目录
            gate_dir = os.path.join(out_dir, f"gate_L{lvl}")
            os.makedirs(gate_dir, exist_ok=True)
            g8 = (gate[0, 0].float().cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
            Image.fromarray(g8).save(os.path.join(gate_dir, f"{s['name']}.png"))
        proc.postprocess(out_b.float(), output_type="pil")[0].save(os.path.join(baseline_dir, f"{s['name']}.png"))
        proc.postprocess(out_s.float(), output_type="pil")[0].save(os.path.join(skip_dir, f"{s['name']}.png"))

    print(f"图片已保存: {baseline_dir}  /  {skip_dir}")


def fmt_metrics(m):
    return "  ".join(f"{k}={v:.4f}" for k, v in m.items())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--hazy_dir", required=True)
    p.add_argument("--gt_dir", required=True)
    p.add_argument("--dataset", choices=["auto", "reside", "nhhaze", "densehaze"], default="auto",
                   help="GT文件名匹配规则，同eval_infer.py/infer_dehaze.py。"
                        "之前这里硬编码成了nhhaze，导致RESIDE/SOTS数据集一张GT都匹配不上")
    p.add_argument("--latent_dirs", nargs="+", required=True, help="infer_dehaze.py 各seed的 latents 目录")
    p.add_argument("--train_ids", default="1-45")
    p.add_argument("--val_ids", default="46-50", help="留空则不做验证，只保存最后一步")
    p.add_argument("--forbid_ids", default=None, help="测试集编号，禁止出现在train/val里")
    p.add_argument("--init_ckpt", default=None, help="第一阶段的 skip 权重（推荐作为热启动）")
    p.add_argument("--output_dir", default="./checkpoints_stage2")
    p.add_argument("--steps", type=int, default=1500, help="optimizer step 数")
    p.add_argument("--batch_size", type=int, default=1,
                   help="真正的batch（一次forward同时处理几张图），而不是梯度累积。"
                        "要求同一个batch内的图尺寸一致，脚本会按latent尺寸自动分组，"
                        "不会把不同尺寸的图硬凑进一个batch导致torch.cat报错")
    p.add_argument("--accum", type=int, default=4,
                   help="梯度累积（等效batch = batch_size * accum 张图）")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--loss_type", default="l1_ssim", choices=["l1", "ssim", "l1_ssim"],
                   help="l1(默认，向后兼容) / ssim(纯SSIM，注意可能导致亮度色彩漂移) / "
                        "l1_ssim(L1 + ssim_weight*(1-SSIM)混合，更稳妥的折中)")
    p.add_argument("--loss_res", default="native", choices=["pipeline", "native"],
                   help="pipeline(默认，向后兼容): 在pipeline原生输出分辨率上算loss，GT也裁小到这个分辨率；"
                        "native: 把输出可微分地放大到GT原生分辨率对应区域再算loss，"
                        "跟evaluate()/eval_infer.py的评测口径完全对齐。诊断显示pipeline分辨率下的"
                        "loss会持续改善但原生分辨率的真实指标反而变差、且差距随训练扩大，"
                        "就是因为训练目标和真实质量在这个分辨率缺口上脱节了——用native能从根上堵住这个缺口")
    p.add_argument("--ssim_weight", type=float, default=1.0, help="仅 --loss_type l1_ssim 时生效")
    p.add_argument("--skip_type", default="plain", choices=["plain", "gated"],
                   help="plain=原版LastLevelSkipFusion；gated=逐像素门控版 out=dec+gate*delta。"
                        "--eval_only或--init_ckpt本身是gated时，类型以checkpoint为准，忽略此参数")
    p.add_argument("--skip_levels", type=int, nargs="+", default=[3],
                   help="在第几个up_block之后挂skip，可多个。Flux2 VAE共4个(0..3)：level 3=最后一个"
                        "(原来的位置)，level 2 输出256通道、全分辨率。0、1是1/4、1/2分辨率，不建议。"
                        "例：--skip_levels 2 3。--eval_only时以checkpoint为准")
    p.add_argument("--skip_arch", default="perlevel", choices=["perlevel", "pyramid"],
                   help="perlevel=原结构(每层各自 interpolate+两层卷积)；pyramid=共享多尺度像素编码器+门控融合"
                        "(skip_pyramid.py，总是门控版，需要从零训练，不能用perlevel的checkpoint热启动)。"
                        "--init_ckpt本身是pyramid时自动按pyramid处理")
    p.add_argument("--pyr_widths", type=int, nargs=3, default=[64, 128, 128],
                   help="仅pyramid新建时生效：金字塔 1/1、1/2、1/4 三级的通道数。数据很少(如NH-HAZE 45张)时可调小防过拟合")
    p.add_argument("--gate_bias_init", type=float, default=0.0,
                   help="仅gated且新建门控头时生效。从头训练用0(gate=0.5)；"
                        "从plain权重热启动建议3(gate≈0.95)，避免已训练好的修正量被门控减半")
    p.add_argument("--lpips_weight", type=float, default=0.0)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--eval_metrics", nargs="+", default=["psnr", "ssim", "L", "C", "S"],
                   choices=["psnr", "ssim", "L", "C", "S"],
                   help="验证时计算哪些指标；L/C/S是SSIM的亮度/对比度/结构三项分解")
    p.add_argument("--select_metric", default="ssim", choices=["psnr", "ssim", "L", "C", "S"],
                   help="用哪个指标挑选best checkpoint；必须包含在--eval_metrics里。"
                        "只关心结构细节的话用 S 或 ssim，而不是默认的 psnr")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval_only", action="store_true",
                   help="不训练，只加载 --init_ckpt 跑一次评测（无skip / 有skip 各一遍），跑完直接退出")
    p.add_argument("--eval_ids", default=None,
                   help="--eval_only 时用哪些编号评测；不传则用 --val_ids。"
                        "eval_only 模式下不受 --forbid_ids 限制，可以直接传测试集编号如 51-55")
    p.add_argument("--device", default="cuda")
    p.add_argument("--save_images_dir", default=None,
                   help="给了就保存解码出的图片（<dir>/baseline/、<dir>/skip/ 各一份，"
                        "pipeline原生分辨率，不做任何resize），eval_only模式下用eval_samples，"
                        "正常训练模式下训练结束时用val_samples")
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = args.device
    os.makedirs(args.output_dir, exist_ok=True)

    train_ids = parse_ids(args.train_ids)
    val_ids = parse_ids(args.val_ids) if args.val_ids else set()
    forbid = parse_ids(args.forbid_ids) if args.forbid_ids else set()
    if not args.eval_only:
        if (train_ids | val_ids) & forbid:
            raise SystemExit(f"train/val 与测试编号重叠: {sorted((train_ids | val_ids) & forbid)}，会造成数据泄漏")
        if train_ids & val_ids:
            raise SystemExit(f"train 与 val 编号重叠: {sorted(train_ids & val_ids)}")
    if args.select_metric not in args.eval_metrics:
        raise SystemExit(f"--select_metric {args.select_metric} 不在 --eval_metrics {args.eval_metrics} 里")
    if args.eval_only and not args.init_ckpt:
        raise SystemExit("--eval_only 需要通过 --init_ckpt 指定要评测的 skip 权重")

    vae = AutoencoderKLFlux2.from_pretrained(args.base_model, subfolder="vae", torch_dtype=torch.float32)
    vae.requires_grad_(False)
    vae.to(dev).eval()
    block_out_channels = list(vae.config.block_out_channels)
    final_level = len(block_out_channels) - 1
    if args.eval_only:
        skip, desc = load_skip_auto(args.init_ckpt, block_out_channels, dev)  # 按checkpoint严格加载(perlevel/pyramid自动识别)
        skip_kind = "gated" if "gated" in desc else "plain"
        print(f"skip 结构(按checkpoint自动识别): {desc}")
    else:
        sd = to_multi_state_dict(torch.load(args.init_ckpt, map_location="cpu"), final_level) if args.init_ckpt else None
        skip_levels = sorted(set(args.skip_levels))
        skip_kind = args.skip_type
        skip_arch = args.skip_arch
        if sd is not None and is_pyramid_state_dict(sd):
            skip_arch = "pyramid"
        if skip_arch == "pyramid":
            if sd is not None and not is_pyramid_state_dict(sd):
                raise SystemExit("--skip_arch pyramid 不能用 perlevel 的 checkpoint 热启动(特征提取器不同，"
                                 "旧 fuse_conv/gate_head 与新特征不对应)。去掉 --init_ckpt 从零训练，"
                                 "或给一个 pyramid 的 checkpoint")
            if sd is not None:  # 继续训练：结构完全按checkpoint
                skip_levels, widths, gate_hidden = pyramid_config_from_state_dict(sd)
                print(f"[提示] 按 pyramid checkpoint 恢复结构: levels={skip_levels}, widths={list(widths)}")
            else:
                widths, gate_hidden = tuple(args.pyr_widths), 32
            skip_kind = "gated"
            skip = build_pyramid_skip(skip_levels, block_out_channels, widths=widths, gate_hidden=gate_hidden,
                                      gate_bias_init=args.gate_bias_init)
            if sd is not None:
                skip.load_state_dict(sd)  # 严格加载
            n_enc = sum(p_.numel() for p_ in skip.encoder.parameters())
            n_fus = sum(p_.numel() for p_ in skip.levels.parameters())
            print(f"skip 结构: pyramid(widths={list(widths)}), levels={skip_levels}, "
                  f"参数量 编码器={n_enc / 1e6:.2f}M 融合层={n_fus / 1e6:.2f}M")
            skip.to(dev, torch.float32)
        else:
            if sd is not None and skip_kind_from_state_dict(sd) == "gated":
                if args.skip_type != "gated":
                    print(f"[提示] checkpoint 是 gated 版，忽略 --skip_type {args.skip_type}")
                skip_kind = "gated"
            skip = build_multi_skip(skip_kind, skip_levels, block_out_channels, gate_bias_init=args.gate_bias_init)
        if skip_arch != "pyramid" and sd is not None:
            missing, unexpected = skip.load_state_dict(sd, strict=False)
            new_levels = sorted(set(skip_levels) - set(ckpt_levels(sd)))
            bad = unexpected_load_issues(missing, unexpected, new_levels)
            if bad:
                raise SystemExit(f"checkpoint 与当前 skip 结构不匹配(checkpoint 含 level {ckpt_levels(sd)}，"
                                 f"当前 --skip_levels {skip_levels}): {bad}")
            if new_levels:
                print(f"[提示] 新增 level {new_levels}，参数使用新初始化(零初始化的修正量)")
            if any(".gate_head." in k for k in missing):
                print(f"[提示] 部分门控头使用新初始化 (gate_bias_init={args.gate_bias_init})")
        print(f"skip 结构: {skip_kind}, levels={skip_levels}")
        skip.to(dev, torch.float32)
    skip.eval() if args.eval_only else skip.train()
    patch_decoder_auto(vae.decoder, skip)  # perlevel -> patch_decoder_multi，pyramid -> patch_decoder_pyramid
    scale = vae_scale_factor_from_vae(vae)
    proc = Flux2ImageProcessor(vae_scale_factor=scale)

    if args.eval_only:
        eval_ids = parse_ids(args.eval_ids) if args.eval_ids else val_ids
        eval_all = load_samples(args, proc, scale, eval_ids)
        seen, eval_samples = set(), []
        for s in eval_all:  # 每张图只用第一份latent，跟训练时validate的口径一致
            if s["name"] not in seen:
                seen.add(s["name"])
                eval_samples.append(s)
        if not eval_samples:
            raise SystemExit("没有可评测的样本：检查 --eval_ids/--val_ids 与 --latent_dirs 是否对应得上")
        print(f"[eval_only] 编号 {sorted(eval_ids)}，共 {len(eval_samples)} 张")
        base = evaluate(vae, proc, eval_samples, dev, use_skip=False, metrics=args.eval_metrics)
        skip_m = evaluate(vae, proc, eval_samples, dev, use_skip=True, metrics=args.eval_metrics)
        print(f"无skip: {fmt_metrics(base)}")
        print(f"有skip: {fmt_metrics(skip_m)}")
        if args.save_images_dir:
            save_outputs(vae, proc, eval_samples, dev, args.save_images_dir, skip=skip)
        return

    train_samples = load_samples(args, proc, scale, train_ids)
    val_all = load_samples(args, proc, scale, val_ids) if val_ids else []
    seen, val_samples = set(), []
    for s in val_all:  # 验证只用每张图的第一份latent，速度快、结果稳定
        if s["name"] not in seen:
            seen.add(s["name"])
            val_samples.append(s)
    if not train_samples:
        raise SystemExit("没有可用的训练样本：检查 --hazy_dir / --gt_dir / --latent_dirs / --train_ids")
    n_img = len({s["name"] for s in train_samples})
    print(f"训练: {n_img} 张图 / {len(train_samples)} 个 (图, latent) 样本；验证: {len(val_samples)} 张")

    seen, train_eval_samples = set(), []
    for s in train_samples:  # 跟val_samples一样：每张图只取第一份latent，用于监控训练集本身的拟合程度
        if s["name"] not in seen:
            seen.add(s["name"])
            train_eval_samples.append(s)

    lp_net = None
    if args.lpips_weight > 0:
        import lpips

        lp_net = lpips.LPIPS(net="alex").to(dev).eval()
        lp_net.requires_grad_(False)

    opt = torch.optim.AdamW(skip.parameters(), lr=args.lr, weight_decay=0.0)
    lr_lambda = lambda s: min(1.0, (s + 1) / args.warmup) * max(0.05, 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    best = -1.0
    if val_samples:
        base = evaluate(vae, proc, val_samples, dev, use_skip=False, metrics=args.eval_metrics)
        init = evaluate(vae, proc, val_samples, dev, use_skip=True, metrics=args.eval_metrics)
        best = init[args.select_metric]
        print(f"[step 0] val   无skip: {fmt_metrics(base)}")
        print(f"[step 0] val   有skip(初始权重): {fmt_metrics(init)}")
    train_base0 = evaluate(vae, proc, train_eval_samples, dev, use_skip=False, metrics=args.eval_metrics)
    train_init0 = evaluate(vae, proc, train_eval_samples, dev, use_skip=True, metrics=args.eval_metrics)
    print(f"[step 0] train 无skip: {fmt_metrics(train_base0)}")
    print(f"[step 0] train 有skip(初始权重): {fmt_metrics(train_init0)}")

    step, micro = 0, 0
    opt.zero_grad()
    while step < args.steps:
        for batch in make_batches(train_samples, args.batch_size):
            z = torch.cat([s["z"] for s in batch], dim=0)
            hazy = torch.cat([s["hazy"] for s in batch], dim=0)
            gt = torch.cat([s["gt"] for s in batch], dim=0)

            out = run(vae, z, hazy, True, dev)

            if args.loss_res == "pipeline":
                tgt = from_u8(gt, dev)
                if args.loss_type == "l1":
                    loss = F.l1_loss(out, tgt)
                else:
                    out_01 = (out.clamp(-1, 1) + 1) / 2
                    tgt_01 = (tgt.clamp(-1, 1) + 1) / 2
                    ssim_val = sk_style_ssim(out_01, tgt_01, data_range=1.0)
                    loss = (1.0 - ssim_val) if args.loss_type == "ssim" else (
                        F.l1_loss(out, tgt) + args.ssim_weight * (1.0 - ssim_val)
                    )
            else:  # native：逐样本放大到GT原生分辨率对应区域(跟evaluate()的original_region一致)再算loss
                out_01_full = (out.clamp(-1, 1) + 1) / 2
                l1_vals, ssim_vals = [], []
                for i, s in enumerate(batch):
                    gt_native = s["gt_native_01"].to(dev)
                    out_up = F.interpolate(
                        out_01_full[i : i + 1], size=gt_native.shape[-2:], mode="bicubic", align_corners=False
                    ).clamp(0, 1)
                    l1_vals.append(F.l1_loss(out_up, gt_native))
                    if args.loss_type != "l1":
                        ssim_vals.append(sk_style_ssim(out_up, gt_native, data_range=1.0))
                l1_native = torch.stack(l1_vals).mean()
                if args.loss_type == "l1":
                    loss = l1_native
                else:
                    ssim_val = torch.stack(ssim_vals).mean()
                    loss = (1.0 - ssim_val) if args.loss_type == "ssim" else (
                        l1_native + args.ssim_weight * (1.0 - ssim_val)
                    )
                tgt = from_u8(gt, dev)  # 仅供下面lpips分支使用(lpips在pipeline分辨率上算，成本更低)

            if lp_net is not None:
                loss = loss + args.lpips_weight * lp_net(out, tgt).mean()
            (loss / args.accum).backward()
            micro += 1
            if micro % args.accum:
                continue

            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1

            if step % 25 == 0:
                print(f"step {step}/{args.steps}  loss {loss.item():.4f}  lr {sched.get_last_lr()[0]:.2e}")
            if step % args.eval_every == 0:
                # m_train = evaluate(vae, proc, train_eval_samples, dev, use_skip=True, metrics=args.eval_metrics)
                # print(f"[step {step}] train: {fmt_metrics(m_train)}")
                # if "ssim" in args.eval_metrics and args.loss_type != "l1":
                #     diag = evaluate_pipeline_res_ssim(vae, proc, train_eval_samples, dev, use_skip=True)
                #     print(f"[step {step}] train SSIM对比(确认loss跟skimage对齐，以及分辨率这个剩余差异): "
                #           f"原生分辨率/skimage(evaluate()用的)={m_train['ssim']:.4f}  "
                #           f"pipeline分辨率/skimage={diag['sk_ssim_pipeline']:.4f}  "
                #           f"pipeline分辨率/loss用的sk_style_ssim={diag['loss_ssim']:.4f}")
                if val_samples:
                    m = evaluate(vae, proc, val_samples, dev, use_skip=True, metrics=args.eval_metrics)
                    mark = ""
                    if m[args.select_metric] > best:
                        best = m[args.select_metric]
                        torch.save(skip.state_dict(), os.path.join(args.output_dir, "skip_stage2_best.pt"))
                        mark = f"  <- best ({args.select_metric})"
                    print(f"[step {step}] val:   {fmt_metrics(m)}{mark}")
                    if skip_kind == "gated":
                        with torch.no_grad():
                            gs = {}
                            for vs in val_samples:
                                run(vae, vs["z"], vs["hazy"], True, dev)
                                for lvl, g in skip.last_gates.items():
                                    gs.setdefault(lvl, []).append(g.flatten().float().cpu())
                        for lvl, parts in sorted(gs.items()):
                            g = torch.cat(parts)
                            g = g[:: max(1, g.numel() // 4_000_000)]  # quantile对元素数有上限，等间隔下采样
                            print(f"[step {step}] gate L{lvl}(val): mean={g.mean():.3f}  std={g.std():.3f}  "
                                  f"p10={g.quantile(0.1):.3f}  p90={g.quantile(0.9):.3f}")
            if step >= args.steps:
                break

    torch.save(skip.state_dict(), os.path.join(args.output_dir, "skip_stage2_last.pt"))
    print(f"\n完成。last: {args.output_dir}/skip_stage2_last.pt"
          + (f"；best({args.select_metric}={best:.4f}): {args.output_dir}/skip_stage2_best.pt" if val_samples else ""))

    if args.save_images_dir and val_samples:
        skip.eval()
        save_outputs(vae, proc, val_samples, dev, args.save_images_dir, skip=skip)


if __name__ == "__main__":
    main()