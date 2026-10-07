"""
eval_psnr.py

加载训练好的 skip_fusion checkpoint，在验证/测试集上对比：
    - baseline: 原始 AutoencoderKLFlux2 decoder（无skip）
    - with skip: 加了 LastLevelSkipFusion 之后的 decoder
输出整体PSNR对比，并可选保存逐张图片的定量结果 + 可视化对比图（GT / baseline / skip 三联图）。

用法示例：
    python eval_psnr.py \
        --vae_path black-forest-labs/FLUX.2-klein-4B \
        --val_dir /path/to/val_images \
        --checkpoint ./checkpoints/skip_fusion_final.pt \
        --cond_source gt \
        --save_samples_dir ./eval_vis --num_vis_samples 8
"""

import argparse
import csv
import os

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from diffusers import AutoencoderKLFlux2
from torch.utils.data import DataLoader

from skip_modules import LastLevelSkipFusion, patch_decoder_with_skip
from train_skip_fusion import ReconDataset, compute_psnr  # 复用训练脚本里的Dataset和PSNR函数


@torch.no_grad()
def evaluate(vae, val_loader, device, use_skip: bool, save_dir=None, num_vis_samples=0):
    """
    返回: (平均PSNR, 每张图的PSNR列表)
    如果 save_dir 不为空，额外保存 num_vis_samples 张 GT/recon 对比图
    """
    per_image_psnr = []
    saved = 0

    for target, cond in val_loader:
        target, cond = target.to(device), cond.to(device)

        posterior = vae.encode(target).latent_dist
        z = posterior.mode()
        z_in = vae.post_quant_conv(z) if vae.post_quant_conv is not None else z

        cond_arg = cond if use_skip else None
        recon = vae.decoder(z_in, cond_img=cond_arg)

        # 逐张算PSNR（batch内每张单独算，不是整个batch算一个MSE）
        for i in range(target.shape[0]):
            mse = F.mse_loss(recon[i], target[i])
            psnr = 10 * torch.log10(4.0 / mse) if mse.item() > 0 else torch.tensor(float("inf"))
            per_image_psnr.append(psnr.item())

        if save_dir is not None and saved < num_vis_samples:
            n_to_save = min(num_vis_samples - saved, target.shape[0])
            for i in range(n_to_save):
                # [-1,1] -> [0,1] 方便存图
                gt_img = (target[i:i+1] + 1) / 2
                recon_img = (recon[i:i+1] + 1) / 2
                cond_img = (cond[i:i+1] + 1) / 2
                tag = "skip" if use_skip else "baseline"
                grid = vutils.make_grid(torch.cat([gt_img, cond_img, recon_img], dim=0), nrow=3)
                vutils.save_image(grid, os.path.join(save_dir, f"sample_{saved + i:03d}_{tag}.png"))
            saved += n_to_save

    avg_psnr = sum(per_image_psnr) / len(per_image_psnr)
    return avg_psnr, per_image_psnr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_path", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--gt_dir", type=str, default=None, help="仅 cond_source=hazy 时需要")
    parser.add_argument("--cond_source", type=str, choices=["gt", "hazy"], default="gt")
    parser.add_argument("--checkpoint", type=str, required=True, help="skip_fusion 的 .pt 权重路径")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_samples_dir", type=str, default=None, help="保存对比图的目录，不传则不保存")
    parser.add_argument("--num_vis_samples", type=int, default=8)
    parser.add_argument("--csv_out", type=str, default=None, help="逐张图片PSNR明细的csv输出路径")
    args = parser.parse_args()

    if args.cond_source == "hazy" and args.gt_dir is None:
        raise ValueError("cond_source=hazy 时必须提供 --gt_dir")

    if args.save_samples_dir is not None:
        os.makedirs(args.save_samples_dir, exist_ok=True)

    # ---------------- 模型 ----------------
    vae = AutoencoderKLFlux2.from_pretrained(args.vae_path, subfolder="vae", torch_dtype=torch.float32)
    vae.requires_grad_(False)
    vae.eval()

    skip_fusion = LastLevelSkipFusion(decoder_channels=vae.config.block_out_channels[0])
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    skip_fusion.load_state_dict(state_dict)
    skip_fusion.eval()
    skip_fusion.to(args.device)

    patch_decoder_with_skip(vae.decoder, skip_fusion)
    vae.to(args.device)

    # ---------------- 数据 ----------------
    gt_dir = args.gt_dir if args.cond_source == "hazy" else None
    val_dataset = ReconDataset(args.val_dir, size=args.image_size, gt_dir=gt_dir)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"验证图片数: {len(val_dataset)}")

    # ---------------- baseline ----------------
    baseline_psnr, baseline_list = evaluate(
        vae, val_loader, args.device, use_skip=False,
        save_dir=args.save_samples_dir, num_vis_samples=args.num_vis_samples,
    )
    print(f"[baseline] 平均PSNR: {baseline_psnr:.3f} dB")

    # ---------------- with skip ----------------
    skip_psnr, skip_list = evaluate(
        vae, val_loader, args.device, use_skip=True,
        save_dir=args.save_samples_dir, num_vis_samples=args.num_vis_samples,
    )
    print(f"[with skip] 平均PSNR: {skip_psnr:.3f} dB")
    print(f"提升: {skip_psnr - baseline_psnr:+.3f} dB")

    # ---------------- 保存逐张明细 ----------------
    if args.csv_out is not None:
        with open(args.csv_out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "image_path", "baseline_psnr", "skip_psnr", "improvement"])
            for idx, path in enumerate(val_dataset.paths):
                b = baseline_list[idx]
                s = skip_list[idx]
                writer.writerow([idx, path, f"{b:.3f}", f"{s:.3f}", f"{s - b:+.3f}"])
        print(f"逐张PSNR明细已保存到: {args.csv_out}")

    if args.save_samples_dir is not None:
        print(f"对比图（GT / cond输入 / 重建）已保存到: {args.save_samples_dir}")


if __name__ == "__main__":
    main()