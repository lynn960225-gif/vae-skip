"""
train_skip_fusion.py

训练 skip_modules.py 里定义的 LastLevelSkipFusion 模块，
验证在 AutoencoderKLFlux2 的 decoder 最后一级加 skip connection
能否突破纯VAE重建的PSNR天花板。

用法示例：
python train_skip_fusion.py \
    --vae_path black-forest-labs/FLUX.2-klein-4B \
    --train_dir /data/storage/users/yliu/datasets/RESIDE_Standard/ITS_train/clear \
    --val_dir /data/storage/users/yliu/datasets/RESIDE_Standard/SOTS_test/indoor/gt \
    --output_dir ./checkpoints \
    --cond_source gt \
    --num_steps 3000



nohup python -u train_skip_fusion.py \
    --vae_path black-forest-labs/FLUX.2-klein-4B \
    --train_dir /data/storage/users/yliu/datasets/dehazing/NH-HAZE/train/hazy \
    --val_dir /data/storage/users/yliu/datasets/dehazing/NH-HAZE/test/hazy \
    --gt_dir /data/storage/users/yliu/datasets/dehazing/NH-HAZE/train/gt \
    --val_gt_dir /data/storage/users/yliu/datasets/dehazing/NH-HAZE/test/gt \
    --cond_source hazy \
    --output_dir ./checkpoints_hazy_cond_nhhaze \
    --num_steps 100 \
    >logs/train_fusion_nhhaze.log &



--cond_source gt    : 场景A，诊断VAE本身的重建上限（skip输入=GT图，即重建目标本身）
--cond_source hazy  : 场景B，服务真实dehazing pipeline（skip输入=对应的hazy图）
                      这个模式下 --train_dir/--val_dir 需要换成hazy图目录，
                      并额外传 --gt_dir 指向对应的GT图目录（文件名一一对应）
"""

import argparse
import glob
import os

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from diffusers import AutoencoderKLFlux2
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from skip_modules import LastLevelSkipFusion, patch_decoder_with_skip


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ReconDataset(Dataset):
    """
    单目录模式（cond_source=gt）：__getitem__ 返回 (x, x) —— 重建目标和skip输入是同一张图
    双目录模式（cond_source=hazy）：__getitem__ 返回 (gt, hazy) —— 按文件名匹配
    """

    def __init__(self, img_dir, size=512, gt_dir=None):
        self.paths = sorted(
            glob.glob(os.path.join(img_dir, "*.png")) + glob.glob(os.path.join(img_dir, "*.jpg"))
        )
        if len(self.paths) == 0:
            raise ValueError(f"没有在 {img_dir} 找到任何 .png/.jpg 图片")

        self.gt_dir = gt_dir
        self.tf = T.Compose(
            [
                T.Resize(size),
                T.CenterCrop(size),
                T.ToTensor(),  # [0,1]
                T.Normalize([0.5] * 3, [0.5] * 3),  # -> [-1,1]，跟VAE输入范围对齐
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        img = self.tf(img)

        if self.gt_dir is None:
            # 场景A：重建目标 == skip输入 == 同一张图
            return img, img
        else:
            # 场景B：img是hazy(train_dir传的是hazy_dir)，gt从gt_dir按文件名找
            gt_base = os.path.basename(path).split('_')[0] + '_GT.png'
            gt_path = os.path.join(self.gt_dir, gt_base)
            gt_img = Image.open(gt_path).convert("RGB")
            gt_img = self.tf(gt_img)
            return gt_img, img  # (重建目标=GT, skip输入=hazy)


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------
def compute_psnr(pred, target):
    # pred, target 都在 [-1, 1] 范围
    mse = F.mse_loss(pred, target)
    if mse.item() == 0:
        return torch.tensor(float("inf"))
    return 10 * torch.log10(4.0 / mse)  # (max-min)^2 = 2^2 = 4


@torch.no_grad()
def eval_psnr(vae, val_loader, device, use_skip: bool):
    total_psnr, n = 0.0, 0
    for target, cond in val_loader:
        target, cond = target.to(device), cond.to(device)
        # encoder的输入永远是"重建目标"本身（即我们希望decoder最终恢复出的那张图对应的latent）；
        # 这跟cond_source无关——encode(target)是重建任务的输入，cond只是额外喂给decoder的辅助信息
        posterior = vae.encode(target).latent_dist
        z = posterior.mode()
        z_in = vae.post_quant_conv(z) if vae.post_quant_conv is not None else z
        cond_arg = cond if use_skip else None
        recon = vae.decoder(z_in, cond_img=cond_arg)
        total_psnr += compute_psnr(recon, target).item() * target.shape[0]
        n += target.shape[0]
    return total_psnr / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_path", type=str, required=True, help="HF repo id 或本地路径")
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--gt_dir", type=str, default=None, help="仅 cond_source=hazy 时需要，对应的GT图目录")
    parser.add_argument("--val_gt_dir", type=str, default=None, help="仅 cond_source=hazy 时需要，验证集的GT图目录")
    parser.add_argument("--cond_source", type=str, choices=["gt", "hazy"], default="gt")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_steps", type=int, default=3000)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.cond_source == "hazy" and (args.gt_dir is None or args.val_gt_dir is None):
        raise ValueError("cond_source=hazy 时必须同时提供 --gt_dir 和 --val_gt_dir")

    os.makedirs(args.output_dir, exist_ok=True)

    # ---------------- 模型 ----------------
    vae = AutoencoderKLFlux2.from_pretrained(args.vae_path, subfolder="vae", torch_dtype=torch.float32)
    vae.requires_grad_(False)
    vae.eval()

    skip_fusion = LastLevelSkipFusion(decoder_channels=vae.config.block_out_channels[0]).to(args.device)
    patch_decoder_with_skip(vae.decoder, skip_fusion)
    vae.to(args.device)

    # ---------------- 数据 ----------------
    train_gt_dir = args.gt_dir if args.cond_source == "hazy" else None
    val_gt_dir = args.val_gt_dir if args.cond_source == "hazy" else None

    train_dataset = ReconDataset(args.train_dir, size=args.image_size, gt_dir=train_gt_dir)
    val_dataset = ReconDataset(args.val_dir, size=args.image_size, gt_dir=val_gt_dir)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"训练图片数: {len(train_dataset)}, 验证图片数: {len(val_dataset)}")
    print(f"effective batch size = {args.batch_size * args.grad_accum_steps}")

    # ---------------- baseline PSNR（跑训练前先测一次，作为对照） ----------------
    baseline_psnr = eval_psnr(vae, val_loader, args.device, use_skip=False)
    print(f"[Step 0] baseline PSNR (无skip，纯原始decoder): {baseline_psnr:.3f} dB")

    # ---------------- 优化器：只优化skip_fusion ----------------
    optimizer = torch.optim.AdamW(skip_fusion.parameters(), lr=args.lr)

    # ---------------- 训练循环 ----------------
    skip_fusion.train()
    step = 0
    accum_counter = 0
    optimizer.zero_grad()

    while step < args.num_steps:
        for target, cond in train_loader:
            target, cond = target.to(args.device), cond.to(args.device)

            with torch.no_grad():
                posterior = vae.encode(target).latent_dist
                z = posterior.mode()
                z_in = vae.post_quant_conv(z) if vae.post_quant_conv is not None else z

            recon = vae.decoder(z_in, cond_img=cond)
            loss = F.l1_loss(recon, target) / args.grad_accum_steps
            loss.backward()
            accum_counter += 1

            if accum_counter == args.grad_accum_steps:
                optimizer.step()
                optimizer.zero_grad()
                accum_counter = 0
                step += 1

                if step % args.log_every == 0:
                    print(f"step {step}/{args.num_steps}  loss {loss.item() * args.grad_accum_steps:.4f}")

                if step % args.eval_every == 0:
                    skip_fusion.eval()
                    skip_psnr = eval_psnr(vae, val_loader, args.device, use_skip=True)
                    skip_fusion.train()
                    print(f"[Step {step}] baseline {baseline_psnr:.3f} dB -> with skip {skip_psnr:.3f} dB "
                          f"(提升 {skip_psnr - baseline_psnr:+.3f} dB)")

                if step % args.save_every == 0:
                    ckpt_path = os.path.join(args.output_dir, f"skip_fusion_step{step}.pt")
                    torch.save(skip_fusion.state_dict(), ckpt_path)
                    print(f"已保存 checkpoint: {ckpt_path}")

                if step >= args.num_steps:
                    break

    # ---------------- 收尾：最终评估 + 保存 ----------------
    skip_fusion.eval()
    final_skip_psnr = eval_psnr(vae, val_loader, args.device, use_skip=True)
    print(f"\n训练结束。baseline PSNR = {baseline_psnr:.3f} dB, 最终 skip PSNR = {final_skip_psnr:.3f} dB, "
          f"提升 = {final_skip_psnr - baseline_psnr:+.3f} dB")

    final_ckpt = os.path.join(args.output_dir, "skip_fusion_final.pt")
    torch.save(skip_fusion.state_dict(), final_ckpt)
    print(f"最终权重已保存到: {final_ckpt}")


if __name__ == "__main__":
    main()