"""
oracle_check.py

二分定位：skip 在"理想latent"(z = encode(GT))下，走与 infer_dehaze.py 完全相同的
预处理 / fp32 / 补丁decoder 路径，还有没有收益？

  - 复现了 ~+3.6dB  -> 推理路径没问题，问题在"生成latent与GT latent的分布差异"
  - 没复现          -> 推理路径(预处理/对齐/尺寸/dtype等)有问题，先修这里

用法：
    python oracle_check.py --base_model black-forest-labs/FLUX.2-klein-4B \
        --skip_ckpt ./checkpoints_hazy_cond/skip_fusion_final.pt \
        --hazy_dir /data/.../ITS_train/hazy --gt_dir /data/.../ITS_train/clear \
        --dataset reside --limit 100 [--out_dir ./oracle_out]
"""

import argparse
import glob
import os

import numpy as np
import torch
from diffusers import AutoencoderKLFlux2
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
import torchvision.transforms as T
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio

from eval_infer import gt_candidates, index_dir
from infer_dehaze import preprocess_cond, vae_scale_factor_from_vae
from skip_modules import LastLevelSkipFusion, patch_decoder_with_skip


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--skip_ckpt", required=True)
    p.add_argument("--hazy_dir", required=True)
    p.add_argument("--gt_dir", required=True)
    p.add_argument("--dataset", choices=["auto", "reside", "nhhaze", "densehaze"], default="auto")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--preprocess", choices=["pipeline", "train"], default="pipeline",
                   help="pipeline=原生分辨率(同infer_dehaze.py)；train=Resize+CenterCrop(同train_skip_fusion.py训练时)")
    p.add_argument("--size", type=int, default=512, help="仅 --preprocess train 时使用")
    p.add_argument("--out_dir", default=None, help="给了就保存 baseline/skip 图，可直接喂给 eval_infer.py")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    vae = AutoencoderKLFlux2.from_pretrained(args.base_model, subfolder="vae", torch_dtype=torch.float32)
    vae.to(args.device).eval()
    skip = LastLevelSkipFusion(decoder_channels=vae.config.block_out_channels[0])
    skip.load_state_dict(torch.load(args.skip_ckpt, map_location="cpu"))
    skip.to(args.device, torch.float32).eval()
    patch_decoder_with_skip(vae.decoder, skip)
    scale = vae_scale_factor_from_vae(vae)
    proc = Flux2ImageProcessor(vae_scale_factor=scale)
    train_tf = T.Compose([T.Resize(args.size), T.CenterCrop(args.size), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
    prep = (lambda im: train_tf(im)[None]) if args.preprocess == "train" else (lambda im: preprocess_cond(proc, scale, im))

    gt_index = index_dir(args.gt_dir)
    hazy_paths = sorted(
        glob.glob(os.path.join(args.hazy_dir, "*.png")) + glob.glob(os.path.join(args.hazy_dir, "*.jpg"))
    )[: args.limit]

    if args.out_dir:
        for k in ("baseline", "skip"):
            os.makedirs(os.path.join(args.out_dir, k), exist_ok=True)

    psnr_b, psnr_s = [], []
    for path in hazy_paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        gt_path = next((gt_index[c] for c in gt_candidates(stem, args.dataset, "", "") if c in gt_index), None)
        if gt_path is None:
            continue

        hazy_t = prep(Image.open(path).convert("RGB")).to(args.device, torch.float32)
        gt_t = prep(Image.open(gt_path).convert("RGB")).to(args.device, torch.float32)

        z = vae.encode(gt_t).latent_dist.mode()  # 理想latent：与训练skip时完全同分布
        z_in = vae.post_quant_conv(z) if vae.post_quant_conv is not None else z
        out_b = vae.decoder(z_in, cond_img=None)
        out_s = vae.decoder(z_in, cond_img=hazy_t)

        to_pil = lambda x: proc.postprocess(x, output_type="pil")[0]
        gt_np = np.asarray(to_pil(gt_t))
        img_b, img_s = to_pil(out_b), to_pil(out_s)
        psnr_b.append(peak_signal_noise_ratio(gt_np, np.asarray(img_b), data_range=255))
        psnr_s.append(peak_signal_noise_ratio(gt_np, np.asarray(img_s), data_range=255))

        if args.out_dir:
            img_b.save(os.path.join(args.out_dir, "baseline", f"{stem}.png"))
            img_s.save(os.path.join(args.out_dir, "skip", f"{stem}.png"))

    if not psnr_b:
        raise SystemExit("没有匹配到任何 hazy/GT 对，检查 --dataset 和目录")
    print(f"[preprocess={args.preprocess}] N={len(psnr_b)}  oracle latent 下：baseline {np.mean(psnr_b):.3f} dB  ->  skip {np.mean(psnr_s):.3f} dB "
          f"(Δ {np.mean(psnr_s) - np.mean(psnr_b):+.3f} dB)")


if __name__ == "__main__":
    main()