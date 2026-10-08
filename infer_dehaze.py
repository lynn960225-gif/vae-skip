"""
infer_dehaze.py


CUDA_VISIBLE_DEVICES=0  python infer_dehaze.py \
        --base_model black-forest-labs/FLUX.2-klein-4B \
        --lora /data/storage/users/yliu/outputs/flux2-i2i-densehaze-4b-merge/checkpoint-3000 --lora_weight_name pytorch_lora_weights.safetensors \
        --skip_ckpt /home/yliu/code/vae-skip-experiment/checkpoints_hazy_cond/skip_fusion_final.pt \
        --hazy_dir /data/storage/users/yliu/datasets/dehazing/Dense_Haze/train/hazy --out_dir ./infer_out_dense_check \
        --gt_dir /data/storage/users/yliu/datasets/dehazing/Dense_Haze/train/gt \
        --prompt "remove haze, restore clear visibility" --steps 4 --seed 5 --save_latents


CUDA_VISIBLE_DEVICES=5  python infer_dehaze.py \
        --base_model black-forest-labs/FLUX.2-klein-4B \
        --lora /data/storage/users/yliu/outputs/flux2-i2i-densehaze-4b-merge/checkpoint-3000 --lora_weight_name pytorch_lora_weights.safetensors \
        --skip_ckpt /home/yliu/code/vae-skip-experiment/checkpoints_stage2_its/skip_stage2_best.pt \
        --hazy_dir /data/storage/users/yliu/datasets/dehazing/Dense_Haze/train/hazy --out_dir ./infer_out_dense_test \
        --gt_dir /data/storage/users/yliu/datasets/dehazing/Dense_Haze/train/gt \
        --prompt "remove haze, restore clear visibility" --steps 4 --seed 5 --save_latents

nohup bash -c '
for seed in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=0 python -u infer_dehaze.py \
      --base_model black-forest-labs/FLUX.2-klein-4B \
      --lora /data/storage/users/yliu/outputs/flux2-i2i-its-cos/checkpoint-3000 --lora_weight_name pytorch_lora_weights.safetensors \
      --skip_ckpt /home/yliu/code/vae-skip-experiment/checkpoints_hazy_cond/skip_fusion_final.pt \
      --hazy_dir /data/storage/users/yliu/datasets/RESIDE_Standard/ITS_train/hazy --out_dir ./infer_out_its_0 \
      --gt_dir /data/storage/users/yliu/datasets/RESIDE_Standard/ITS_train/clear \
      --prompt "remove haze, restore clear visibility" --steps 4 --seed 0 --save_latents
done
' > logs/infer_its_multiseed_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "PID: $!"


nohup bash -c '
for seed in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=2 python -u infer_dehaze.py \
      --base_model black-forest-labs/FLUX.2-klein-4B \
      --lora /data/storage/users/yliu/outputs/flux2-i2i-nhhaze-4b-skip/checkpoint-2900 --lora_weight_name pytorch_lora_weights.safetensors \
      --skip_ckpt /home/yliu/code/vae-skip-experiment/checkpoints_stage2_nhhaze/skip_stage2_best.pt \
      --hazy_dir /data/storage/users/yliu/datasets/dehazing/NH-HAZE/train/hazy --out_dir ./infer_out_nhhaze_skip_$seed \
      --gt_dir /data/storage/users/yliu/datasets/dehazing/NH-HAZE/train/gt \
      --prompt "remove haze, restore clear visibility" --steps 4 --seed $seed --save_latents
done
' > logs/infer_nhhaze_skip_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "PID: $!"

nohup bash -c '
for seed in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=2 python -u infer_dehaze.py \
      --base_model black-forest-labs/FLUX.2-klein-4B \
      --lora /data/storage/users/yliu/outputs/flux2-i2i-densehaze-4b-skip-ssim/checkpoint-3000 --lora_weight_name pytorch_lora_weights.safetensors \
      --skip_ckpt /home/yliu/code/vae-skip-experiment/checkpoints_stage2_densehaze/skip_stage2_best.pt \
      --hazy_dir /data/storage/users/yliu/datasets/dehazing/Dense_Haze/train/hazy --out_dir ./infer_out_densehaze_skip_$seed \
      --gt_dir /data/storage/users/yliu/datasets/dehazing/Dense_Haze/train/gt \
      --prompt "remove haze, restore clear visibility" --steps 4 --seed $seed --save_latents
done
' > logs/infer_densehaze_skip_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "PID: $!"

"""
"""
infer_dehaze.py


CUDA_VISIBLE_DEVICES=0  python infer_dehaze.py \
        --base_model black-forest-labs/FLUX.2-klein-4B \
        --lora /data/storage/users/yliu/outputs/flux2-i2i-densehaze-4b-merge/checkpoint-3000 --lora_weight_name pytorch_lora_weights.safetensors \
        --skip_ckpt /home/yliu/code/vae-skip-experiment/checkpoints_hazy_cond/skip_fusion_final.pt \
        --hazy_dir /data/storage/users/yliu/datasets/dehazing/Dense_Haze/train/hazy --out_dir ./infer_out_dense_check \
        --gt_dir /data/storage/users/yliu/datasets/dehazing/Dense_Haze/train/gt \
        --prompt "remove haze, restore clear visibility" --steps 4 --seed 5 --save_latents


CUDA_VISIBLE_DEVICES=5  python infer_dehaze.py \
        --base_model black-forest-labs/FLUX.2-klein-4B \
        --lora /data/storage/users/yliu/outputs/flux2-i2i-densehaze-4b-merge/checkpoint-3000 --lora_weight_name pytorch_lora_weights.safetensors \
        --skip_ckpt /home/yliu/code/vae-skip-experiment/checkpoints_stage2_its/skip_stage2_best.pt \
        --hazy_dir /data/storage/users/yliu/datasets/dehazing/Dense_Haze/train/hazy --out_dir ./infer_out_dense_test \
        --gt_dir /data/storage/users/yliu/datasets/dehazing/Dense_Haze/train/gt \
        --prompt "remove haze, restore clear visibility" --steps 4 --seed 5 --save_latents

nohup bash -c '
for seed in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=0 python -u infer_dehaze.py \
      --base_model black-forest-labs/FLUX.2-klein-4B \
      --lora /data/storage/users/yliu/outputs/flux2-i2i-its-cos/checkpoint-3000 --lora_weight_name pytorch_lora_weights.safetensors \
      --skip_ckpt /home/yliu/code/vae-skip-experiment/checkpoints_hazy_cond/skip_fusion_final.pt \
      --hazy_dir /data/storage/users/yliu/datasets/RESIDE_Standard/ITS_train/hazy --out_dir ./infer_out_its_0 \
      --gt_dir /data/storage/users/yliu/datasets/RESIDE_Standard/ITS_train/clear \
      --prompt "remove haze, restore clear visibility" --steps 4 --seed 0 --save_latents
done
' > logs/infer_its_multiseed_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "PID: $!"


nohup bash -c '
for seed in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=2 python -u infer_dehaze.py \
      --base_model black-forest-labs/FLUX.2-klein-4B \
      --lora /data/storage/users/yliu/outputs/flux2-i2i-nhhaze-4b-skip/checkpoint-2900 --lora_weight_name pytorch_lora_weights.safetensors \
      --skip_ckpt /home/yliu/code/vae-skip-experiment/checkpoints_stage2_nhhaze/skip_stage2_best.pt \
      --hazy_dir /data/storage/users/yliu/datasets/dehazing/NH-HAZE/train/hazy --out_dir ./infer_out_nhhaze_skip_$seed \
      --gt_dir /data/storage/users/yliu/datasets/dehazing/NH-HAZE/train/gt \
      --prompt "remove haze, restore clear visibility" --steps 4 --seed $seed --save_latents
done
' > logs/infer_nhhaze_skip_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "PID: $!"

nohup bash -c '
for seed in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=2 python -u infer_dehaze.py \
      --base_model black-forest-labs/FLUX.2-klein-4B \
      --lora /data/storage/users/yliu/outputs/flux2-i2i-densehaze-4b-skip-ssim/checkpoint-3000 --lora_weight_name pytorch_lora_weights.safetensors \
      --skip_ckpt /home/yliu/code/vae-skip-experiment/checkpoints_stage2_densehaze/skip_stage2_best.pt \
      --hazy_dir /data/storage/users/yliu/datasets/dehazing/Dense_Haze/train/hazy --out_dir ./infer_out_densehaze_skip_$seed \
      --gt_dir /data/storage/users/yliu/datasets/dehazing/Dense_Haze/train/gt \
      --prompt "remove haze, restore clear visibility" --steps 4 --seed $seed --save_latents
done
' > logs/infer_densehaze_skip_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "PID: $!"

"""

import argparse
import glob
import os

import numpy as np
import torch
from diffusers import Flux2KleinPipeline
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from eval_infer import gt_candidates, index_dir, make_gt_aligner, original_region
from skip_pyramid import load_skip_auto, patch_decoder_auto  # 同时支持旧版 per-level 和 pyramid 的 checkpoint
from traj_guidance import attach_trajectory_guidance, load_traj

try:
    from ssim_decompose import ssim_decompose

    HAS_SSIM_TERMS = True
except ImportError:
    HAS_SSIM_TERMS = False


def load_pipe(args):
    pipe = Flux2KleinPipeline.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
    if args.lora:
        pipe.load_lora_weights(args.lora, weight_name=args.lora_weight_name)
    pipe.to(args.device)

    # VAE(含skip)全程fp32：bf16的舍入误差量级足以吃掉 41->45 dB 这种差异
    pipe.vae.to(torch.float32)
    skip, desc = load_skip_auto(args.skip_ckpt, list(pipe.vae.config.block_out_channels), args.device)
    print(f"skip 结构(按checkpoint自动识别): {desc}")
    patch_decoder_auto(pipe.vae.decoder, skip)

    # 轨迹侧像素引导(可选)：必须与训练该 LoRA 时是同一个 traj_guidance.pt，否则条件分布对不上
    if getattr(args, "traj_ckpt", None):
        traj = load_traj(args.traj_ckpt, args.device)
        attach_trajectory_guidance(pipe, traj)
        print(f"已挂载轨迹侧像素引导: {args.traj_ckpt}  config={traj.config}")
    return pipe


def vae_scale_factor_from_vae(vae):
    """按diffusers管线内部的标准算法，从VAE本身推导vae_scale_factor，
    不需要加载完整Flux2KleinPipeline。训练/校验脚本只load了vae时用这个，
    保证和pipe.vae_scale_factor永远是同一个数，不会因为改动而悄悄分裂。"""
    return 2 ** (len(vae.config.block_out_channels) - 1)


def preprocess_cond(image_processor, vae_scale_factor, hazy: Image.Image):
    """
    复刻 Flux2KleinPipeline.__call__ 中对 condition image 的预处理
    (超过1MP先缩放 -> 裁到 vae_scale_factor*2 的整数倍 -> [-1,1])。

    只依赖 image_processor + vae_scale_factor（一个int），不需要完整pipeline实例，
    因此 infer_dehaze.py / oracle_check.py / train_skip_stage2.py 可以共用这一个函数，
    而不是各自维护一份"应该等价"的实现——那样容易在改动时悄悄产生分歧。
    """
    w, h = hazy.size
    if w * h > 1024 * 1024:
        hazy = image_processor._resize_to_target_area(hazy, 1024 * 1024)
        w, h = hazy.size
    m = vae_scale_factor * 2
    w, h = (w // m) * m, (h // m) * m
    return image_processor.preprocess(hazy, height=h, width=w, resize_mode="crop")  # (1,3,h,w), [-1,1]


@torch.no_grad()
def decode(pipe, latents, cond, use_skip: bool):
    vae = pipe.vae
    z = latents.to(vae.device, torch.float32)

    if not use_skip:
        # 官方decode路径；补丁后的decoder在cond_img=None时等价于原始decoder
        img = vae.decode(z, return_dict=False)[0]
    else:
        cond = cond.to(vae.device, torch.float32)
        s = pipe.vae_scale_factor
        # 不能让skip内部的F.interpolate悄悄掩盖尺寸错位，所以这里显式检查
        assert cond.shape[-2] == z.shape[-2] * s and cond.shape[-1] == z.shape[-1] * s, (
            f"cond尺寸 {tuple(cond.shape[-2:])} 与 latent×{s} = "
            f"{(z.shape[-2] * s, z.shape[-1] * s)} 不一致，检查预处理是否与pipeline一致"
        )
        z_in = vae.post_quant_conv(z) if vae.post_quant_conv is not None else z
        img = vae.decoder(z_in, cond_img=cond)

    return pipe.image_processor.postprocess(img, output_type="pil")[0]


@torch.no_grad()
def sanity_check(pipe, latents):
    """补丁不应改变 cond_img=None 时的行为：官方decode == 手动 post_quant_conv + decoder"""
    vae = pipe.vae
    z = latents.to(vae.device, torch.float32)
    a = vae.decode(z, return_dict=False)[0]
    z_in = vae.post_quant_conv(z) if vae.post_quant_conv is not None else z
    b = vae.decoder(z_in, cond_img=None)
    err = (a - b).abs().max().item()
    print(f"[sanity] vae.decode 与 手动decoder路径最大差异: {err:.2e}")
    assert err < 1e-3, "手动解码路径与vae.decode不一致：说明decode里还有本脚本没复刻的处理，先别信A/B结果"


def find_gt(stem, gt_index, dataset):
    return next((gt_index[c] for c in gt_candidates(stem, dataset, "", "") if c in gt_index), None)


def compute_metrics(pred: Image.Image, gt: Image.Image, aligner, eval_res="original"):
    """
    eval_res="pipeline": GT缩放/裁剪到匹配pred的输出分辨率(与eval_infer.py --eval_res pipeline一致)。
      GT的细节被一起降采样掉了，会让模糊/错位这类问题被低估，数字偏乐观。
    eval_res="original"(默认): GT保持原生分辨率不动，把pred放大回GT尺寸再比较
      (与eval_infer.py --eval_res original、以及常见论文协议一致)。这是应该用来跟外部评测脚本
      对照、写进报告的口径；上面"pipeline"模式只适合快速跑分/内部A-B，不该用来对外报数字。
    """
    if eval_res == "original":
        region = original_region(pred.size, gt.size)
        gt_cropped = gt.crop(region)
        pred = pred.resize(gt_cropped.size, Image.BICUBIC)
        gt = gt_cropped
    elif gt.size != pred.size:
        gt = aligner(gt)
        if gt.size != pred.size:
            return None  # 对齐后仍不一致（极端长宽比等），跳过而不是报错中断整批推理

    a, b = np.asarray(pred), np.asarray(gt)
    m = {
        "psnr": peak_signal_noise_ratio(b, a, data_range=255),
        "ssim": structural_similarity(a, b, channel_axis=2, data_range=255),
    }
    if HAS_SSIM_TERMS:
        ga = np.asarray(pred.convert("L"), dtype=np.float64)
        gb = np.asarray(gt.convert("L"), dtype=np.float64)
        l_map, c_map, s_map, *_ = ssim_decompose(ga, gb)
        m["S"] = float(s_map.mean())
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--lora", default=None)
    p.add_argument("--lora_weight_name", default=None)
    p.add_argument("--skip_ckpt", required=True)
    p.add_argument("--traj_ckpt", default=None,
                   help="训练时用 --traj_guidance 得到的 traj_guidance.pt（与该 LoRA 配套）；不给则不启用轨迹侧像素引导")
    p.add_argument("--hazy_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--prompt", required=True, help="与训练LoRA时一致的prompt")
    p.add_argument("--steps", type=int, default=4,
                   help="蒸馏版klein-4B用4；base版klein-base-4B需约50，必须与训练/缓存latent时一致")
    p.add_argument("--guidance_scale", type=float, default=1.0,
                   help="蒸馏版忽略该值；base版是真实生效的参数，须与训练时一致")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--save_latents", action="store_true", help="缓存latent，之后可重复解码/做skip的第二阶段微调")
    p.add_argument("--gt_dir", default=None, help="给了就同时评测baseline/skip相对GT的PSNR/SSIM")
    p.add_argument("--dataset", choices=["auto", "reside", "nhhaze", "densehaze"], default="auto",
                   help="GT文件名匹配规则，同eval_infer.py")
    p.add_argument("--eval_res", choices=["pipeline", "original"], default="original",
                   help="original(默认，与外部eval.py/论文协议一致): GT不动，输出放大回GT分辨率再比；"
                        "pipeline: GT缩小到输出分辨率再比，数字会偏乐观，只适合内部快速A/B，不要用于对外报数字")
    args = p.parse_args()

    dirs = {k: os.path.join(args.out_dir, k) for k in ["baseline", "skip", "latents"]}
    for k, d in dirs.items():
        if k != "latents" or args.save_latents:
            os.makedirs(d, exist_ok=True)

    pipe = load_pipe(args)

    gt_index = index_dir(args.gt_dir) if args.gt_dir else None
    aligner = make_gt_aligner() if args.gt_dir else None
    metrics_b, metrics_s, no_gt = [], [], []
    if args.gt_dir and not HAS_SSIM_TERMS:
        print("[warn] 找不到 ssim_decompose.py，只评测 PSNR/SSIM，跳过 Structure项(S)")

    paths = sorted(glob.glob(os.path.join(args.hazy_dir, "*.png")) + glob.glob(os.path.join(args.hazy_dir, "*.jpg")))
    print(f"共 {len(paths)} 张hazy图")

    for i, path in enumerate(paths):
        stem = os.path.splitext(os.path.basename(path))[0]
        hazy = Image.open(path).convert("RGB")
        cond = preprocess_cond(pipe.image_processor, pipe.vae_scale_factor, hazy)

        generator = torch.Generator(device=args.device).manual_seed(args.seed)
        latents = pipe(
            image=hazy,
            prompt=args.prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            output_type="latent",
        ).images

        if i == 0:
            sanity_check(pipe, latents)

        img_b = decode(pipe, latents, cond, use_skip=False)
        img_s = decode(pipe, latents, cond, use_skip=True)
        img_b.save(os.path.join(dirs["baseline"], f"{stem}.png"))
        img_s.save(os.path.join(dirs["skip"], f"{stem}.png"))
        if args.save_latents:
            torch.save(latents.float().cpu(), os.path.join(dirs["latents"], f"{stem}.pt"))

        line = f"[{i + 1}/{len(paths)}] {stem}"
        if args.gt_dir:
            gt_path = find_gt(stem, gt_index, args.dataset)
            if gt_path is None:
                no_gt.append(stem)
            else:
                gt = Image.open(gt_path).convert("RGB")
                m_b = compute_metrics(img_b, gt, aligner, args.eval_res)
                m_s = compute_metrics(img_s, gt, aligner, args.eval_res)
                if m_b is None or m_s is None:
                    no_gt.append(stem)  # 尺寸无法对齐，视同缺GT，不计入统计
                else:
                    metrics_b.append(m_b)
                    metrics_s.append(m_s)
                    fmt = lambda m: f"psnr={m['psnr']:.2f} ssim={m['ssim']:.4f}" + (
                        f" S={m['S']:.4f}" if "S" in m else ""
                    )
                    line += f"  baseline[{fmt(m_b)}]  skip[{fmt(m_s)}]"
        print(line)

    print(f"\n完成。baseline 输出: {dirs['baseline']}\n     skip 输出: {dirs['skip']}")

    if args.gt_dir:
        if no_gt:
            print(f"\n{len(no_gt)} 张没有对应GT或尺寸无法对齐，未计入统计: {no_gt[:10]}"
                  + (" ..." if len(no_gt) > 10 else ""))
        if metrics_b:
            keys = metrics_b[0].keys()
            print(f"\n===== 均值 (N={len(metrics_b)}) =====")
            print(f"{'metric':<8}{'baseline':>12}{'skip':>12}{'Δ':>10}")
            for k in keys:
                mb = float(np.mean([m[k] for m in metrics_b]))
                ms = float(np.mean([m[k] for m in metrics_s]))
                print(f"{k:<8}{mb:>12.4f}{ms:>12.4f}{ms - mb:>+10.4f}")


if __name__ == "__main__":
    main()