"""
eval_infer.py

评测 infer_dehaze.py 产出的若干输出目录（如 baseline / skip）相对GT的指标，并做配对比较。
兼容 RESIDE(SOTS/ITS/OTS)、NH-HAZE、Dense-Haze。

指标：PSNR、SSIM(skimage, RGB)、SSIM三项均值(L/C/S，复用ssim_decompose.py)、可选LPIPS。

--dataset   文件名匹配规则（auto 会依次尝试所有规则）
              reside    : 1400_1.png / 0001_0.8_0.2.jpg / 1_1_0.9.png -> 1400.png / 0001.png / 1.png
              nhhaze    : 01_hazy.png -> 01_GT.png
              densehaze : 01_hazy.png -> 01_GT.png
--ids       只评测部分编号，如 "51-55" 或 "51,53,55"（NH-HAZE / Dense-Haze 常用 51-55 做测试）
--eval_res  pipeline : 在pipeline实际输出分辨率下评测（默认；GT按同样规则对齐）
            original : 输出上采样回GT原分辨率再评测（更接近论文的全分辨率协议，
                       但请核对你要对比的那篇论文的具体协议）

用法示例：
    python eval_infer.py --dataset nhhaze --ids 51-55 \
        --pred_dirs baseline=./out_nh/baseline skip=./out_nh/skip \
        --gt_dir /data/NH-HAZE/GT --out_csv ./out_nh/metrics.csv --lpips
"""

import argparse
import csv
import os
import re

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
HIGHER_BETTER = {"psnr": True, "ssim": True, "L": True, "C": True, "S": True, "lpips": False}


# ---------------------------------------------------------------- 文件匹配
def index_dir(d):
    return {os.path.splitext(f)[0].lower(): os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(IMG_EXTS)}


def gt_candidates(stem, dataset, hazy_suffix, gt_suffix):
    """按优先级返回可能的GT文件名(stem, 小写)。"""
    s = stem.lower()
    cands = []
    if hazy_suffix or gt_suffix:  # 用户显式指定的规则优先
        base = s[: -len(hazy_suffix)] if hazy_suffix and s.endswith(hazy_suffix.lower()) else s
        cands.append(base + gt_suffix.lower())

    hazy_gt = [(s[:-5] if s.endswith("_hazy") else s) + "_gt"]  # NH-HAZE / Dense-Haze
    reside = [s.split("_")[0]]  # RESIDE: 编号_雾浓度... -> 编号
    if dataset == "reside":
        cands += reside
    elif dataset in ("nhhaze", "densehaze"):
        cands += hazy_gt
    else:
        cands += hazy_gt + reside

    return cands + [s, s.replace("hazy", "gt"), s.replace("_hazy", "")]


def parse_ids(spec):
    ids = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            ids.update(range(int(a), int(b) + 1))
        elif part:
            ids.add(int(part))
    return ids


def leading_int(stem):
    m = re.match(r"\d+", stem)
    return int(m.group()) if m else None


# ---------------------------------------------------------------- 分辨率对齐
def make_gt_aligner():
    """pipeline分辨率模式：用与 infer_dehaze.py / pipeline 完全相同的规则处理GT（>1MP缩放 -> 16整数倍裁剪）"""
    from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor

    proc = Flux2ImageProcessor(vae_scale_factor=16)

    def align(img: Image.Image) -> Image.Image:
        w, h = img.size
        if w * h > 1024 * 1024:
            img = proc._resize_to_target_area(img, 1024 * 1024)
            w, h = img.size
        w, h = (w // 16) * 16, (h // 16) * 16
        t = proc.preprocess(img, height=h, width=w, resize_mode="crop")
        return proc.postprocess(t, output_type="pil")[0]

    return align


def original_region(pred_size, gt_size, target_area=1024 * 1024):
    """
    original模式：pipeline对>target_area的图先用 _resize_to_target_area 等比缩小
    (scale=sqrt(target_area/面积)，尺寸用int()截断，不是round)，否则不缩放，
    再把宽高各自独立裁到16的整数倍。

    两种情况分开处理，不能用同一个公式:
    - 面积<=target_area(没发生缩放，比如RESIDE SOTS这类小图): s=1 是精确值，不是近似，
      因为根本没有调用过缩放函数，没有任何需要反推的东西。
    - 面积>target_area(发生过缩放，比如NH-HAZE/Dense-Haze): 真实的中间尺寸用的是
      int()截断而不是四舍五入，如果直接用公式 s=sqrt(target_area/面积) 重新计算再反推，
      截断引入的1-2像素误差经过除以较小的s(大图通常在0.5~0.7)会被放大到2-3像素——
      改用"直接从真实产出的pred尺寸反推"(s=max(w/W,h/H))更可靠，因为它是从真实发生过的
      结果里反着推出来的，天然包含了int()截断这类实现细节，不需要重新精确复刻这些细节。
    """
    (w, h), (W, H) = pred_size, gt_size
    s = 1.0 if W * H <= target_area else max(w / W, h / H)
    rw, rh = min(W, round(w / s)), min(H, round(h / s))
    left, top = (W - rw) // 2, (H - rh) // 2
    return (left, top, left + rw, top + rh)


# ---------------------------------------------------------------- 指标
def build_metric_fn(use_lpips, device):
    try:
        from ssim_decompose import ssim_decompose  # 同目录下的诊断脚本

        has_terms = True
    except ImportError:
        has_terms = False
        print("[warn] 找不到 ssim_decompose.py，跳过 SSIM 三项(L/C/S)")

    lp_net = None
    if use_lpips:
        import lpips

        lp_net = lpips.LPIPS(net="alex").to(device).eval()

    def fn(pred: Image.Image, gt: Image.Image):
        a, b = np.asarray(pred), np.asarray(gt)
        m = {
            "psnr": peak_signal_noise_ratio(b, a, data_range=255),
            "ssim": structural_similarity(a, b, channel_axis=2, data_range=255),
        }
        if has_terms:
            ga = np.asarray(pred.convert("L"), dtype=np.float64)
            gb = np.asarray(gt.convert("L"), dtype=np.float64)
            l, c, s, *_ = ssim_decompose(ga, gb)
            m.update(L=float(l.mean()), C=float(c.mean()), S=float(s.mean()))
        if lp_net is not None:
            import torch

            to_t = lambda x: torch.from_numpy(x).permute(2, 0, 1)[None].float().to(device) / 127.5 - 1
            with torch.no_grad():
                m["lpips"] = lp_net(to_t(a), to_t(b)).item()
        return m

    return fn


def bootstrap_ci(delta, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    means = [rng.choice(delta, size=len(delta), replace=True).mean() for _ in range(n)]
    return np.percentile(means, [2.5, 97.5])


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dirs", nargs="+", required=True, help="name=dir，第一个为baseline")
    p.add_argument("--gt_dir", required=True)
    p.add_argument("--dataset", choices=["auto", "reside", "nhhaze", "densehaze"], default="auto")
    p.add_argument("--ids", default=None, help='只评测这些编号，如 "51-55" 或 "51,53,55"')
    p.add_argument("--eval_res", choices=["pipeline", "original"], default="pipeline")
    p.add_argument("--hazy_suffix", default="", help="自定义规则：输出stem里hazy图的后缀")
    p.add_argument("--gt_suffix", default="", help="自定义规则：GT stem相对基名的后缀")
    p.add_argument("--out_csv", default="./metrics.csv")
    p.add_argument("--lpips", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--topk", type=int, default=5)
    args = p.parse_args()

    variants = dict(x.split("=", 1) for x in args.pred_dirs)
    names = list(variants)
    ref = names[0]
    indices = {n: index_dir(d) for n, d in variants.items()}
    gt_index = index_dir(args.gt_dir)
    id_filter = parse_ids(args.ids) if args.ids else None
    align = make_gt_aligner() if args.eval_res == "pipeline" else None
    metric_fn = build_metric_fn(args.lpips, args.device)

    print(f"dataset={args.dataset}  eval_res={args.eval_res}  ids={args.ids or 'all'}  variants={names}")

    rows, skipped = [], []
    for stem in sorted(indices[ref]):
        if not all(stem in indices[n] for n in names):
            skipped.append((stem, "并非所有变体都有该输出"))
            continue

        cands = gt_candidates(stem, args.dataset, args.hazy_suffix, args.gt_suffix)
        gt_path = next((gt_index[c] for c in cands if c in gt_index), None)
        if gt_path is None:
            skipped.append((stem, f"找不到GT，尝试过 {cands[:3]}"))
            continue

        if id_filter is not None:  # 主动过滤，不算"跳过"
            gid = leading_int(os.path.splitext(os.path.basename(gt_path))[0])
            if gid not in id_filter:
                continue

        preds = {n: Image.open(indices[n][stem]).convert("RGB") for n in names}
        if len({im.size for im in preds.values()}) != 1:
            skipped.append((stem, "各变体输出尺寸不一致"))
            continue
        gt = Image.open(gt_path).convert("RGB")

        if args.eval_res == "original":
            gt = gt.crop(original_region(preds[ref].size, gt.size))
            preds = {n: im.resize(gt.size, Image.BICUBIC) for n, im in preds.items()}
        elif gt.size != preds[ref].size:
            gt = align(gt)
            if gt.size != preds[ref].size:
                skipped.append((stem, f"尺寸无法对齐 pred={preds[ref].size} gt={gt.size}"))
                continue

        row = {"name": stem}
        for n in names:
            for k, v in metric_fn(preds[n], gt).items():
                row[f"{n}_{k}"] = v
        rows.append(row)
        print(f"[{len(rows)}] {stem}  " + "  ".join(f"{n}: {row[f'{n}_psnr']:.2f}dB/{row[f'{n}_ssim']:.4f}" for n in names))

    if skipped:
        print(f"\n跳过 {len(skipped)} 张：")
        for s, why in skipped[:10]:
            print(f"  {s}: {why}")
    if not rows:
        raise SystemExit("没有任何可评测的图片，检查 --gt_dir / --dataset / --ids")

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    metrics = [k[len(ref) + 1:] for k in rows[0] if k.startswith(ref + "_")]

    print(f"\n===== 均值 (N={len(rows)}) =====")
    print(f"{'metric':<8}" + "".join(f"{n:>12}" for n in names))
    for k in metrics:
        print(f"{k:<8}" + "".join(f"{np.mean([r[f'{n}_{k}'] for r in rows]):>12.4f}" for n in names))

    if len(rows) < 20:
        print(f"\n[注意] N={len(rows)} 样本量很小，下面的 95% CI 和改善占比只能作参考，别据此下结论。")

    for n in names[1:]:
        print(f"\n===== 配对比较: {n} vs {ref} =====")
        print(f"{'metric':<8}{'Δ mean':>10}{'95% CI':>22}{'改善占比':>10}")
        for k in metrics:
            d = np.array([r[f"{n}_{k}"] - r[f"{ref}_{k}"] for r in rows])
            lo, hi = bootstrap_ci(d)
            better = (d > 0) if HIGHER_BETTER.get(k, True) else (d < 0)
            print(f"{k:<8}{d.mean():>+10.4f}{f'[{lo:+.4f}, {hi:+.4f}]':>22}{better.mean() * 100:>9.1f}%")

        d_ssim = np.array([r[f"{n}_ssim"] - r[f"{ref}_ssim"] for r in rows])
        order = np.argsort(d_ssim)
        k = min(args.topk, len(rows))
        print(f"\n{n} 相对 {ref} —— SSIM提升最大 / 下降最大的样本（可拿去跑 ssim_decompose.py）：")
        for i in order[::-1][:k]:
            print(f"  {d_ssim[i]:+.4f}  {rows[i]['name']}")
        print("  ...")
        for i in order[:k]:
            print(f"  {d_ssim[i]:+.4f}  {rows[i]['name']}")

    print(f"\n逐图明细已保存: {args.out_csv}")


if __name__ == "__main__":
    main()