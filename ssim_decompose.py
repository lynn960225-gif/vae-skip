"""
ssim_decompose.py

把SSIM拆解成 luminance / contrast / structure 三项局部map，分别可视化，
用来诊断SSIM低到底是因为：
  - 局部亮度/曝光不对（luminance map 低）—— 指向色偏/曝光问题
  - 局部对比度不够（contrast map 低）—— 指向输出发糊/动态范围压缩
  - 局部结构/纹理对不上（structure map 低）—— 指向内容错位/幻觉问题（扩散模型生成了不对齐的细节）

用法:
    python ssim_decompose.py --dehazed dehazed.png --gt gt.png --out out_dir/
"""

import argparse
import os

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter
import matplotlib.pyplot as plt


def to_gray_float(img_path):
    img = Image.open(img_path).convert("L")  # 转灰度，SSIM标准做法是在亮度通道上算
    return np.asarray(img, dtype=np.float64)


def local_stats(x, y, win_size=11):
    """用均匀窗口(box filter)算局部均值、方差、协方差，等价于skimage默认的滑动窗口SSIM"""
    mu_x = uniform_filter(x, size=win_size)
    mu_y = uniform_filter(y, size=win_size)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = uniform_filter(x * x, size=win_size) - mu_x2
    sigma_y2 = uniform_filter(y * y, size=win_size) - mu_y2
    sigma_xy = uniform_filter(x * y, size=win_size) - mu_xy

    # 数值误差可能导致方差略小于0，clip一下
    sigma_x2 = np.clip(sigma_x2, 0, None)
    sigma_y2 = np.clip(sigma_y2, 0, None)

    return mu_x, mu_y, sigma_x2, sigma_y2, sigma_xy


def ssim_decompose(pred, gt, win_size=11, data_range=255.0):
    """返回 (luminance_map, contrast_map, structure_map, ssim_map)，每个都是跟原图一样大的2D array"""
    K1, K2 = 0.01, 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    C3 = C2 / 2

    mu_x, mu_y, sigma_x2, sigma_y2, sigma_xy = local_stats(pred, gt, win_size)
    sigma_x = np.sqrt(sigma_x2)
    sigma_y = np.sqrt(sigma_y2)

    l_map = (2 * mu_x * mu_y + C1) / (mu_x ** 2 + mu_y ** 2 + C1)
    c_map = (2 * sigma_x * sigma_y + C2) / (sigma_x2 + sigma_y2 + C2)
    s_map = (sigma_xy + C3) / (sigma_x * sigma_y + C3)

    ssim_map = l_map * c_map * s_map
    return l_map, c_map, s_map, ssim_map, mu_x, mu_y, sigma_x, sigma_y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dehazed", type=str, required=True)
    parser.add_argument("--gt", type=str, required=True)
    parser.add_argument("--out", type=str, default="./ssim_decompose_out")
    parser.add_argument("--win_size", type=int, default=11)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    pred = to_gray_float(args.dehazed)
    gt = to_gray_float(args.gt)
    assert pred.shape == gt.shape, f"两张图尺寸不一致: {pred.shape} vs {gt.shape}"

    l_map, c_map, s_map, ssim_map, mu_pred, mu_gt, sigma_pred, sigma_gt = ssim_decompose(
        pred, gt, win_size=args.win_size
    )

    print(f"整体SSIM(重新计算，应该跟你之前算的差不多): {ssim_map.mean():.4f}")
    print(f"luminance项均值:  {l_map.mean():.4f}  (越接近1说明局部亮度越匹配)")
    print(f"contrast项均值:   {c_map.mean():.4f}  (越接近1说明局部对比度越匹配)")
    print(f"structure项均值:  {s_map.mean():.4f}  (越接近1说明局部结构/纹理越对齐)")
    print(f"pred局部标准差均值: {sigma_pred.mean():.4f}  gt局部标准差均值: {sigma_gt.mean():.4f}")

    # 额外算一个"局部亮度差"的直接map（比luminance项更直观，正负号能看出是偏亮还是偏暗）
    local_brightness_diff = mu_pred - mu_gt
    # 局部标准差差值：负值=pred比gt更平滑/细节更少（真实的细节丢失），
    # 接近0=两边局部纹理量差不多（此时structure项低更可能是数值不稳定，不是真实细节丢失）
    local_std_diff = sigma_pred - sigma_gt

    fig, axes = plt.subplots(3, 3, figsize=(16, 15))

    pred_img = np.asarray(Image.open(args.dehazed).convert("RGB"))
    gt_img = np.asarray(Image.open(args.gt).convert("RGB"))

    # ---- 第一行：原图 + SSIM总图 ----
    axes[0, 0].imshow(pred_img)
    axes[0, 0].set_title("Dehazed")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(gt_img)
    axes[0, 1].set_title("GT")
    axes[0, 1].axis("off")

    im2 = axes[0, 2].imshow(ssim_map, cmap="viridis", vmin=0, vmax=1)
    axes[0, 2].set_title(f"SSIM map (mean={ssim_map.mean():.3f})")
    axes[0, 2].axis("off")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    # ---- 第二行：SSIM三项分解 ----
    im3 = axes[1, 0].imshow(l_map, cmap="viridis", vmin=0, vmax=1)
    axes[1, 0].set_title(f"Luminance项 (mean={l_map.mean():.3f})\n低=局部亮度/曝光不匹配")
    axes[1, 0].axis("off")
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)

    im4 = axes[1, 1].imshow(c_map, cmap="viridis", vmin=0, vmax=1)
    axes[1, 1].set_title(f"Contrast项 (mean={c_map.mean():.3f})\n低=局部对比度不匹配（发糊/动态范围压缩）")
    axes[1, 1].axis("off")
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)

    im5 = axes[1, 2].imshow(s_map, cmap="viridis", vmin=0, vmax=1)
    axes[1, 2].set_title(f"Structure项 (mean={s_map.mean():.3f})\n低=局部纹理/边缘对不齐（幻觉/错位 或 数值不稳定）")
    axes[1, 2].axis("off")
    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046)

    # ---- 第三行：亮度差 + 局部标准差对比（区分"真实细节丢失" vs "SSIM数值不稳定"） ----
    im6 = axes[2, 0].imshow(local_brightness_diff, cmap="coolwarm", vmin=-50, vmax=50)
    axes[2, 0].set_title("局部亮度差 (pred-gt)\n红=偏亮 蓝=偏暗")
    axes[2, 0].axis("off")
    plt.colorbar(im6, ax=axes[2, 0], fraction=0.046)

    # pred和gt的局部标准差用同一组vmin/vmax，方便直接肉眼比较谁的纹理量更大
    std_vmax = max(sigma_pred.max(), sigma_gt.max()) * 0.8
    im7 = axes[2, 1].imshow(sigma_pred, cmap="magma", vmin=0, vmax=std_vmax)
    axes[2, 1].set_title(f"Pred局部标准差 (mean={sigma_pred.mean():.2f})\n越亮=局部纹理/细节越丰富")
    axes[2, 1].axis("off")
    plt.colorbar(im7, ax=axes[2, 1], fraction=0.046)

    im8 = axes[2, 2].imshow(local_std_diff, cmap="coolwarm", vmin=-15, vmax=15)
    axes[2, 2].set_title("局部标准差差 (pred_std - gt_std)\n蓝=pred比GT更平滑(真实细节丢失)\n红=pred比GT纹理更多(可能过度生成)")
    axes[2, 2].axis("off")
    plt.colorbar(im8, ax=axes[2, 2], fraction=0.046)

    plt.tight_layout()
    out_path = os.path.join(args.out, "ssim_decompose.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"可视化已保存到: {out_path}")


if __name__ == "__main__":
    main()