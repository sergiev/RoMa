import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from romatch.models.model_zoo import roma_model
import matplotlib.pyplot as plt
from romatch.utils import warp_kpts


def build_model(weights_path: str, resolution=(640, 640)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.float16 if device.type == "cuda" else torch.float32
    weights = torch.load(weights_path, map_location="cpu")
    if weights.get("model"):
        weights = weights["model"]
    matcher = roma_model(
        resolution=resolution,
        upsample_preds=False,
        weights=weights,
        dinov2_weights=None,
        device=device,
        amp_dtype=amp_dtype,
    )
    matcher.symmetric = False
    return matcher, device


def draw_keypoints(im: np.ndarray, points: np.ndarray, color=(0, 255, 0)):
    out = im.copy()
    for x, y in points.astype(int):
        cv2.circle(out, (int(x), int(y)), 2, color, -1, lineType=cv2.LINE_AA)
    return out


def draw_matches(imA: np.ndarray, imB: np.ndarray, ptsA: np.ndarray, ptsB: np.ndarray, color=(255, 0, 0)):
    hA, wA = imA.shape[:2]
    hB, wB = imB.shape[:2]
    H = max(hA, hB)
    canvas = np.zeros((H, wA + wB, 3), dtype=np.uint8)
    canvas[:hA, :wA] = imA
    canvas[:hB, wA : wA + wB] = imB
    for (x1, y1), (x2, y2) in zip(ptsA.astype(int), ptsB.astype(int)):
        cv2.line(canvas, (int(x1), int(y1)), (int(x2) + wA, int(y2)), color, 1, lineType=cv2.LINE_AA)
    return canvas


def visualize_matches(im_A, im_B, matches, certainty, depth1, depth2, T_1to2, K1, K2):
    """Расширенная визуализация соответствий между парой изображений

    Создает 3 визуализации:
    1. Карта ошибок с цветовой кодировкой (зеленый->желтый->красный)
    2. Соответствия с линиями
    3. Warped изображение (результат "сшивки")
    """
    import torch.nn.functional as F
    from matplotlib.colors import LinearSegmentedColormap

    h, w = matches.shape[:2]

    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=im_A.device).view(3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=im_A.device).view(3, 1, 1)

    im_A_vis = torch.clamp(im_A * imagenet_std + imagenet_mean, 0, 1).detach().cpu()
    im_B_vis = torch.clamp(im_B * imagenet_std + imagenet_mean, 0, 1).detach().cpu()

    im_A_np = im_A_vis.permute(1, 2, 0).numpy()
    im_B_np = im_B_vis.permute(1, 2, 0).numpy()

    x1_norm = matches[..., :2].reshape(1, h * w, 2)
    mask_gt, x2_gt = warp_kpts(
        x1_norm.double(),
        depth1[None].double(),
        depth2[None].double(),
        T_1to2[None, :3, :].double(),
        K1[None].double(),
        K2[None].double(),
    )

    x2_pred_norm = matches[..., 2:].reshape(1, h * w, 2)

    x2_gt_px = torch.stack(
        (
            w * (x2_gt[0, :, 0] + 1) / 2,
            h * (x2_gt[0, :, 1] + 1) / 2,
        ),
        dim=1,
    )
    x2_pred_px = torch.stack(
        (
            w * (x2_pred_norm[0, :, 0] + 1) / 2,
            h * (x2_pred_norm[0, :, 1] + 1) / 2,
        ),
        dim=1,
    )

    error = (x2_pred_px - x2_gt_px).norm(dim=1)
    error_map = error.reshape(h, w)

    valid_mask = mask_gt[0].reshape(h, w).bool()
    valid_mask_flat = valid_mask.view(-1)
    error_flat = error.view(-1)
    valid_errors = error_flat[valid_mask_flat]

    mean_error = valid_errors.mean().item() if valid_errors.numel() > 0 else float("nan")
    median_error = valid_errors.median().item() if valid_errors.numel() > 0 else float("nan")
    ratio_1 = (valid_errors < 1.0).float().mean().item() if valid_errors.numel() > 0 else 0.0
    ratio_3 = (valid_errors < 3.0).float().mean().item() if valid_errors.numel() > 0 else 0.0
    ratio_5 = (valid_errors < 5.0).float().mean().item() if valid_errors.numel() > 0 else 0.0

    error_map_np = error_map.detach().cpu().numpy()
    valid_mask_np = valid_mask.detach().cpu().numpy()

    warped_B = F.grid_sample(
        im_B.unsqueeze(0),
        grid=matches[..., 2:].unsqueeze(0),
        mode="bilinear",
        align_corners=False,
    )[0]
    warped_B_np = warped_B.detach().cpu().permute(1, 2, 0).numpy()

    # === 3. ВИЗУАЛИЗАЦИЯ ===
    fig_total = plt.figure(figsize=(20, 12))
    gs = fig_total.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # --- Строка 1: Исходные изображения и warped ---
    ax1 = fig_total.add_subplot(gs[0, 0])
    ax1.imshow(im_A_np)
    ax1.set_title("Image A (Target)", fontsize=12)
    ax1.axis("off")

    ax2 = fig_total.add_subplot(gs[0, 1])
    ax2.imshow(im_B_np)
    ax2.set_title("Image B (Source)", fontsize=12)
    ax2.axis("off")

    ax3 = fig_total.add_subplot(gs[0, 2])
    ax3.imshow(warped_B_np)
    ax3.set_title("Image B warped to A", fontsize=12)
    ax3.axis("off")

    # --- Строка 2: Карта ошибок и overlay ---
    ax4 = fig_total.add_subplot(gs[1, 0])

    # Создаем custom colormap: зеленый -> желтый -> красный
    colors_list = ["green", "yellow", "orange", "red"]
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list("error_cmap", colors_list, N=n_bins)

    # Ошибки: 0-1px зеленый, 1-3px желтый, 3-5px оранжевый, >5px красный
    error_display = np.ma.masked_where(~valid_mask_np, error_map_np)
    im_err = ax4.imshow(error_display, cmap=cmap, vmin=0, vmax=5, interpolation="nearest")
    ax4.set_title(f"Error Map (pixels)\nMean: {mean_error:.2f}px", fontsize=12)
    ax4.axis("off")

    # Colorbar
    cbar = plt.colorbar(im_err, ax=ax4, fraction=0.046, pad=0.04)
    cbar.set_label("Error (pixels)", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    # --- Overlay: Image A + warped B ---
    ax5 = fig_total.add_subplot(gs[1, 1])
    # Blend images
    overlay = np.clip(im_A_np+warped_B_np, 0, 1)
    ax5.imshow(overlay)
    ax5.set_title("Overlay (A + warped B)", fontsize=12)
    ax5.axis("off")

    # --- Difference map ---
    ax6 = fig_total.add_subplot(gs[1, 2])
    diff = np.abs(im_A_np - warped_B_np).mean(axis=2)
    ax6.imshow(diff, cmap="hot", vmin=0, vmax=0.3)
    ax6.set_title("Absolute Difference", fontsize=12)
    ax6.axis("off")

    # --- Строка 3: Соответствия с цветовой кодировкой по ошибке ---
    ax7 = fig_total.add_subplot(gs[2, :])

    # Создаем side-by-side view
    combined = np.concatenate([im_A_np, im_B_np], axis=1)
    ax7.imshow(combined)
    ax7.set_title("Matches colored by error (green=good, red=bad)", fontsize=12)
    ax7.axis("off")

    # Выбираем точки для визуализации
    certainty_np = certainty.detach().cpu().numpy()
    threshold = np.percentile(certainty_np, 90)
    high_cert_mask = certainty_np > threshold

    # Сэмплируем точки
    y_coords, x_coords = np.where(high_cert_mask & valid_mask_np)
    if len(y_coords) > 200:
        indices = np.random.choice(len(y_coords), 200, replace=False)
        y_coords = y_coords[indices]
        x_coords = x_coords[indices]

    if len(y_coords) > 0:
        # Координаты точек
        kpts0 = np.stack([x_coords, y_coords], axis=1)

        matches_np = matches.detach().cpu().numpy()
        kpts1_norm = matches_np[y_coords, x_coords, 2:]
        kpts1 = np.stack([w * (kpts1_norm[:, 0] + 1) / 2, h * (kpts1_norm[:, 1] + 1) / 2], axis=1)

        # Цвета на основе ошибки
        errors = error_map_np[y_coords, x_coords]
        # Нормализуем к [0, 1] для colormap
        errors_norm = np.clip(errors / 5.0, 0, 1)  # 0-5px -> 0-1
        colors = cmap(errors_norm)

        # Рисуем линии
        for i in range(len(kpts0)):
            x1, y1 = kpts0[i]
            x2, y2 = kpts1[i, 0] + w, kpts1[i, 1]  # сдвиг для второго изображения

            ax7.plot([x1, x2], [y1, y2], color=colors[i], linewidth=1.5, alpha=0.6)

        # Рисуем точки
        ax7.scatter(kpts0[:, 0], kpts0[:, 1], c=colors, s=30, zorder=2, edgecolors="white", linewidths=0.5)
        ax7.scatter(kpts1[:, 0] + w, kpts1[:, 1], c=colors, s=30, zorder=2, edgecolors="white", linewidths=0.5)

    # === Статистика ===
    valid_total = valid_mask_flat.sum().item()
    stats_text = f"""Statistics:
Mean error: {mean_error:.2f}px
Median error: {median_error:.2f}px
<1px (green): {ratio_1*100:.1f}%
<3px (yellow): {ratio_3*100:.1f}%
<5px (orange): {ratio_5*100:.1f}%
Valid points: {valid_total}/{valid_mask_np.size}"""

    fig_total.text(
        0.02,
        0.02,
        stats_text,
        fontsize=10,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # Сохраняем
    visual = [fig_total]
    # plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=150)
    plt.close()
    # Также сохраняем отдельно warped изображение для удобства
    fig_warp, ax_warp = plt.subplots(1, 3, figsize=(15, 5))
    ax_warp[0].imshow(im_A_np)
    ax_warp[0].set_title("Target (A)")
    ax_warp[0].axis("off")

    ax_warp[1].imshow(warped_B_np)
    ax_warp[1].set_title("Warped Source (B→A)")
    ax_warp[1].axis("off")

    ax_warp[2].imshow(overlay)
    ax_warp[2].set_title("Blend")
    ax_warp[2].axis("off")

    plt.tight_layout()
    visual.append(fig_warp) 
    plt.close()
    return visual


def main():
    parser = argparse.ArgumentParser(description="Visualize pipeline: kpts, matches, RANSAC, warp result")
    parser.add_argument(
        "--image0",
        default="/home/sema/radar/datasets/umbra_tiles/Belize/00596/2024-09-04-03-14-36_UMBRA-05_GEC.tif",
        type=str,
        help="Path to first image",
    )
    parser.add_argument(
        "--image1",
        default="/home/sema/radar/datasets/umbra_tiles/Belize/00596/2024-09-04-16-40-16_UMBRA-08_GEC.tif",
        type=str,
        help="Path to second image",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="/home/sema/radar/RoMa/workspace/checkpoints/train_roma_outdoor_latest.pth",
        help="Path to RoMa weights (.pth)",
    )
    parser.add_argument("--resolution", type=int, nargs=2, default=(1008, 1008), help="Coarse resolution for RoMa")
    parser.add_argument("--out_dir", type=str, default="viz_out", help="Output directory for visualizations")
    parser.add_argument("--max_points", type=int, default=3000)
    parser.add_argument("--certainty_thresh", type=float, default=0.0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    matcher, device = build_model(args.weights, tuple(args.resolution))

    warp, certainty = matcher.match(args.image0, args.image1, device=device)
    Hh = warp.shape[0]
    Ww = warp.shape[1] // 2 if matcher.symmetric else warp.shape[1]

    # 1) Ключевые точки (сэмплируем по certainty)
    mask = (certainty > args.certainty_thresh).flatten()
    coords = warp.reshape(-1, 4)[mask]
    if coords.shape[0] == 0:
        coords = warp.reshape(-1, 4)
    if coords.shape[0] > args.max_points:
        idx = torch.randperm(coords.shape[0])[: args.max_points]
        coords = coords[idx]
    kptsA_px, kptsB_px = matcher.to_pixel_coordinates(coords, Hh, Ww, Hh, Ww)
    kptsA_px_np = kptsA_px.cpu().numpy()
    kptsB_px_np = kptsB_px.cpu().numpy()

    imA = cv2.imread(args.image0, cv2.IMREAD_GRAYSCALE)
    imB = cv2.imread(args.image1, cv2.IMREAD_GRAYSCALE)
    imA_c3 = cv2.cvtColor(imA, cv2.COLOR_GRAY2BGR)
    imB_c3 = cv2.cvtColor(imB, cv2.COLOR_GRAY2BGR)

    kpA_img = draw_keypoints(imA_c3, kptsA_px_np, (0, 255, 0))
    kpB_img = draw_keypoints(imB_c3, kptsB_px_np, (0, 255, 0))
    cv2.imwrite(os.path.join(args.out_dir, "01_keypoints_A.png"), kpA_img)
    cv2.imwrite(os.path.join(args.out_dir, "01_keypoints_B.png"), kpB_img)

    # 2) Сопоставления
    matches_img = draw_matches(imA_c3, imB_c3, kptsA_px_np, kptsB_px_np, (255, 0, 0))
    cv2.imwrite(os.path.join(args.out_dir, "02_matches.png"), matches_img)

    # 3) RANSAC
    Hmat, inliers = cv2.findHomography(
        kptsA_px_np.astype(np.float32),
        kptsB_px_np.astype(np.float32),
        cv2.USAC_MAGSAC,
        ransacReprojThreshold=8.0,
        confidence=0.999,
        maxIters=10000,
    )
    if inliers is None:
        inlier_mask = np.zeros((kptsA_px_np.shape[0],), dtype=bool)
    else:
        inlier_mask = inliers.flatten().astype(bool)
    matches_inlier_img = draw_matches(imA_c3, imB_c3, kptsA_px_np[inlier_mask], kptsB_px_np[inlier_mask], (0, 255, 0))
    matches_outlier_img = draw_matches(
        imA_c3, imB_c3, kptsA_px_np[~inlier_mask], kptsB_px_np[~inlier_mask], (0, 0, 255)
    )
    cv2.imwrite(os.path.join(args.out_dir, "03_matches_inliers.png"), matches_inlier_img)
    cv2.imwrite(os.path.join(args.out_dir, "03_matches_outliers.png"), matches_outlier_img)

    # 4) Гомографическое преобразование второго изображения
    if Hmat is not None:
        warped_B = cv2.warpPerspective(imB_c3, Hmat, (imA_c3.shape[1], imA_c3.shape[0]))
        cv2.imwrite(os.path.join(args.out_dir, "04_warped_B_to_A.png"), warped_B)

    print("Saved visualizations to:", args.out_dir)


if __name__ == "__main__":
    main()
