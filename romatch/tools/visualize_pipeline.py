import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from romatch.models.model_zoo import roma_model
import matplotlib.pyplot as plt
from romatch.utils import warp_kpts_planar, get_inverse_T, warp_kpts, get_gt_warp
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap, Normalize


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


def _get_cv2_affine_matrix(T_norm, h, w):
    """Converts a normalized transformation matrix to a pixel-based affine matrix for cv2."""
    R = T_norm[:2, :2].cpu().numpy()
    t = T_norm[:2, 3].cpu().numpy()
    M_cv2 = np.zeros((2, 3))
    M_cv2[:2, :2] = R
    M_cv2[0, 2] = (w / 2) * (-R[0, 0] - R[0, 1] + t[0]) + w / 2
    M_cv2[1, 2] = (h / 2) * (-R[1, 0] - R[1, 1] + t[1]) + h / 2
    return M_cv2

def visualize_matches(
    ax, im_A_vis, im_B_vis, matches, certainty, valid_mask_fwd_np, h, w, cmap, error_map_forward_np, norm
):

    combined_np = np.concatenate(
        [
            (im_A_vis.permute(1, 2, 0).numpy() * 255).astype(np.uint8),
            (im_B_vis.permute(1, 2, 0).numpy() * 255).astype(np.uint8),
        ],
        axis=1,
    )
    ax.imshow(combined_np)
    ax.set_title("Matches colored by error (A -> B)", fontsize=12)
    ax.axis("off")

    certainty_np = certainty.detach().cpu().numpy()
    goal_q = 200
    n_points = np.prod(certainty_np.shape)
    threshold = np.percentile(certainty_np, 100 - (goal_q + 1) / n_points * 100)
    high_cert_mask = certainty_np > threshold

    y_coords, x_coords = np.where(high_cert_mask & valid_mask_fwd_np)
    print(len(y_coords))
    if len(y_coords) > 0:
        kpts0 = np.stack([x_coords, y_coords], axis=1)

        matches_np = matches.detach().cpu().numpy()
        kpts1_norm = matches_np[y_coords, x_coords, 2:]
        kpts1 = np.stack([w * (kpts1_norm[:, 0] + 1) / 2, h * (kpts1_norm[:, 1] + 1) / 2], axis=1)

        errors = error_map_forward_np[y_coords, x_coords]
        errors_norm = norm(errors)
        colors = cmap(errors_norm)

        for i in range(len(kpts0)):
            x1, y1 = kpts0[i]
            x2, y2 = kpts1[i, 0] + w, kpts1[i, 1]

            ax.plot([x1, x2], [y1, y2], color=colors[i], linewidth=1.5, alpha=0.6)

        ax.scatter(kpts0[:, 0], kpts0[:, 1], c=colors, s=30, zorder=2, edgecolors="white", linewidths=0.5)
        ax.scatter(kpts1[:, 0] + w, kpts1[:, 1], c=colors, s=30, zorder=2, edgecolors="white", linewidths=0.5)


def _draw_quadrant_center_correspondences(
    ax, im_A_np, im_B_np, T_1to2, T_2to1, h, w, planar_mode, depth1=None, depth2=None, K1=None, K2=None
):
    """Draws lines for center and quadrant center correspondences between two images."""
    vis_img = np.concatenate((im_A_np, im_B_np), axis=1)
    ax.imshow(vis_img)
    ax.set_title("GT Quadrant/Center Projections", fontsize=12)
    ax.axis("off")

    points_norm = torch.tensor(
        [
            [0.0, 0.0],  # Center
            [-0.5, -0.5],  # Top-Left
            [0.5, -0.5],  # Top-Right
            [-0.5, 0.5],  # Bottom-Left
            [0.5, 0.5],  # Bottom-Right
        ],
        device=T_1to2.device,
    ).reshape(1, 5, 2)

    color_A2B = "magenta"  # Фиолетовый
    color_B2A = "cyan"

    if planar_mode:
        # Project A -> B
        _, points_A_in_B_norm = warp_kpts_planar(points_norm, T_1to2.unsqueeze(0))
        # Project B -> A
        _, points_B_in_A_norm = warp_kpts_planar(points_norm, T_2to1.unsqueeze(0))
    else:
        # Project A -> B
        _, points_A_in_B_norm = warp_kpts(
            points_norm,
            depth1.unsqueeze(0),
            depth2.unsqueeze(0),
            T_1to2.unsqueeze(0),
            K1.unsqueeze(0),
            K2.unsqueeze(0),
        )
        # Project B -> A
        _, points_B_in_A_norm = warp_kpts(
            points_norm,
            depth2.unsqueeze(0),
            depth1.unsqueeze(0),
            T_2to1.unsqueeze(0),
            K2.unsqueeze(0),
            K1.unsqueeze(0),
        )

    for i in range(points_norm.shape[1]):
        # A -> B projection
        p_A_start_px = ((points_norm[0, i, 0].item() + 1) * w / 2, (points_norm[0, i, 1].item() + 1) * h / 2)
        p_A_end_px = (
            (points_A_in_B_norm[0, i, 0].item() + 1) * w / 2 + w,
            (points_A_in_B_norm[0, i, 1].item() + 1) * h / 2,
        )
        ax.plot(
            [p_A_start_px[0], p_A_end_px[0]],
            [p_A_start_px[1], p_A_end_px[1]],
            color=color_A2B,
            linewidth=1.5,
            linestyle="-",
        )
        ax.scatter(
            [p_A_start_px[0], p_A_end_px[0]],
            [p_A_start_px[1], p_A_end_px[1]],
            c=color_A2B,
            s=40,
            zorder=2,
            edgecolors="black",
            linewidths=1,
        )

        # B -> A projection
        p_B_start_px = ((points_norm[0, i, 0].item() + 1) * w / 2 + w, (points_norm[0, i, 1].item() + 1) * h / 2)
        p_B_end_px = (
            (points_B_in_A_norm[0, i, 0].item() + 1) * w / 2,
            (points_B_in_A_norm[0, i, 1].item() + 1) * h / 2,
        )
        ax.plot(
            [p_B_start_px[0], p_B_end_px[0]],
            [p_B_start_px[1], p_B_end_px[1]],
            color=color_B2A,
            linewidth=1.5,
            linestyle="-",
        )
        ax.scatter(
            [p_B_start_px[0], p_B_end_px[0]],
            [p_B_start_px[1], p_B_end_px[1]],
            c=color_B2A,
            s=40,
            zorder=2,
            edgecolors="black",
            linewidths=1,
        )


def _get_pred_overlay(matches, certainty, valid_mask_np, im_A_np, im_B_np, h, w):
    """Estimates homography from matches and creates a predicted overlay."""
    certainty_np = certainty.detach().cpu().numpy()
    threshold = np.percentile(certainty_np, 85)
    y_coords_h, x_coords_h = np.where((certainty_np > threshold) & valid_mask_np)

    if len(y_coords_h) < 4:
        return im_A_np.copy()

    kptsA_h_px = np.stack([x_coords_h, y_coords_h], axis=1).astype(np.float32)

    matches_np = matches.detach().cpu().numpy()
    kptsB_h_norm = matches_np[y_coords_h, x_coords_h, 2:]
    kptsB_h_px = np.stack([w * (kptsB_h_norm[:, 0] + 1) / 2, h * (kptsB_h_norm[:, 1] + 1) / 2], axis=1).astype(
        np.float32
    )

    H_BtoA_pred, _ = cv2.findHomography(
        kptsB_h_px, kptsA_h_px, cv2.USAC_MAGSAC, ransacReprojThreshold=8.0, confidence=0.999
    )

    if H_BtoA_pred is not None:
        im_B_warped_to_A_pred_np = cv2.warpPerspective(im_B_np, H_BtoA_pred, (w, h))
        overlay_B_on_A_pred = cv2.addWeighted(im_A_np, 0.5, im_B_warped_to_A_pred_np, 0.5, 0)
        return overlay_B_on_A_pred
    else:
        return im_A_np.copy()


def _create_gt_overlay_3d(im_to_warp_np, im_ref_np, depth_to_warp, depth_ref, T_ref_to_warp, K_ref, K_to_warp, h, w):
    """
    Creates a GT overlay by warping one image onto another using 3D information.
    Note: This uses an inverse warp (from reference to source) for cv2.remap.
    """
    with torch.no_grad():
        # For each pixel in the reference image, find its corresponding coordinate in the "to_warp" image.
        warp_norm, mask = get_gt_warp(
            depth_ref.unsqueeze(0),
            depth_to_warp.unsqueeze(0),
            T_ref_to_warp.unsqueeze(0),
            K_ref.unsqueeze(0),
            K_to_warp.unsqueeze(0),
        )
    warp_norm = warp_norm.squeeze(0).cpu().numpy()
    mask = mask.squeeze(0).cpu().numpy().astype(bool)

    map_x = (w * (warp_norm[..., 0] + 1) / 2).astype(np.float32)
    map_y = (h * (warp_norm[..., 1] + 1) / 2).astype(np.float32)

    im_warped_np = cv2.remap(
        im_to_warp_np,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    im_warped_np[~mask] = 0  # Apply mask
    overlay = cv2.addWeighted(im_ref_np, 0.5, im_warped_np, 0.5, 0)
    return overlay


def visualize_total(
    im_A, im_B, matches, certainty, T_1to2, K1=None, K2=None, depth1=None, depth2=None, planar_mode=False
):
    """Расширенная визуализация соответствий между парой изображений

    Создает комплексную визуализацию, включающую:
    1. Исходные изображения и проверку проекции центров/квадрантов.
    2. Две карты ошибок (A -> B и B -> A) с кастомной цветовой шкалой.
    3. Наложения (overlays) одного изображения на другое по GT.
    4. Линии соответствий, окрашенные по величине ошибки.
    """

    h, w = matches.shape[:2]

    # umbra constants
    mean = torch.tensor([0.221370, 0.221370, 0.221370], device=im_A.device).view(3, 1, 1)
    std = torch.tensor([0.137669, 0.137669, 0.137669], device=im_A.device).view(3, 1, 1)

    im_A_vis = torch.clamp(im_A * std + mean, 0, 1).detach().cpu()
    im_B_vis = torch.clamp(im_B * std + mean, 0, 1).detach().cpu()

    im_A_np = (im_A_vis.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    im_B_np = (im_B_vis.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    # --- 1. ВЫЧИСЛЕНИЕ ОШИБОК (A -> B и B -> A) ---
    T_2to1 = get_inverse_T(T_1to2)

    # Ошибка B -> A
    x1_norm = matches[..., :2].reshape(1, h * w, 2)
    if planar_mode:
        mask_gt_fwd, x2_gt_norm = warp_kpts_planar(x1_norm, T_1to2[None, ...])
    else:
        if depth1 is None or depth2 is None or K1 is None or K2 is None:
            raise ValueError("Depth maps and camera intrinsics are required for non-planar mode.")
        mask_gt_fwd, x2_gt_norm = warp_kpts(
            x1_norm,
            depth1.unsqueeze(0),
            depth2.unsqueeze(0),
            T_1to2.unsqueeze(0),
            K1.unsqueeze(0),
            K2.unsqueeze(0),
        )

    x2_pred_norm = matches[..., 2:].reshape(1, h * w, 2)

    x2_gt_px = torch.stack((w * (x2_gt_norm[0, :, 0] + 1) / 2, h * (x2_gt_norm[0, :, 1] + 1) / 2), dim=1)
    x2_pred_px = torch.stack((w * (x2_pred_norm[0, :, 0] + 1) / 2, h * (x2_pred_norm[0, :, 1] + 1) / 2), dim=1)

    error_forward = (x2_pred_px - x2_gt_px).norm(dim=1)
    error_map_forward = error_forward.reshape(h, w)

    # --- Восстановление корректного расчета обратной ошибки (циклическая состоятельность) ---
    # 1. Берем предсказанные точки в B (x2_pred_norm)
    # 2. Проецируем их обратно в A, используя GT-трансформацию T_2to1
    if planar_mode:
        mask_gt_bwd, x1_cycled_gt_norm = warp_kpts_planar(x2_pred_norm, T_2to1[None, ...])
    else:
        mask_gt_bwd, x1_cycled_gt_norm = warp_kpts(
            x2_pred_norm,
            depth2.unsqueeze(0),
            depth1.unsqueeze(0),
            T_2to1.unsqueeze(0),
            K2.unsqueeze(0),
            K1.unsqueeze(0),
        )

    # 3. Конвертируем исходные точки в A и "возвращенные" точки в пиксели
    x1_src_px = torch.stack((w * (x1_norm[0, :, 0] + 1) / 2, h * (x1_norm[0, :, 1] + 1) / 2), dim=1)
    x1_cycled_gt_px = torch.stack(
        (w * (x1_cycled_gt_norm[0, :, 0] + 1) / 2, h * (x1_cycled_gt_norm[0, :, 1] + 1) / 2), dim=1
    )

    # 4. Считаем ошибку как расстояние между исходной и "возвращенной" точкой
    error_backward = (x1_src_px - x1_cycled_gt_px).norm(dim=1)
    error_map_backward = error_backward.reshape(h, w)
    # --- Конец восстановления ---

    valid_mask_fwd = mask_gt_fwd[0].reshape(h, w).bool()
    valid_mask_bwd = mask_gt_bwd[0].reshape(h, w).bool()
    valid_errors = error_map_forward[valid_mask_fwd]

    mean_error = valid_errors.mean().item() if valid_errors.numel() > 0 else float("nan")
    median_error = valid_errors.median().item() if valid_errors.numel() > 0 else float("nan")
    ratio_1 = (valid_errors < 1.0).float().mean().item() if valid_errors.numel() > 0 else 0.0
    ratio_3 = (valid_errors < 3.0).float().mean().item() if valid_errors.numel() > 0 else 0.0
    ratio_5 = (valid_errors < 5.0).float().mean().item() if valid_errors.numel() > 0 else 0.0

    error_map_forward_np = error_map_forward.detach().cpu().numpy()
    error_map_backward_np = error_map_backward.detach().cpu().numpy()
    valid_mask_fwd_np = valid_mask_fwd.detach().cpu().numpy()
    valid_mask_bwd_np = valid_mask_bwd.detach().cpu().numpy()

    colors_list = ["cyan", "green", "yellow", "red"]
    nodes = [0.0, 5.0 / 15.0, 10.0 / 15.0, 1.0]
    cmap = LinearSegmentedColormap.from_list("error_cmap", list(zip(nodes, colors_list)))
    norm = Normalize(vmin=0, vmax=15)

    # --- 2. ПОДГОТОВКА OVERLAY ИЗОБРАЖЕНИЙ ---
    if planar_mode:
        M_AtoB_cv2 = _get_cv2_affine_matrix(T_1to2, h, w)
        M_BtoA_cv2 = _get_cv2_affine_matrix(T_2to1, h, w)

        im_A_warped_to_B_np = cv2.warpAffine(im_A_np, M_AtoB_cv2, (w, h))
        overlay_A_on_B = cv2.addWeighted(im_B_np, 0.5, im_A_warped_to_B_np, 0.5, 0)

        im_B_warped_to_A_np = cv2.warpAffine(im_B_np, M_BtoA_cv2, (w, h))
        overlay_B_on_A = cv2.addWeighted(im_A_np, 0.5, im_B_warped_to_A_np, 0.5, 0)
    else:
        # Для непланарного случая GT overlay создается через рендеринг с использованием обратного варпа
        # A -> B (A warped onto B)
        overlay_A_on_B = _create_gt_overlay_3d(im_A_np, im_B_np, depth1, depth2, T_2to1, K2, K1, h, w)

        # B -> A (B warped onto A)
        overlay_B_on_A = _create_gt_overlay_3d(im_B_np, im_A_np, depth2, depth1, T_1to2, K1, K2, h, w)

    # --- 2.5 PRED OVERLAY ---
    overlay_B_on_A_pred = _get_pred_overlay(matches, certainty, valid_mask_fwd_np, im_A_np, im_B_np, h, w)

    # === 3. ВИЗУАЛИЗАЦИЯ ===
    # --- Основная фигура и GridSpec ---
    fig_total = plt.figure(figsize=(20, 12))
    # Основная сетка: 1 строка, 2 столбца. Левая часть для изображений, правая для аналитики.
    gs_main = fig_total.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.05)

    # Вложенная сетка для левой части (изображения и оверлеи)
    # 3 строки, 2 столбца с минимальным расстоянием между столбцами
    gs_left = gs_main[0, 0].subgridspec(3, 2, wspace=0.05, hspace=0.1)

    # Вложенная сетка для правой части (аналитика)
    # 3 строки, 2 столбца
    gs_right = gs_main[0, 1].subgridspec(3, 2, wspace=0.1, hspace=0.1)

    # --- Размещение графиков в левой сетке ---
    # Исходные изображения
    ax_A = fig_total.add_subplot(gs_left[0, 0])
    ax_A.imshow((im_A_vis.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    ax_A.set_title("Image A", fontsize=12)
    ax_A.axis("off")

    ax_B = fig_total.add_subplot(gs_left[0, 1])
    ax_B.imshow((im_B_vis.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    ax_B.set_title("Image B", fontsize=12)
    ax_B.axis("off")

    # GT Оверлеи
    ax_overlay_B_on_A = fig_total.add_subplot(gs_left[1, 0])
    ax_overlay_B_on_A.imshow(overlay_B_on_A)
    ax_overlay_B_on_A.set_title("Overlay: B onto A (GT)", fontsize=12)
    ax_overlay_B_on_A.axis("off")

    ax_overlay_A_on_B = fig_total.add_subplot(gs_left[1, 1])
    ax_overlay_A_on_B.imshow(overlay_A_on_B)
    ax_overlay_A_on_B.set_title("Overlay: A onto B (GT)", fontsize=12)
    ax_overlay_A_on_B.axis("off")

    # Pred Оверлей и Статистика
    ax_overlay_pred = fig_total.add_subplot(gs_left[2, 0])
    ax_overlay_pred.imshow(overlay_B_on_A_pred)
    ax_overlay_pred.set_title("Overlay: B onto A (Pred)", fontsize=12)
    ax_overlay_pred.axis("off")

    ax_stats = fig_total.add_subplot(gs_left[2, 1])
    stats_text = f"""Statistics (A -> B error):
Mean error: {mean_error:.2f}px
Median error: {median_error:.2f}px
<1px: {ratio_1*100:.1f}%
<3px: {ratio_3*100:.1f}%
<5px: {ratio_5*100:.1f}%
Valid points: {valid_mask_fwd_np.sum()}/{valid_mask_fwd_np.size}"""
    ax_stats.text(
        0.05, 0.5, stats_text, fontsize=12, va="center", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    )
    ax_stats.axis("off")

    # --- Размещение графиков в правой сетке ---
    # GT-квадранты
    ax_quad = fig_total.add_subplot(gs_right[0, :])
    _draw_quadrant_center_correspondences(ax_quad, im_A_np, im_B_np, T_1to2, T_2to1, h, w, planar_mode, depth1, depth2, K1, K2)

    # Предсказанные соответствия
    ax_matches = fig_total.add_subplot(gs_right[1, :])
    visualize_matches(
        ax_matches, im_A_vis, im_B_vis, matches, certainty, valid_mask_fwd_np, h, w, cmap, error_map_forward_np, norm
    )

    # Карты ошибок
    ax_err_fwd = fig_total.add_subplot(gs_right[2, 0])
    error_display_fwd = np.ma.masked_where(~valid_mask_fwd_np, error_map_forward_np)
    cmap.set_bad(color="gray", alpha=0.5)
    im_err = ax_err_fwd.imshow(error_display_fwd, cmap=cmap, norm=norm, interpolation="nearest")
    ax_err_fwd.set_title("Error Map (A -> B)", fontsize=12)
    ax_err_fwd.tick_params(
        axis="both", which="both", bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False
    )
    ax_err_fwd.axis("off")

    ax_err_bwd = fig_total.add_subplot(gs_right[2, 1])
    error_display_bwd = np.ma.masked_where(~valid_mask_bwd_np, error_map_backward_np)
    ax_err_bwd.imshow(error_display_bwd, cmap=cmap, norm=norm, interpolation="nearest")
    ax_err_bwd.set_title("Error Map (B -> A)", fontsize=12)
    ax_err_bwd.tick_params(
        axis="both", which="both", bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False
    )
    ax_err_bwd.axis("off")

    # Общий colorbar для правой части
    cbar = fig_total.colorbar(
        im_err, ax=fig_total.get_axes(), orientation="vertical", location="right", aspect=40, pad=0.05
    )
    cbar.set_label("Euclidean Error (pixels)", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    return [fig_total]


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
