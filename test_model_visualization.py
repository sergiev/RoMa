"""
Тест визуализации с реальной моделью
"""
import sys
sys.path.insert(0, '/home/sema/radar/RoMa')

import torch
import numpy as np
from romatch.datasets.umbra import UmbraScene
from romatch.utils import warp_kpts
import json
from experiments.train_roma_umbra import get_planar_model

print("="*60)
print("ТЕСТ ВИЗУАЛИЗАЦИИ С РЕАЛЬНОЙ МОДЕЛЬЮ")
print("="*60)

# Загружаем датасет
with open('/home/sema/radar/datasets/umbra_tiles/560/pairs_val.json', 'r') as f:
    scene_info = json.load(f)

dataset = UmbraScene(scene_info, image_size=560)
sample = dataset[0]

im_A = sample['im_A']
im_B = sample['im_B']
depth1 = sample['im_A_depth']
depth2 = sample['im_B_depth']
K1 = sample['K1']
K2 = sample['K2']
T_1to2 = sample['T_1to2']

print(f"\nИзображения загружены:")
print(f"  im_A shape: {im_A.shape}")
print(f"  im_B shape: {im_B.shape}")

# Загружаем модель
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights_path = "/home/sema/radar/RoMa/workspace/20251012_medium_native_tunedloss/train_roma_umbra_best.pth"
weights = torch.load(weights_path, map_location="cuda")
if weights.get("model"):
    weights = weights["model"]
resolution="medium"
matcher = get_planar_model(pretrained_backbone=True, resolution=resolution, attenuate_cert=False)

matcher.train(False)
print(f"\nУстройство: {device}")
matcher.load_state_dict(weights)
matcher.symmetric = False

print("Модель загружена")

# Запускаем matching
im_A_batch = im_A.unsqueeze(0).to(device)
im_B_batch = im_B.unsqueeze(0).to(device)

with torch.no_grad():
    matches, certainty = matcher.match(im_A_batch, im_B_batch, batched=True)

matches = matches[0].cpu()
certainty = certainty[0].cpu()

h, w = matches.shape[:2]

print(f"\nРезультаты matching:")
print(f"  matches shape: {matches.shape}")
print(f"  certainty shape: {certainty.shape}")
print(f"  h={h}, w={w}")

# Проверяем формат matches
print(f"\n" + "="*60)
print("ПРОВЕРКА ФОРМАТА MATCHES")
print("="*60)

# Проверим несколько точек
test_points = [(100, 100), (200, 200), (300, 300)]
for y, x in test_points:
    if y < h and x < w:
        m = matches[y, x]
        print(f"\nPixel ({x}, {y}):")  # (x, y) - как мы обычно говорим
        print(f"  matches[{y}, {x}] = {m.tolist()}")
        print(f"  x1_norm, y1_norm = {m[0]:.4f}, {m[1]:.4f}")
        print(f"  x2_norm, y2_norm = {m[2]:.4f}, {m[3]:.4f}")
        
        # Конвертируем в пиксели
        x1_px = w * (m[0].item() + 1) / 2
        y1_px = h * (m[1].item() + 1) / 2
        x2_px = w * (m[2].item() + 1) / 2
        y2_px = h * (m[3].item() + 1) / 2
        
        print(f"  (x1, y1)_px = ({x1_px:.1f}, {y1_px:.1f}) [должно быть ~({x}, {y})]")
        print(f"  (x2, y2)_px = ({x2_px:.1f}, {y2_px:.1f})")
        
        # Проверка: x1_px должно быть примерно равно x, y1_px примерно равно y
        if abs(x1_px - x) > 2 or abs(y1_px - y) > 2:
            print(f"  ⚠️ ВНИМАНИЕ: Большое расхождение!")

# Вычисляем ground truth
print(f"\n" + "="*60)
print("ВЫЧИСЛЕНИЕ GROUND TRUTH")
print("="*60)

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

# Конвертируем в пиксели
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

# Вычисляем ошибку
error = (x2_pred_px - x2_gt_px).norm(dim=1)
error_map = error.reshape(h, w)

print(f"Ground truth вычислен")
print(f"  error_map shape: {error_map.shape}")
print(f"  mean error: {error.mean().item():.2f}px")
print(f"  median error: {error.median().item():.2f}px")

# Проверяем ошибки для тестовых точек
print(f"\n" + "="*60)
print("ОШИБКИ ДЛЯ ТЕСТОВЫХ ТОЧЕК")
print("="*60)

for y, x in test_points:
    if y < h and x < w:
        idx = y * w + x
        err = error[idx].item()
        gt = x2_gt_px[idx]
        pred = x2_pred_px[idx]
        
        print(f"\nPixel ({x}, {y}):")
        print(f"  Ground truth:  ({gt[0]:.1f}, {gt[1]:.1f})px")
        print(f"  Predicted:     ({pred[0]:.1f}, {pred[1]:.1f})px")
        print(f"  Error:         {err:.2f}px")
        print(f"  Difference:    dx={pred[0]-gt[0]:.1f}, dy={pred[1]-gt[1]:.1f}")

# Проверяем распределение ошибок
print(f"\n" + "="*60)
print("РАСПРЕДЕЛЕНИЕ ОШИБОК")
print("="*60)

valid_mask = mask_gt[0].reshape(h, w).bool()
valid_errors = error_map[valid_mask]

if valid_errors.numel() > 0:
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print(f"Перцентили ошибок (valid points):")
    for p in percentiles:
        val = torch.quantile(valid_errors, p/100.0).item()
        print(f"  {p:2d}%: {val:6.2f}px")
    
    ratio_1 = (valid_errors < 1.0).float().mean().item()
    ratio_3 = (valid_errors < 3.0).float().mean().item()
    ratio_5 = (valid_errors < 5.0).float().mean().item()
    
    print(f"\nДоля точек с ошибкой:")
    print(f"  <1px: {ratio_1*100:.1f}%")
    print(f"  <3px: {ratio_3*100:.1f}%")
    print(f"  <5px: {ratio_5*100:.1f}%")

# Проверяем certainty
print(f"\n" + "="*60)
print("CERTAINTY")
print("="*60)

certainty_np = certainty.detach().cpu().numpy()
print(f"Certainty stats:")
print(f"  min: {certainty_np.min():.4f}")
print(f"  max: {certainty_np.max():.4f}")
print(f"  mean: {certainty_np.mean():.4f}")
print(f"  median: {np.median(certainty_np):.4f}")

# Проверяем корреляцию между certainty и ошибкой
high_cert_mask = certainty_np > np.percentile(certainty_np, 90)
high_cert_errors = error_map.numpy()[high_cert_mask]
all_errors = error_map.numpy()[valid_mask.numpy()]

if len(high_cert_errors) > 0 and len(all_errors) > 0:
    print(f"\nОшибки для точек с высоким certainty (>90%):")
    print(f"  mean: {high_cert_errors.mean():.2f}px vs all: {all_errors.mean():.2f}px")
    print(f"  median: {np.median(high_cert_errors):.2f}px vs all: {np.median(all_errors):.2f}px")

print("\n" + "="*60)
print("ПРОВЕРКА ВИЗУАЛИЗАЦИИ")
print("="*60)

# Выберем несколько точек с высоким certainty
y_coords, x_coords = np.where(high_cert_mask & valid_mask.numpy())
if len(y_coords) > 10:
    indices = np.random.choice(len(y_coords), 10, replace=False)
    y_coords = y_coords[indices]
    x_coords = x_coords[indices]

print(f"Выбрано {len(y_coords)} точек с высоким certainty:")
for i, (y, x) in enumerate(zip(y_coords, x_coords)):
    idx = y * w + x
    err = error[idx].item()
    cert = certainty_np[y, x]
    gt = x2_gt_px[idx]
    pred = x2_pred_px[idx]
    
    # Координаты для визуализации (как в visualize_pipeline.py)
    kpt0 = (x, y)  # (x, y) в первом изображении
    
    # Из matches
    matches_np = matches.numpy()
    kpt1_norm = matches_np[y, x, 2:]  # нормализованные координаты во втором изображении
    kpt1 = (w * (kpt1_norm[0] + 1) / 2, h * (kpt1_norm[1] + 1) / 2)
    
    print(f"\n  Point {i}: pixel=({x}, {y})")
    print(f"    certainty: {cert:.3f}")
    print(f"    error: {err:.2f}px")
    print(f"    kpt0: ({kpt0[0]}, {kpt0[1]})")
    print(f"    kpt1_pred: ({kpt1[0]:.1f}, {kpt1[1]:.1f})")
    print(f"    kpt1_gt: ({gt[0]:.1f}, {gt[1]:.1f})")
    print(f"    difference: ({kpt1[0]-gt[0]:.1f}, {kpt1[1]-gt[1]:.1f})")

print("\n✓ Тест завершен!")

