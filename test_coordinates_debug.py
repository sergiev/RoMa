"""
Тест для проверки порядка координат в визуализации
"""
import sys
sys.path.insert(0, '/home/sema/radar/RoMa')

import torch
import numpy as np
from romatch.datasets.umbra import UmbraScene
from romatch.utils import warp_kpts
import json

# Загружаем датасет
with open('/home/sema/radar/datasets/umbra_tiles/560/pairs_val.json', 'r') as f:
    scene_info = json.load(f)

dataset = UmbraScene(scene_info, image_size=560)
sample = dataset[0]

depth1 = sample['im_A_depth']
depth2 = sample['im_B_depth']
K1 = sample['K1']
K2 = sample['K2']
T_1to2 = sample['T_1to2']

h, w = depth1.shape

print("="*60)
print("ПРОВЕРКА ПОРЯДКА КООРДИНАТ")
print("="*60)
print(f"\nРазмер изображения: {w} x {h}")
print(f"\nK1:\n{K1.numpy()}")
print(f"\nT_1to2 (rotation + translation):\n{T_1to2[:3, :].numpy()}")

# Создаем несколько тестовых точек в нормализованных координатах
test_points_norm = torch.tensor([
    [0.0, 0.0],    # центр
    [-0.5, -0.5],  # верхний левый квадрант
    [0.5, -0.5],   # верхний правый квадрант
    [-0.5, 0.5],   # нижний левый квадрант
    [0.5, 0.5],    # нижний правый квадрант
], dtype=torch.float32).unsqueeze(0)  # (1, 5, 2)

print("\n" + "="*60)
print("ТЕСТОВЫЕ ТОЧКИ (normalized coords)")
print("="*60)
for i, pt in enumerate(test_points_norm[0]):
    print(f"Point {i}: norm=({pt[0]:6.2f}, {pt[1]:6.2f})")

# Конвертируем в пиксельные координаты (как в warp_kpts)
test_points_px = torch.stack(
    (w * (test_points_norm[..., 0] + 1) / 2, h * (test_points_norm[..., 1] + 1) / 2), dim=-1
)

print("\n" + "="*60)
print("КОНВЕРСИЯ В ПИКСЕЛИ (как в warp_kpts строка 404)")
print("="*60)
print(f"Формула: x_px = w * (x_norm + 1) / 2, y_px = h * (y_norm + 1) / 2")
for i, (pt_norm, pt_px) in enumerate(zip(test_points_norm[0], test_points_px[0])):
    print(f"Point {i}: norm=({pt_norm[0]:6.2f}, {pt_norm[1]:6.2f}) -> px=({pt_px[0]:6.1f}, {pt_px[1]:6.1f})")
    # Проверка
    expected_x = w * (pt_norm[0].item() + 1) / 2
    expected_y = h * (pt_norm[1].item() + 1) / 2
    assert abs(pt_px[0].item() - expected_x) < 0.01, f"X coord mismatch!"
    assert abs(pt_px[1].item() - expected_y) < 0.01, f"Y coord mismatch!"

# Теперь проверим warp_kpts
print("\n" + "="*60)
print("WARP_KPTS TEST")
print("="*60)

mask_gt, warped_points_norm = warp_kpts(
    test_points_norm.double(),
    depth1[None].double(),
    depth2[None].double(),
    T_1to2[None, :3, :].double(),
    K1[None].double(),
    K2[None].double(),
)

# Конвертируем warped точки в пиксели
warped_points_px = torch.stack(
    (w * (warped_points_norm[..., 0] + 1) / 2, h * (warped_points_norm[..., 1] + 1) / 2), dim=-1
)

print("Input -> Warped (normalized coords):")
for i in range(len(test_points_norm[0])):
    pt_in = test_points_norm[0, i]
    pt_out = warped_points_norm[0, i]
    valid = mask_gt[0, i].item()
    print(f"Point {i}: ({pt_in[0]:6.2f}, {pt_in[1]:6.2f}) -> ({pt_out[0]:6.2f}, {pt_out[1]:6.2f}) [valid={valid:.0f}]")

print("\nInput -> Warped (pixel coords):")
for i in range(len(test_points_px[0])):
    pt_in_px = test_points_px[0, i]
    pt_out_px = warped_points_px[0, i]
    diff = pt_out_px - pt_in_px
    print(f"Point {i}: ({pt_in_px[0]:6.1f}, {pt_in_px[1]:6.1f}) -> ({pt_out_px[0]:6.1f}, {pt_out_px[1]:6.1f}) | diff=({diff[0]:6.1f}, {diff[1]:6.1f})")

# Проверка визуализации: как вычисляется ошибка
print("\n" + "="*60)
print("ERROR CALCULATION (как в visualize_matches)")
print("="*60)

# Создаем fake predicted координаты
pred_points_norm = test_points_norm + torch.tensor([0.05, -0.03]).unsqueeze(0)  # небольшое смещение

pred_points_px = torch.stack(
    (w * (pred_points_norm[..., 0] + 1) / 2, h * (pred_points_norm[..., 1] + 1) / 2), dim=-1
)

# Вычисляем ошибку как в visualize_matches (строка 100)
error = (pred_points_px - warped_points_px).norm(dim=-1)[0]

print("Ground truth vs Predicted (pixel coords):")
for i in range(len(warped_points_px[0])):
    gt_px = warped_points_px[0, i]
    pr_px = pred_points_px[0, i]
    err = error[i]
    print(f"Point {i}: GT=({gt_px[0]:6.1f}, {gt_px[1]:6.1f}) | Pred=({pr_px[0]:6.1f}, {pr_px[1]:6.1f}) | Error={err:5.2f}px")

print("\n" + "="*60)
print("ПРОВЕРКА КООРДИНАТНОГО ФОРМАТА")
print("="*60)

# Проверяем, правильно ли интерпретируются координаты
# Создадим точку явно в "левом верхнем углу" в pixel coords
top_left_px = torch.tensor([[100.0, 50.0]], dtype=torch.float32)  # x=100, y=50
# Конвертируем в normalized
top_left_norm = torch.stack(
    (2 * top_left_px[..., 0] / w - 1, 2 * top_left_px[..., 1] / h - 1), dim=-1
)
# И обратно
top_left_px_back = torch.stack(
    (w * (top_left_norm[..., 0] + 1) / 2, h * (top_left_norm[..., 1] + 1) / 2), dim=-1
)

print(f"Test: (100, 50) px -> norm -> px")
print(f"  Original px: ({top_left_px[0, 0]:.1f}, {top_left_px[0, 1]:.1f})")
print(f"  Normalized:  ({top_left_norm[0, 0]:.4f}, {top_left_norm[0, 1]:.4f})")
print(f"  Back to px:  ({top_left_px_back[0, 0]:.1f}, {top_left_px_back[0, 1]:.1f})")

assert torch.allclose(top_left_px, top_left_px_back, atol=0.01), "Round-trip conversion failed!"
print("  ✓ Round-trip conversion OK")

print("\n" + "="*60)
print("ИНТЕРПРЕТАЦИЯ")
print("="*60)
print("""
В коде используется формат (x, y) где:
- x это горизонтальная координата (width dimension)
- y это вертикальная координата (height dimension)

При преобразовании normalized -> pixel:
- x_px = w * (x_norm + 1) / 2
- y_px = h * (y_norm + 1) / 2

Это ПРАВИЛЬНО если:
- coords[..., 0] это x (horizontal)
- coords[..., 1] это y (vertical)

Проверим matches из модели...
""")

print("\n" + "="*60)
print("ПРОВЕРКА MATCHES ИЗ МОДЕЛИ")
print("="*60)

# Создаем fake matches как их возвращает модель
# matches shape: (H, W, 4) где [:, :, :2] это (x1, y1) и [:, :, 2:] это (x2, y2)
fake_matches = torch.zeros(h, w, 4)

# Заполняем identity mapping (каждая точка мапится на себя)
for y in range(h):
    for x in range(w):
        # normalized coords для pixel (x, y)
        x_norm = 2 * x / w - 1 + 1/w  # center of pixel
        y_norm = 2 * y / h - 1 + 1/h
        fake_matches[y, x, 0] = x_norm  # x1
        fake_matches[y, x, 1] = y_norm  # y1
        fake_matches[y, x, 2] = x_norm  # x2 (identity)
        fake_matches[y, x, 3] = y_norm  # y2 (identity)

# Проверим несколько точек
test_pixels = [(100, 50), (200, 150), (400, 300)]
print("Identity mapping test (каждая точка мапится на себя):")
for px_x, px_y in test_pixels:
    if px_y < h and px_x < w:
        match = fake_matches[px_y, px_x]  # ВНИМАНИЕ: индексация [y, x]!
        print(f"  Pixel ({px_x}, {px_y}): matches={match.tolist()}")
        # Проверим, что x1 и x2 примерно равны
        x1_px_back = w * (match[0].item() + 1) / 2
        y1_px_back = h * (match[1].item() + 1) / 2
        x2_px_back = w * (match[2].item() + 1) / 2
        y2_px_back = h * (match[3].item() + 1) / 2
        print(f"    -> (x1,y1)_px = ({x1_px_back:.1f}, {y1_px_back:.1f})")
        print(f"    -> (x2,y2)_px = ({x2_px_back:.1f}, {y2_px_back:.1f})")

print("\n✓ Тест завершен!")

