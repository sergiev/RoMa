"""
Тест для проверки бага с размерами изображения
"""
import sys
sys.path.insert(0, '/home/sema/radar/RoMa')

import torch
from romatch.datasets.umbra import UmbraScene
import json

# Загружаем датасет
with open('/home/sema/radar/datasets/umbra_tiles/560/pairs_val.json', 'r') as f:
    scene_info = json.load(f)

dataset = UmbraScene(scene_info, image_size=560)
sample = dataset[0]

im_A = sample['im_A']
im_B = sample['im_B']
depth1 = sample['im_A_depth']

print("="*60)
print("ПРОВЕРКА РАЗМЕРОВ")
print("="*60)
print(f"im_A shape: {im_A.shape}")  # (C, H, W)
print(f"im_B shape: {im_B.shape}")
print(f"depth1 shape: {depth1.shape}")  # (H, W)

# Имитируем matches от модели
# Предположим модель возвращает matches в resolution (560, 560)
matches_h, matches_w = 560, 560
print(f"\nmatches shape: ({matches_h}, {matches_w}, 4)")

# Какие h, w используются в visualize_matches?
h, w = matches_h, matches_w  # Из matches.shape[:2]
print(f"\nВ visualize_matches:")
print(f"  h, w = matches.shape[:2] = ({h}, {w})")

# Какие h, w ДОЛЖНЫ использоваться для depth?
actual_h, actual_w = depth1.shape
print(f"\nФактический размер изображения (из depth):")
print(f"  actual_h, actual_w = depth.shape = ({actual_h}, {actual_w})")

if h != actual_h or w != actual_w:
    print(f"\n⚠️⚠️⚠️ БАГ НАЙДЕН! ⚠️⚠️⚠️")
    print(f"matches используют ({h}, {w}), но изображение ({actual_h}, {actual_w})")
else:
    print(f"\n✓ Размеры совпадают")

# Проверка влияния на вычисление
print("\n" + "="*60)
print("ВЛИЯНИЕ НА ВЫЧИСЛЕНИЕ КООРДИНАТ")
print("="*60)

# Тестовая точка в normalized coords
x_norm, y_norm = 0.5, -0.3

# Неправильная конверсия (используя matches size)
x_px_wrong = w * (x_norm + 1) / 2
y_px_wrong = h * (y_norm + 1) / 2

# Правильная конверсия (используя image size)
x_px_correct = actual_w * (x_norm + 1) / 2
y_px_correct = actual_h * (y_norm + 1) / 2

print(f"Normalized coords: ({x_norm}, {y_norm})")
print(f"Wrong conversion (using matches size): ({x_px_wrong:.1f}, {y_px_wrong:.1f})")
print(f"Correct conversion (using image size): ({x_px_correct:.1f}, {y_px_correct:.1f})")
print(f"Difference: ({abs(x_px_wrong - x_px_correct):.1f}, {abs(y_px_wrong - y_px_correct):.1f}) pixels")

if abs(x_px_wrong - x_px_correct) > 0.01 or abs(y_px_wrong - y_px_correct) > 0.01:
    print("\n⚠️ ПРОБЛЕМА: Ошибка в конверсии координат!")
else:
    print("\n✓ Конверсия правильная")

print("\n" + "="*60)
print("СИМУЛЯЦИЯ ВЛИЯНИЯ НА ERROR CALCULATION")
print("="*60)

# Симулируем ситуацию с разными размерами
# Допустим модель обучалась на 1008x1008, а теперь работает на 560x560
sim_model_res = 1008
sim_image_res = 560

print(f"Симуляция: модель обучена на {sim_model_res}x{sim_model_res}")
print(f"           но тестируется на {sim_image_res}x{sim_image_res}")

# Тестовая точка
x_norm_test = 0.0  # center

# Если используется model resolution для конверсии
x_px_model = sim_model_res * (x_norm_test + 1) / 2
# Если используется image resolution
x_px_image = sim_image_res * (x_norm_test + 1) / 2

print(f"\nТочка в центре (x_norm={x_norm_test}):")
print(f"  Конверсия с model res: {x_px_model:.1f}px")
print(f"  Конверсия с image res: {x_px_image:.1f}px")
print(f"  Ошибка: {abs(x_px_model - x_px_image):.1f}px")

# Для угла
x_norm_corner = 0.5
x_px_model_corner = sim_model_res * (x_norm_corner + 1) / 2
x_px_image_corner = sim_image_res * (x_norm_corner + 1) / 2

print(f"\nТочка в углу (x_norm={x_norm_corner}):")
print(f"  Конверсия с model res: {x_px_model_corner:.1f}px")
print(f"  Конверсия с image res: {x_px_image_corner:.1f}px")
print(f"  Ошибка: {abs(x_px_model_corner - x_px_image_corner):.1f}px")

scale_factor = sim_model_res / sim_image_res
print(f"\nМасштабный фактор: {scale_factor:.2f}x")
print(f"Это означает, что все ошибки будут в {scale_factor:.2f}x раз больше!")

