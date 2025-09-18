"""
CustomScene – torch Dataset для обучения/валидации на парах изображений,
полученных из одной сцены Umbra (GeoTIFF).

Каждый элемент возвращает словарь, совместимый с моделью RoMa:
    im_A, im_B          – тензоры изображений (C, H, W), uint8 -> float32, нормализованы
    im_A_depth, im_B_depth - тензоры карт глубины (H, W)
    K1, K2              – матрицы интринсиков (3, 3)
    T_1to2              – относительная поза от 1 ко 2 (4, 4)
"""
import os
import rasterio
from PIL import Image
import numpy as np
from romatch.utils import get_tuple_transform_ops
import torch
from torch.utils.data import Dataset, ConcatDataset
import json


class CustomScene(Dataset):
    def __init__(self, scene_info, image_size=640, scene_name="custom_scene") -> None:
        self.image_paths = scene_info["image_paths"]
        self.pairs = scene_info["pairs"]
        self.scene_name = scene_name
        self.im_transform_ops = get_tuple_transform_ops(
            resize=(image_size, image_size),
            normalize=True,
            num_channels=1, # <--- Указываем, что у нас одноканальные изображения
        )
        self.image_size = image_size
        
    def __len__(self):
        return len(self.pairs)

    def load_im_and_georef(self, path):
        # Используем rasterio для открытия GeoTIFF
        with rasterio.open(path) as src:
            # Читаем только первый канал
            image_array = src.read(1).astype(np.uint8)
            # Получаем аффинную матрицу
            affine = src.transform
            # Конвертируем в PIL Image для трансформаций (создастся grayscale L-mode image)
            img = Image.fromarray(image_array)
            return img, affine

    def __getitem__(self, index):
        idx1, idx2 = self.pairs[index]

        path1 = self.image_paths[idx1]
        path2 = self.image_paths[idx2]

        im1_pil, affine1 = self.load_im_and_georef(path1)
        im2_pil, affine2 = self.load_im_and_georef(path2)

        # Применяем стандартные трансформации (изменение размера, нормализация)
        im1, im2 = self.im_transform_ops((im1_pil, im2_pil))

        # --- Вычисление геометрических данных из геопривязки ---

        # 1. Создание матриц Intrinsics (K1, K2) для ортографической камеры
        # Для ортографической проекции K - это матрица масштабирования и сдвига.
        # sx = affine1.a, sy = -affine1.e (минус, т.к. y-ось пикселей идет вниз)
        # cx, cy - центр изображения
        K1 = torch.tensor(
            [
                [affine1.a, 0, self.image_size / 2],
                [0, -affine1.e, self.image_size / 2],
                [0, 0, 1],
            ],
            dtype=torch.float32,
        )
        K2 = torch.tensor(
            [
                [affine2.a, 0, self.image_size / 2],
                [0, -affine2.e, self.image_size / 2],
                [0, 0, 1],
            ],
            dtype=torch.float32,
        )

        # 2. Вычисление относительной позы (T_1to2)
        # T_1to2 преобразует 3D точку из системы координат камеры 1 в систему камеры 2.
        # Для ортофотоснимков, система координат камеры - это мировая система координат (с точностью до сдвига).
        # Вращение R - единичная матрица. Сдвиг t определяется разницей в world-координатах.
        T_1_to_world = torch.eye(4, dtype=torch.float32)
        T_1_to_world[0, 3] = affine1.c  # tx - Easting
        T_1_to_world[1, 3] = affine1.f  # ty - Northing

        T_2_to_world = torch.eye(4, dtype=torch.float32)
        T_2_to_world[0, 3] = affine2.c  # tx
        T_2_to_world[1, 3] = affine2.f  # ty

        # T_1_to_2 = inv(T_2_to_world) * T_1_to_world
        T_world_to_2 = torch.inverse(T_2_to_world)
        T_1to2 = torch.matmul(T_world_to_2, T_1_to_world)

        # 3. Создание карты глубины
        # Для ортографической проекции глубина постоянна. Используем 1.0.
        depth = torch.ones(self.image_size, self.image_size, dtype=torch.float32)

        return {
            "im_A": im1,
            "im_B": im2,
            "im_A_depth": depth,
            "im_B_depth": depth.clone(),
            "K1": K1,
            "K2": K2,
            "T_1to2": T_1to2,
            "im_A_path": path1,
            "im_B_path": path2,
        }


class CustomBuilder:
    def __init__(self, data_root, **kwargs) -> None:
        self.data_root = data_root

    def build_scenes(self, split="train", **kwargs):
        # Мы ожидаем, что data_root - это прямой путь к json файлу
        scene_info_path = self.data_root
        if not os.path.exists(scene_info_path):
            raise FileNotFoundError(f"Файл с информацией о сцене не найден: {scene_info_path}")
            
        with open(scene_info_path, "r") as f:
            scene_info = json.load(f)
        
        # RoMa ожидает список сцен, даже если сцена одна
        scene = CustomScene(scene_info, **kwargs)
        return [scene]

    def weight_scenes(self, concat_dataset, alpha=0.75):
        # Простая реализация взвешивания, если у нас несколько сцен
        ns = [len(d) for d in concat_dataset.datasets]
        ws = torch.cat([torch.ones(n) / n**alpha for n in ns])
        return ws
