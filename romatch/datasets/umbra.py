"""
UmbraScene – torch Dataset для обучения/валидации на парах изображений,
полученных из одной сцены Umbra (GeoTIFF).

ВЕРСИЯ С АУГМЕНТАЦИЯМИ для улучшенной генерализации на SAR данных.

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
from romatch.utils.utils import TupleCompose, TupleNormalize, TupleResize, TupleToTensorScaled
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as tvf
import json

def get_tuple_transform_ops(resize=None, normalize=True):
    ops = []
    if resize:
        ops.append(TupleResize(resize))
    ops.append(TupleToTensorScaled())
    if normalize:
        ops.append(
            TupleNormalize(mean=[0.221370] * 3, std=[0.137669] * 3) # 560
            # TupleNormalize(mean=[0.180998] * 3, std=[0.151903] * 3) # 1024
        )  # Umbra tiles mean/std
    return TupleCompose(ops)

class UmbraScene(Dataset):
    def __init__(self, 
                 scene_info, 
                 image_size, 
                 scene_name="custom_scene",
                 use_horizontal_flip_aug=False,
                 use_vertical_flip_aug=False,
                 shake_t=0,
                 ) -> None:
        """
        Args:
            scene_info: словарь с 'image_paths' и 'pairs'
            image_size: целевой размер изображения
            scene_name: имя сцены
            use_horizontal_flip_aug: использовать horizontal flip (рекомендуется)
            use_vertical_flip_aug: использовать vertical flip (опционально для SAR)
            shake_t: максимальный сдвиг в пикселях для random translation
        """
        self.image_paths = scene_info["image_paths"]
        self.original_pairs = scene_info["pairs"]
        self.scene_name = scene_name
        self.image_size = image_size
        
        # Augmentation параметры
        self.use_horizontal_flip_aug = use_horizontal_flip_aug
        self.use_vertical_flip_aug = use_vertical_flip_aug
        self.shake_t = shake_t
        
        self.im_transform_ops = get_tuple_transform_ops(
            resize=(image_size, image_size),
            normalize=True,
        )
        
        # Валидация и фильтрация пар при инициализации
        self.pairs = self._validate_and_filter_pairs()

    def _validate_and_filter_pairs(self):
        """
        Проверяет все пары на валидность геометрического преобразования.
        Удаляет пары, приводящие к сингулярной матрице гомографии.
        """
        print(f"[{self.scene_name}] Запуск валидации для {len(self.original_pairs)} пар...")
        valid_pairs = []
        invalid_count = 0
        for idx1, idx2 in self.original_pairs:
            try:
                # Используем ту же логику загрузки, что и в __getitem__, чтобы размеры совпадали
                im1_pil, affine1, _ = self.load_im_and_georef(self.image_paths[idx1])
                im2_pil, affine2, _ = self.load_im_and_georef(self.image_paths[idx2])

                # Вычисляем гомографию, используя актуальный размер изображения после возможного кропа
                H_np = self.compute_homography(affine1, affine2, im1_pil.size, self.image_size)
                
                # Преобразуем в torch.tensor float32, чтобы точно имитировать условия сбоя
                H_torch = torch.from_numpy(H_np).float()
                R_torch = H_torch[:2, :2]
                
                # Проверяем определитель, используя torch
                determinant = torch.det(R_torch)
                if abs(determinant) < 1e-8:
                    # Эта пара приведет к сингулярной матрице
                    invalid_count += 1
                    continue # Пропускаем эту пару
                    
                valid_pairs.append((idx1, idx2))
            except Exception as e:
                # Также отлавливаем другие возможные ошибки при чтении файлов
                print(f"[{self.scene_name}] Ошибка при обработке пары ({idx1}, {idx2}): {e}. Пара будет пропущена.")
                invalid_count += 1

        if invalid_count > 0:
            print(f"[{self.scene_name}] Валидация завершена. Найдено и удалено {invalid_count} проблемных пар.")
        else:
            print(f"[{self.scene_name}] Валидация завершена. Все пары корректны.")
            
        return valid_pairs

    def __len__(self):
        return len(self.pairs)

    def load_im_and_georef(self, path):
        """Загружает изображение и его геопривязку"""
        with rasterio.open(path) as src:
            # Читаем только первый канал
            image_array = src.read(1).astype(np.uint8)
            # Получаем аффинную матрицу
            affine = src.transform
            # Конвертируем в PIL Image для трансформаций
            img = Image.fromarray(image_array).convert("RGB")
            original_size = (src.width, src.height)
            
            # Center crop если изображение 1024 и target 1008
            # Это избегает downscaling при сохранении кратности 112 (14*8)
            if self.image_size % 112 == 0 and original_size == (1024, 1024) and self.image_size == 1008:
                # Center crop 1024 -> 1008
                margin = (1024 - 1008) // 2  # 8 пикселей с каждой стороны
                img = tvf.crop(img, margin, margin, 1008, 1008)
                # Корректируем affine для cropped области
                from rasterio.transform import Affine
                affine = affine * Affine.translation(margin, margin)
                original_size = (1008, 1008)
            
            return img, affine, original_size

    def affine_to_matrix(self, affine):
        """Конвертирует rasterio.Affine в numpy 3x3 матрицу"""
        return np.array([
            [affine.a, affine.b, affine.c],
            [affine.d, affine.e, affine.f],
            [0, 0, 1]
        ], dtype=np.float32)

    def compute_homography(self, affine1, affine2, orig_size, target_size):
        """
        Вычисляет гомографию между двумя изображениями с учетом resize.
        
        Для ортофотоснимков трансформация:
        pixel_1 -> world -> pixel_2
        
        С учетом resize:
        normalized_1 -> pixel_1_orig -> world -> pixel_2_orig -> normalized_2
        """
        # Матрицы pixel -> world
        A1 = self.affine_to_matrix(affine1)
        A2 = self.affine_to_matrix(affine2)
        
        orig_w, orig_h = orig_size
        
        # Матрица: normalized [-1,1] -> pixel_orig [0, orig_size]
        N_to_P1 = np.array([
            [orig_w/2, 0, orig_w/2],
            [0, orig_h/2, orig_h/2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Матрица: pixel_orig [0, orig_size] -> normalized [-1,1]
        P2_to_N = np.array([
            [2/orig_w, 0, -1],
            [0, 2/orig_h, -1],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Полная гомография: normalized_1 -> world -> normalized_2
        # H = P2_to_N @ inv(A2) @ A1 @ N_to_P1
        H = P2_to_N @ np.linalg.inv(A2) @ A1 @ N_to_P1
        
        return H

    def homography_to_K_and_T(self, H, image_size):
        """
        Decomposes homography into K and T for RoMa compatibility.
        For a planar scene, T_1to2 is derived from the homography H
        that maps normalized coordinates from image 1 to image 2.
        """
        # For orthographic images, we use a simple K
        K = np.eye(3, dtype=np.float32)
        
        # The transformation T should directly represent the homography effect
        # on normalized coordinates.
        T = np.eye(4, dtype=np.float32)
        
        # The 2x2 part of H is the rotation and scale component
        T[0:2, 0:2] = H[0:2, 0:2]
        
        # The translation part of H
        T[0, 3] = H[0, 2]
        T[1, 3] = H[1, 2]
        
        return K, T

    def horizontal_flip(self, im_A, im_B, K1, K2, T_1to2):
        """Horizontal flip augmentation with correction for T_1to2"""
        im_A = im_A.flip(-1)
        im_B = im_B.flip(-1)
        
        # Adjust T_1to2 for horizontal flip
        # x' = -x => in normalized coords, this flips the sign of x component
        flip_mat = torch.tensor([[-1, 0], [0, 1]], dtype=T_1to2.dtype, device=T_1to2.device)
        T_1to2[:2, :2] = flip_mat @ T_1to2[:2, :2] @ flip_mat
        T_1to2[0, 3] = -T_1to2[0, 3] # Translation also flips
        
        return im_A, im_B, K1, K2, T_1to2
    
    def vertical_flip(self, im_A, im_B, K1, K2):
        """Vertical flip augmentation с корректировкой intrinsics"""
        im_A = im_A.flip(-2)
        im_B = im_B.flip(-2)
        
        flip_mat = torch.tensor([
            [1, 0, 0],
            [0, -1, self.image_size],
            [0, 0, 1.]
        ], dtype=K1.dtype, device=K1.device)
        
        K1 = flip_mat @ K1
        K2 = flip_mat @ K2
        
        return im_A, im_B, K1, K2
    
    def rand_shake(self, *things):
        """Random translation augmentation (применяется к тензорам ПОСЛЕ нормализации)"""
        t = np.random.choice(range(-self.shake_t, self.shake_t + 1), size=2)
        return [
            tvf.affine(thing, angle=0.0, translate=list(t), scale=1.0, shear=[0.0, 0.0])
            for thing in things
        ], t

    def __getitem__(self, index):
        idx1, idx2 = self.pairs[index]

        path1 = self.image_paths[idx1]
        path2 = self.image_paths[idx2]

        im1_pil, affine1, orig_size1 = self.load_im_and_georef(path1)
        im2_pil, affine2, orig_size2 = self.load_im_and_georef(path2)
        
        # --- Вычисление геометрических данных из геопривязки ---
        # Теперь гомография вычисляется здесь, используя актуальные размеры
        H = self.compute_homography(affine1, affine2, im1_pil.size, self.image_size)
        
        # Разлагаем гомографию на K и T
        K, T_1to2 = self.homography_to_K_and_T(H, self.image_size)

        # Применяем стандартные трансформации (resize + normalize)
        im1, im2 = self.im_transform_ops((im1_pil, im2_pil))

        # Для Umbra SAR обе камеры имеют одинаковые intrinsics (ортографическая проекция)
        K1 = torch.from_numpy(K).float()
        K2 = torch.from_numpy(K).float()
        T_1to2 = torch.from_numpy(T_1to2).float()
        
        # Создание карты глубины
        depth_value = 1.0
        depth1 = torch.ones(1, self.image_size, self.image_size) * depth_value
        depth2 = torch.ones(1, self.image_size, self.image_size) * depth_value
        
        # Random shake
        if self.shake_t > 0:
            [im1, im2, depth1, depth2], t = self.rand_shake(im1, im2, depth1, depth2)
            
            t_norm = t / self.image_size * 2
            T_1to2[0, 3] -= t_norm[0]
            T_1to2[1, 3] -= t_norm[1]

        # Horizontal flip
        if self.use_horizontal_flip_aug and np.random.rand() > 0.5:
            im1, im2, K1, K2, T_1to2 = self.horizontal_flip(im1, im2, K1, K2, T_1to2)
        
        # Vertical flip
        if self.use_vertical_flip_aug and np.random.rand() > 0.5:
            im1, im2, K1, K2 = self.vertical_flip(im1, im2, K1, K2)

        return {
            "im_A": im1,
            "im_B": im2,
            "im_A_depth": depth1.squeeze(0),
            "im_B_depth": depth2.squeeze(0),
            "K1": K1,
            "K2": K2,
            "T_1to2": T_1to2,
            "im_A_path": path1,
            "im_B_path": path2,
        }


class UmbraBuilder:
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
        scene = UmbraScene(scene_info, **kwargs)
        return [scene]

    def weight_scenes(self, concat_dataset, alpha=0.75):
        # Простая реализация взвешивания, если у нас несколько сцен
        ns = [len(d) for d in concat_dataset.datasets]
        ws = torch.cat([torch.ones(n) / n**alpha for n in ns])
        return ws

