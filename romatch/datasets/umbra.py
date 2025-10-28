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
                 planar_mode,
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
            planar_mode: если True, создает упрощенную геометрию для planar warp,
                        если False, создает полную 3D геометрию для depth-based warp
        """
        self.image_paths = scene_info["image_paths"]
        self.original_pairs = scene_info["pairs"]
        self.scene_name = scene_name
        self.image_size = image_size
        self.planar_mode = planar_mode
        
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
        
        H operates on normalized coordinates [-1, 1].
        
        Two modes:
        - planar_mode=True: Simple identity K, 2D affine in T (for warp_kpts_planar)
        - planar_mode=False: Proper K with focal length, 3D pose T (for warp_kpts)
        """
        if self.planar_mode:
            # PLANAR MODE: Simple geometry for warp_kpts_planar
            # warp_kpts_planar uses only T[:2,:2] and T[:2,3] in normalized coords
            K = np.eye(3, dtype=np.float32)
            
            T = np.eye(4, dtype=np.float32)
            T[0:2, 0:2] = H[0:2, 0:2]  # 2D rotation/scale
            T[0, 3] = H[0, 2]          # x translation
            T[1, 3] = H[1, 2]          # y translation
            
            return K, T
        else:
            # NON-PLANAR MODE: Эмулируем planar warp через 3D геометрию
            # Для warp_kpts нужно чтобы: K^-1 @ [x,y,1]*d → 3D → T → K @ → [x',y']
            # давало тот же результат что H @ [x,y,1] в normalized coords
            
            # K: normalized [-1,1] → pixel [0, image_size]
            # Используем focal_length = image_size/2, чтобы нормализованные координаты
            # корректно мапились в 3D space
            K = np.array([
                [image_size / 2.0, 0, image_size / 2.0],
                [0, image_size / 2.0, image_size / 2.0],
                [0, 0, 1]
            ], dtype=np.float32)
            
            # Для planar scene на глубине d=1:
            # warp_kpts делает: x_3d = K^-1 @ [x,y,1] * d
            #                   x'_3d = R @ x_3d + t
            #                   [x',y'] = K @ x'_3d / x'_3d[2]
            #
            # Для ортофото: R должна быть in-plane rotation, t - in-plane translation
            # При d=1 везде, глубина должна сохраняться: x'_3d[2] = 1
            
            # H работает на normalized coords, конвертируем в affine на [-1,1]:
            # Извлекаем компоненты H (уже в normalized space)
            a11, a12, tx = H[0, 0], H[0, 1], H[0, 2]
            a21, a22, ty = H[1, 0], H[1, 1], H[1, 2]
            
            # Для сохранения planar геометрии: R - in-plane rotation
            # t - in-plane translation scaled на depth
            # При depth=1: x' = R @ x + t где x,x' в normalized space с depth
            
            # R: только in-plane (z-axis rotation)
            R = np.array([
                [a11, a12, 0],
                [a21, a22, 0],
                [0, 0, 1]
            ], dtype=np.float32)
            
            # Translation: масштабируем на 1/focal_length чтобы учесть масштаб K
            # tx, ty в normalized coords [-1,1], нужно в camera space при d=1
            t = np.array([tx, ty, 0], dtype=np.float32)
            
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = R
            T[:3, 3] = t
            
            return K, T

    def horizontal_flip(self, im_A, im_B, K1, K2, T_1to2):
        """Horizontal flip augmentation with correction for T_1to2 and K"""
        im_A = im_A.flip(-1)
        im_B = im_B.flip(-1)
        
        if self.planar_mode:
            # PLANAR MODE: Work with normalized 2D coordinates
            # x' = -x => flips sign of x component
            flip_mat = torch.tensor([[-1, 0], [0, 1]], dtype=T_1to2.dtype, device=T_1to2.device)
            T_1to2[:2, :2] = flip_mat @ T_1to2[:2, :2] @ flip_mat
            T_1to2[0, 3] = -T_1to2[0, 3]
        else:
            # NON-PLANAR MODE: Work with 3D geometry
            # Horizontal flip changes camera coordinate system
            # Flip matrix for 3D: reflects across yz-plane
            flip_3d = torch.tensor([
                [-1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ], dtype=T_1to2.dtype, device=T_1to2.device)
            
            # Transform: T'= flip @ T @ flip
            T_1to2[:3, :3] = flip_3d @ T_1to2[:3, :3] @ flip_3d
            T_1to2[:3, 3] = flip_3d @ T_1to2[:3, 3]
            
            # Adjust intrinsics: cx' = width - cx
            K1_new = K1.clone()
            K2_new = K2.clone()
            K1_new[0, 2] = self.image_size - K1[0, 2]
            K2_new[0, 2] = self.image_size - K2[0, 2]
            K1, K2 = K1_new, K2_new
        
        return im_A, im_B, K1, K2, T_1to2
    
    def vertical_flip(self, im_A, im_B, K1, K2, T_1to2):
        """Vertical flip augmentation с корректировкой intrinsics"""
        im_A = im_A.flip(-2)
        im_B = im_B.flip(-2)
        
        if self.planar_mode:
            # PLANAR MODE: Only adjust K (used minimally in planar mode)
            flip_mat = torch.tensor([
                [1, 0, 0],
                [0, -1, self.image_size],
                [0, 0, 1.]
            ], dtype=K1.dtype, device=K1.device)
            
            K1 = flip_mat @ K1
            K2 = flip_mat @ K2
        else:
            # NON-PLANAR MODE: Adjust both K and T
            flip_3d = torch.tensor([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, 1]
            ], dtype=T_1to2.dtype, device=T_1to2.device)
            
            T_1to2[:3, :3] = flip_3d @ T_1to2[:3, :3] @ flip_3d
            T_1to2[:3, 3] = flip_3d @ T_1to2[:3, 3]
            
            K1_new = K1.clone()
            K2_new = K2.clone()
            K1_new[1, 2] = self.image_size - K1[1, 2]
            K2_new[1, 2] = self.image_size - K2[1, 2]
            K1, K2 = K1_new, K2_new
        
        return im_A, im_B, K1, K2, T_1to2
    
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
            
            if self.planar_mode:
                # PLANAR MODE: Work in normalized coordinates
                t_norm = t / self.image_size * 2
                T_1to2[0, 3] -= t_norm[0]
                T_1to2[1, 3] -= t_norm[1]
            else:
                # NON-PLANAR MODE: Adjust translation in camera coordinates
                # Translation in pixels needs to be converted to camera units
                t_cam = t.astype(np.float32)  # pixels
                T_1to2[0, 3] -= t_cam[0]
                T_1to2[1, 3] -= t_cam[1]

        # Augmentations (только для planar_mode=True, для False требуется доработка)
        if self.planar_mode:
            # Horizontal flip
            if self.use_horizontal_flip_aug and np.random.rand() > 0.5:
                im1, im2, K1, K2, T_1to2 = self.horizontal_flip(im1, im2, K1, K2, T_1to2)
            
            # Vertical flip
            if self.use_vertical_flip_aug and np.random.rand() > 0.5:
                im1, im2, K1, K2, T_1to2 = self.vertical_flip(im1, im2, K1, K2, T_1to2)
        else:
            # TODO: Корректные augmentations для 3D geometry
            pass

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

