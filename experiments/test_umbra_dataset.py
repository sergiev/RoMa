"""
Скрипт для тестирования датасета UmbraScene и визуализации примеров.

Использование:
    python test_umbra_dataset.py --data_path /path/to/pairs_val.json --output_dir test_output
"""
import os
import sys
import json
import torch
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
from pathlib import Path

# Добавляем путь к модулю romatch
sys.path.insert(0, str(Path(__file__).parent.parent))

from romatch.datasets.umbra import UmbraScene


def visualize_sample(sample, idx, output_dir):
    """Визуализирует один сэмпл из датасета"""
    
    # Денормализация изображений
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    im_A = sample["im_A"] * imagenet_std + imagenet_mean
    im_B = sample["im_B"] * imagenet_std + imagenet_mean
    im_A = torch.clamp(im_A, 0, 1)
    im_B = torch.clamp(im_B, 0, 1)
    
    # Конвертируем в numpy (H, W, C)
    im_A_np = im_A.permute(1, 2, 0).numpy()
    im_B_np = im_B.permute(1, 2, 0).numpy()
    
    # Получаем остальные данные
    depth_A = sample["im_A_depth"].numpy()
    depth_B = sample["im_B_depth"].numpy()
    K1 = sample["K1"].numpy()
    K2 = sample["K2"].numpy()
    T_1to2 = sample["T_1to2"].numpy()
    
    # Создаем визуализацию
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Изображения
    axes[0, 0].imshow(im_A_np)
    axes[0, 0].set_title(f'Image A\n{Path(sample["im_A_path"]).name}')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(im_B_np)
    axes[0, 1].set_title(f'Image B\n{Path(sample["im_B_path"]).name}')
    axes[0, 1].axis('off')
    
    # Разница между изображениями
    diff = np.abs(im_A_np - im_B_np).mean(axis=2)
    axes[0, 2].imshow(diff, cmap='hot')
    axes[0, 2].set_title('Absolute Difference')
    axes[0, 2].axis('off')
    
    # Depth maps
    im_depth_A = axes[1, 0].imshow(depth_A, cmap='viridis')
    axes[1, 0].set_title('Depth A')
    axes[1, 0].axis('off')
    plt.colorbar(im_depth_A, ax=axes[1, 0], fraction=0.046)
    
    im_depth_B = axes[1, 1].imshow(depth_B, cmap='viridis')
    axes[1, 1].set_title('Depth B')
    axes[1, 1].axis('off')
    plt.colorbar(im_depth_B, ax=axes[1, 1], fraction=0.046)
    
    # Информация о матрицах
    info_text = f"""Intrinsics K1:
{K1}

Intrinsics K2:
{K2}

Transform T_1to2:
{T_1to2}

Translation:
tx: {T_1to2[0, 3]:.2f}m
ty: {T_1to2[1, 3]:.2f}m
tz: {T_1to2[2, 3]:.2f}m
"""
    axes[1, 2].text(0.1, 0.5, info_text, 
                    fontfamily='monospace', 
                    fontsize=8,
                    verticalalignment='center')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Сохраняем
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"sample_{idx:04d}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization to: {save_path}")


def test_dataset(data_path, output_dir, num_samples=5):
    """Тестирует загрузку и визуализацию датасета"""
    
    print(f"Loading dataset from: {data_path}")
    
    # Загружаем метаданные
    with open(data_path, "r") as f:
        scene_info = json.load(f)
    
    print(f"Found {len(scene_info['image_paths'])} images")
    print(f"Found {len(scene_info['pairs'])} pairs")
    
    # Создаем датасет
    dataset = UmbraScene(scene_info, image_size=640, scene_name="test")
    print(f"Dataset created with {len(dataset)} samples")
    
    # Выбираем случайные индексы
    if num_samples > len(dataset):
        num_samples = len(dataset)
    
    indices = np.random.choice(len(dataset), size=num_samples, replace=False)
    
    # Визуализируем каждый сэмпл
    print(f"\nVisualizing {num_samples} samples...")
    for i, idx in enumerate(indices):
        print(f"Processing sample {i+1}/{num_samples} (index {idx})...")
        try:
            sample = dataset[idx]
            visualize_sample(sample, idx, output_dir)
            
            # Выводим статистику
            print(f"  Image A shape: {sample['im_A'].shape}")
            print(f"  Image B shape: {sample['im_B'].shape}")
            print(f"  Depth A shape: {sample['im_A_depth'].shape}")
            print(f"  Paths: {Path(sample['im_A_path']).name} <-> {Path(sample['im_B_path']).name}")
            print()
            
        except Exception as e:
            print(f"  Error processing sample {idx}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nTest completed! Visualizations saved to: {output_dir}")
    
    # Статистика по всему датасету
    print("\n=== Dataset Statistics ===")
    print(f"Total pairs: {len(dataset)}")
    print(f"Total images: {len(scene_info['image_paths'])}")
    
    # Считаем уникальные сцены
    scenes = set()
    for path in scene_info['image_paths']:
        # Предполагаем структуру: .../scene_name/tile_id/image.tif
        parts = Path(path).parts
        if len(parts) >= 3:
            scene_name = parts[-3]
            scenes.add(scene_name)
    print(f"Unique scenes: {len(scenes)}")
    if scenes:
        print(f"Scene names: {', '.join(sorted(scenes))}")


def main():
    parser = ArgumentParser(description="Test UmbraScene dataset")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Путь к JSON файлу с парами")
    parser.add_argument("--output_dir", type=str, default="test_output",
                       help="Директория для сохранения визуализаций")
    parser.add_argument("--num_samples", type=int, default=5,
                       help="Количество сэмплов для визуализации")
    
    args = parser.parse_args()
    
    test_dataset(args.data_path, args.output_dir, args.num_samples)


if __name__ == "__main__":
    main()


