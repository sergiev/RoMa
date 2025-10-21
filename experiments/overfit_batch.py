import argparse
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import sys
import os

# Добавляем корень проекта в PYTHONPATH для корректного импорта
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'RoMa'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from romatch.datasets.umbra import UmbraScene
from experiments.train_roma_umbra import get_planar_model
from romatch.losses.robust_loss import RobustLosses
from romatch.tools.visualize_pipeline import visualize_total
from romatch.utils import warp_kpts_planar
import romatch as romatch
import wandb


def get_matches_from_flow(flow):
    """
    Преобразует поле потока (flow) в формат соответствий (matches).
    """
    # flow приходит в формате [B, 2, H, W]
    B, _, H, W = flow.shape
    device = flow.device
    
    # Создаем сетку нормализованных координат для изображения A
    y_coords, x_coords = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )
    # grid в формате [H, W, 2]
    grid = torch.stack((x_coords, y_coords), dim=-1)
    # expand до [B, H, W, 2]
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)
    
    # Приводим flow к формату [B, H, W, 2] для сложения
    flow_permuted = flow.permute(0, 2, 3, 1)
    
    # Координаты в изображении B = координаты в A + смещение (flow)
    coords_B = grid + flow_permuted
    
    # Объединяем в формат matches (xA, yA, xB, yB)
    matches = torch.cat([grid, coords_B], dim=-1)
    return matches


def generate_visualization(model, batch, filename):
    """Генерирует и сохраняет визуализацию для первого элемента в батче, выводит EPE в консоль."""
    print(f"\n--- Создание визуализации: {filename} ---")
    model.eval()
    with torch.no_grad():
        model_output = model(batch, batched=True)
        predictions = model_output[1]
        
        flow = predictions['flow']
        certainty = predictions['certainty']
        matches = get_matches_from_flow(flow)

    # --- Расчет EPE для явного вывода ---
    with torch.no_grad():
        first_item_matches = matches[0]
        h, w = first_item_matches.shape[:2]
        x1_norm = first_item_matches[..., :2].reshape(1, h * w, 2)
        mask_gt_fwd, x2_gt_norm = warp_kpts_planar(x1_norm, batch['T_1to2'][0].unsqueeze(0))
        x2_pred_norm = first_item_matches[..., 2:].reshape(1, h * w, 2)
        
        x2_gt_px = torch.stack((w * (x2_gt_norm[0, :, 0] + 1) / 2, h * (x2_gt_norm[0, :, 1] + 1) / 2), dim=1)
        x2_pred_px = torch.stack((w * (x2_pred_norm[0, :, 0] + 1) / 2, h * (x2_pred_norm[0, :, 1] + 1) / 2), dim=1)
        
        error_forward = (x2_pred_px - x2_gt_px).norm(dim=1)
        valid_mask_fwd = mask_gt_fwd[0].bool()
        valid_errors = error_forward[valid_mask_fwd]
        mean_error = valid_errors.mean().item() if valid_errors.numel() > 0 else float("nan")
        print(f"  [METRIC] Calculated Mean Error (EPE): {mean_error:.2f}px")
    # --- Конец расчета EPE ---

    im_A = batch['im_A'][0]
    im_B = batch['im_B'][0]
    T_1to2 = batch['T_1to2'][0]
    K1 = batch.get('K1', [None])[0]
    K2 = batch.get('K2', [None])[0]
    depth1 = batch.get('depth1', [None])[0]
    depth2 = batch.get('depth2', [None])[0]

    fig = visualize_total(
        im_A=im_A,
        im_B=im_B,
        matches=matches[0],
        certainty=certainty[0].squeeze(),
        T_1to2=T_1to2,
        K1=K1,
        K2=K2,
        depth1=depth1,
        depth2=depth2,
        planar_mode=True,
    )[0]

    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)
    print(f"Визуализация сохранена в: {filename}")


def overfit_batch(args):
    """
    Проверяет способность модели к переобучению на одном батче.
    """
    print("--- Запуск теста на переобучение на одном батче ---")
    print(f"--- Используется SEED: {args.seed} ---")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_data_path = "/home/sema/radar/datasets/umbra_tiles/560/pairs_train.json"
    image_size = 560
    batch_size = 2
    num_iterations = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Используемое устройство: {device}")

    wandb.init(project="romatch_verification", mode="disabled")

    with open(train_data_path, "r") as f:
        scene_info = json.load(f)
    dataset = UmbraScene(scene_info, image_size=image_size, shake_t=16)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    fixed_batch_raw = next(iter(data_loader))
    fixed_batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v 
        for k, v in fixed_batch_raw.items()
    }
    
    print("Загрузка модели...")
    model = get_planar_model(pretrained_backbone=True, resolution="medium").to(device)

    generate_visualization(model, fixed_batch, f"overfit_visualization_before_seed_{args.seed}.png")

    model.train()

    romatch.STEP_SIZE = batch_size
    romatch.GLOBAL_STEP = 0

    loss_fn = RobustLosses(
        ce_weight=0.0003,
        local_dist={1: 8, 2: 12, 4: 16, 8: 20},
        planar_mode=True,
    )
    
    parameters = [
        {"params": model.encoder.parameters(), "lr": batch_size * 5e-6}, # Возвращено к исходному
        {"params": model.decoder.parameters(), "lr": batch_size * 1e-4}, # Возвращено к исходному
    ]
    optimizer = torch.optim.AdamW(parameters)
    
    losses = []
    print(f"\n--- 4. Запуск цикла переобучения на {num_iterations} итераций ---")
    
    for i in range(num_iterations):
        optimizer.zero_grad()
        corresps = model(fixed_batch, batched=True)
        loss = loss_fn(corresps, fixed_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
        if (i + 1) % 20 == 0:
            print(f"Итерация [{i+1}/{num_iterations}], Потери: {loss.item():.6f}")

    print("\nПереобучение завершено.")
    
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel("Итерация")
    plt.ylabel("Потери")
    plt.title(f"Переобучение на одном батче (Seed: {args.seed})")
    plt.grid(True)
    output_path = f"overfit_loss_seed_{args.seed}.png"
    plt.savefig(output_path)
    print(f"График потерь сохранен в: {output_path}")

    generate_visualization(model, fixed_batch, f"overfit_visualization_after_seed_{args.seed}.png")

    final_loss = losses[-1]
    initial_loss = losses[0]
    if final_loss < initial_loss * 0.1:
        print(f"\nУСПЕХ: Потери значительно уменьшились с {initial_loss:.4f} до {final_loss:.4f}.")
    else:
        print(f"\nВНИМАНИЕ: Потери не уменьшились значительно. Начальные: {initial_loss:.4f}, Конечные: {final_loss:.4f}.")

    print("\n--- Тест на переобучение завершен ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed for data loading.")
    args = parser.parse_args()
    overfit_batch(args)
