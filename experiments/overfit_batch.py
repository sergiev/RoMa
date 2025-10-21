import argparse
from romatch.utils import warp_kpts, warp_kpts_planar
import torch
import json
import numpy as np
from torch.utils.data import DataLoader
import sys
import os
from torch.utils.tensorboard import SummaryWriter

# Добавляем корень проекта в PYTHONPATH для корректного импорта
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from romatch.datasets.umbra import UmbraScene
from experiments.train_roma_umbra import get_planar_model
from romatch.losses.robust_loss import RobustLosses
from romatch.tools.visualize_pipeline import visualize_total
from romatch.train.train import train_k_steps
import romatch as romatch
import wandb


def geometric_dist(dense_matches, depth1, depth2, T_1to2, K1, K2, planar_mode):
    """Вычисляет геометрическое расстояние между предсказанными и истинными соответствиями"""
    b, h1, w1, d = dense_matches.shape
    with torch.no_grad():
        x1 = dense_matches[..., :2].reshape(b, h1 * w1, 2)
        if planar_mode:
            mask, x2 = warp_kpts_planar(x1.double(), T_1to2.double())
        else:
            mask, x2 = warp_kpts(
                x1.double(),
                depth1.double(),
                depth2.double(),
                T_1to2.double(),
                K1.double(),
                K2.double(),
            )
        x2 = torch.stack((w1 * (x2[..., 0] + 1) / 2, h1 * (x2[..., 1] + 1) / 2), dim=-1)
        prob = mask.float().reshape(b, h1, w1)
    x2_hat = dense_matches[..., 2:]
    x2_hat = torch.stack((w1 * (x2_hat[..., 0] + 1) / 2, h1 * (x2_hat[..., 1] + 1) / 2), dim=-1)
    gd = (x2_hat - x2.reshape(b, h1, w1, 2)).norm(dim=-1)
    gd = gd[prob == 1]
    pck_1 = (gd < 1.0).float().mean()
    pck_3 = (gd < 3.0).float().mean()
    pck_5 = (gd < 5.0).float().mean()
    return gd, pck_1, pck_3, pck_5


def run_and_log_benchmark(model, batch, tb_writer, step, prefix=""):
    """Runs benchmark logic on a single batch and logs results."""
    print(f"\n--- Running validation for step {step} ---")
    model.eval()
    planar_mode = True  # Hardcoded for this script
    with torch.no_grad():
        matches, certainty = model.match(batch["im_A"], batch["im_B"], batched=True)
        gd, pck_1, pck_3, pck_5 = geometric_dist(
            matches,
            batch["im_A_depth"],
            batch["im_B_depth"],
            batch["T_1to2"],
            batch["K1"],
            batch["K2"],
            planar_mode,
        )

    results = {
        "epe": gd.mean().item(),
        "pck_1": pck_1.item(),
        "pck_3": pck_3.item(),
        "pck_5": pck_5.item(),
    }

    # Generate visualization
    visual = [
        visualize_total(
            im_A=batch["im_A"][b],
            im_B=batch["im_B"][b],
            matches=matches[b],
            certainty=certainty[b],
            T_1to2=batch["T_1to2"][b],
            K1=batch["K1"][b],
            K2=batch["K2"][b],
            depth1=batch["im_A_depth"][b],
            depth2=batch["im_B_depth"][b],
            planar_mode=planar_mode,
        )[0]
        for b in range(batch["im_A"].shape[0])
    ]
    results["visual"] = visual

    # Log to TensorBoard
    log_benchmark_results(tb_writer, results, step, prefix)
    model.train()  # Set model back to train mode


class FixedBatchLoader:
    """A dummy dataloader that always returns the same batch, infinitely."""

    def __init__(self, batch):
        self.batch = batch

    def __iter__(self):
        return self

    def __next__(self):
        return self.batch


def log_benchmark_results(tb_writer, results, step, prefix=""):
    """Logs benchmark results to TensorBoard."""
    if tb_writer is None:
        return
    print(f"Logging benchmark results to TensorBoard at step {step} with prefix '{prefix}'...")
    for key, value in results.items():
        if isinstance(value, (int, float)):
            tb_writer.add_scalar(f"Benchmark/{prefix}{key}", scalar_value=value, global_step=step)
    for i, fig in enumerate(results.get("visual", [])):
        tb_writer.add_figure(f"Benchmark/{prefix}visual_{i}", figure=fig, global_step=step)


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

    # TensorBoard writer
    tb_log_dir = f"workspace/overfit_logs_seed_{args.seed}"
    os.makedirs(tb_log_dir, exist_ok=True)
    tb_writer = SummaryWriter(tb_log_dir)
    print(f"TensorBoard logging to: {tb_log_dir}")

    wandb.init(project="romatch_verification", mode="disabled")

    with open(train_data_path, "r") as f:
        scene_info = json.load(f)
    dataset = UmbraScene(scene_info, image_size=image_size, shake_t=16)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    fixed_batch_raw = next(iter(data_loader))
    fixed_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in fixed_batch_raw.items()}

    print("Загрузка модели...")
    model = get_planar_model(pretrained_backbone=True, resolution="medium").to(device)

    # --- Валидация до обучения ---
    run_and_log_benchmark(model, fixed_batch, tb_writer, 0, prefix="Before_")

    model.train()

    romatch.STEP_SIZE = batch_size
    romatch.GLOBAL_STEP = 0

    loss_fn = RobustLosses(
        ce_weight=0.0003,
        local_dist={1: 8, 2: 12, 4: 16, 8: 20},
        planar_mode=True,
    )

    parameters = [
        {"params": model.encoder.parameters(), "lr": batch_size * 5e-6},  # Возвращено к исходному
        {"params": model.decoder.parameters(), "lr": batch_size * 1e-4},  # Возвращено к исходному
    ]
    optimizer = torch.optim.AdamW(parameters, weight_decay=0.01)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, threshold=0.01, threshold_mode="rel"
    )
    grad_scaler = torch.amp.GradScaler("cuda", growth_interval=1_000_000)

    print(f"\n--- Запуск цикла переобучения на {num_iterations} итераций ---")

    # Вместо цикла используем train_k_steps
    # Создаем data loader, который всегда возвращает один и тот же батч
    fixed_dataloader = FixedBatchLoader(fixed_batch)

    train_k_steps(
        0,
        num_iterations,
        dataloader=fixed_dataloader,
        model=model,
        objective=loss_fn,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        grad_scaler=grad_scaler,
        grad_clip_norm=0.01,
        writer=tb_writer,
    )

    print("\nПереобучение завершено.")

    # --- Валидация после обучения ---
    run_and_log_benchmark(model, fixed_batch, tb_writer, num_iterations, prefix="After_")

    # Проверку по значению потерь убираем, так как train_k_steps не возвращает историю потерь
    print("\n--- Тест на переобучение завершен ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed for data loading.")
    args = parser.parse_args()
    overfit_batch(args)
