"""
Скрипт обучения RoMa на данных Umbra SAR.

Использует UmbraScene датасет для обучения модели сопоставления радарных изображений.
"""

import os
import sys
import torch
from argparse import ArgumentParser
import json

from torch import nn
from torch.utils.data import ConcatDataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
from matplotlib import pyplot as plt
from romatch.benchmarks.umbra_dense_benchmark import UmbraDenseBenchmark
from romatch.datasets.umbra import UmbraScene
from romatch.losses.robust_loss import RobustLosses
from romatch.train.train import train_k_steps
from romatch.models.matcher import *
from romatch.models.transformer import Block, TransformerDecoder, MemEffAttention
from romatch.models.encoders import *
from romatch.models.planar_decoder import PlanarDecoder, SimplifiedGlobalMatcher
from romatch.checkpointing import CheckPoint
from torch.utils.tensorboard import SummaryWriter
from romatch.tools.visualize_pipeline import visualize_total


resolutions = {
    "low": (448, 448),  # 14*8*4
    "medium": (14 * 8 * 5, 14 * 8 * 5),  # 560x560
    "high": (14 * 8 * 6, 14 * 8 * 6),  # 672x672
    "mega": (14 * 8 * 9, 14 * 8 * 9),  # 1008x1008 - center crop из 1024
    "ultra": (14 * 8 * 10, 14 * 8 * 10),  # 1120x1120 - upscale на 9.4%
}


def get_planar_model(pretrained_backbone=True, resolution="medium", **kwargs):
    """
    Создает модель с оптимизированным декодером для planar scenes (SAR).

    Основные отличия:
    - SimplifiedGlobalMatcher вместо GP + TransformerDecoder
    - Увеличенные радиусы local correlation (11, 7, 4 вместо 7, 3, 2)
    - Прямая regression вместо classification
    """
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

    # ConvRefiners с увеличенными радиусами для SAR
    dw = True
    hidden_blocks = 8
    kernel_size = 5
    displacement_emb = "linear"
    disable_local_corr_grad = True

    conv_refiner = nn.ModuleDict(
        {
            "16": ConvRefiner(
                2 * 512 + 128 + (2 * 11 + 1) ** 2,  # radius 11 (было 7)
                2 * 512 + 128 + (2 * 11 + 1) ** 2,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=128,
                local_corr_radius=11,  # увеличено с 7
                corr_in_other=True,
                amp=True,
                disable_local_corr_grad=disable_local_corr_grad,
                bn_momentum=0.01,
            ),
            "8": ConvRefiner(
                2 * 512 + 64 + (2 * 7 + 1) ** 2,  # radius 7 (было 3)
                2 * 512 + 64 + (2 * 7 + 1) ** 2,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=64,
                local_corr_radius=7,  # увеличено с 3
                corr_in_other=True,
                amp=True,
                disable_local_corr_grad=disable_local_corr_grad,
                bn_momentum=0.01,
            ),
            "4": ConvRefiner(
                2 * 256 + 32 + (2 * 4 + 1) ** 2,  # radius 4 (было 2)
                2 * 256 + 32 + (2 * 4 + 1) ** 2,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=32,
                local_corr_radius=4,  # увеличено с 2
                corr_in_other=True,
                amp=True,
                disable_local_corr_grad=disable_local_corr_grad,
                bn_momentum=0.01,
            ),
            "2": ConvRefiner(
                2 * 64 + 16,
                128 + 16,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=16,
                amp=True,
                disable_local_corr_grad=disable_local_corr_grad,
                bn_momentum=0.01,
            ),
            "1": ConvRefiner(
                2 * 9 + 6,
                24,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=6,
                amp=True,
                disable_local_corr_grad=disable_local_corr_grad,
                bn_momentum=0.01,
            ),
        }
    )

    # Global matcher
    global_matcher = SimplifiedGlobalMatcher(feat_dim=512, hidden_dim=512)

    # Feature projections
    proj16 = nn.Sequential(nn.Conv2d(1024, 512, 1, 1), nn.BatchNorm2d(512))
    proj8 = nn.Sequential(nn.Conv2d(512, 512, 1, 1), nn.BatchNorm2d(512))
    proj4 = nn.Sequential(nn.Conv2d(256, 256, 1, 1), nn.BatchNorm2d(256))
    proj2 = nn.Sequential(nn.Conv2d(128, 64, 1, 1), nn.BatchNorm2d(64))
    proj1 = nn.Sequential(nn.Conv2d(64, 9, 1, 1), nn.BatchNorm2d(9))

    proj = nn.ModuleDict(
        {
            "16": proj16,
            "8": proj8,
            "4": proj4,
            "2": proj2,
            "1": proj1,
        }
    )

    # Planar decoder
    decoder = PlanarDecoder(
        global_matcher,
        proj,
        conv_refiner,
        detach=True,
        scales=["16", "8", "4", "2", "1"],
    )

    h, w = resolutions[resolution]
    encoder = CNNandDinov2(
        cnn_kwargs=dict(pretrained=pretrained_backbone, amp=True),
        amp=True,
        use_vgg=True,
    )

    matcher = RegressionMatcher(encoder, decoder, h=h, w=w, **kwargs)
    return matcher


def get_model(pretrained_backbone=True, resolution="medium", **kwargs):
    """Создает модель RoMa с заданными параметрами (стандартная версия)"""
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

    gp_dim = 512
    feat_dim = 512
    decoder_dim = gp_dim + feat_dim
    cls_to_coord_res = 64

    coordinate_decoder = TransformerDecoder(
        nn.Sequential(*[Block(decoder_dim, 8, attn_class=MemEffAttention) for _ in range(5)]),
        decoder_dim,
        cls_to_coord_res**2 + 1,
        is_classifier=True,
        amp=True,
        pos_enc=False,
    )

    dw = True
    hidden_blocks = 8
    kernel_size = 5
    displacement_emb = "linear"
    disable_local_corr_grad = True

    conv_refiner = nn.ModuleDict(
        {
            "16": ConvRefiner(
                2 * 512 + 128 + (2 * 7 + 1) ** 2,
                2 * 512 + 128 + (2 * 7 + 1) ** 2,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=128,
                local_corr_radius=7,
                corr_in_other=True,
                amp=True,
                disable_local_corr_grad=disable_local_corr_grad,
                bn_momentum=0.01,
            ),
            "8": ConvRefiner(
                2 * 512 + 64 + (2 * 3 + 1) ** 2,
                2 * 512 + 64 + (2 * 3 + 1) ** 2,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=64,
                local_corr_radius=3,
                corr_in_other=True,
                amp=True,
                disable_local_corr_grad=disable_local_corr_grad,
                bn_momentum=0.01,
            ),
            "4": ConvRefiner(
                2 * 256 + 32 + (2 * 2 + 1) ** 2,
                2 * 256 + 32 + (2 * 2 + 1) ** 2,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=32,
                local_corr_radius=2,
                corr_in_other=True,
                amp=True,
                disable_local_corr_grad=disable_local_corr_grad,
                bn_momentum=0.01,
            ),
            "2": ConvRefiner(
                2 * 64 + 16,
                128 + 16,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=16,
                amp=True,
                disable_local_corr_grad=disable_local_corr_grad,
                bn_momentum=0.01,
            ),
            "1": ConvRefiner(
                2 * 9 + 6,
                24,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=6,
                amp=True,
                disable_local_corr_grad=disable_local_corr_grad,
                bn_momentum=0.01,
            ),
        }
    )

    kernel_temperature = 0.2
    learn_temperature = False
    no_cov = True
    kernel = CosKernel
    only_attention = False
    basis = "fourier"

    gp16 = GP(
        kernel,
        T=kernel_temperature,
        learn_temperature=learn_temperature,
        only_attention=only_attention,
        gp_dim=gp_dim,
        basis=basis,
        no_cov=no_cov,
    )
    gps = nn.ModuleDict({"16": gp16})

    proj16 = nn.Sequential(nn.Conv2d(1024, 512, 1, 1), nn.BatchNorm2d(512))
    proj8 = nn.Sequential(nn.Conv2d(512, 512, 1, 1), nn.BatchNorm2d(512))
    proj4 = nn.Sequential(nn.Conv2d(256, 256, 1, 1), nn.BatchNorm2d(256))
    proj2 = nn.Sequential(nn.Conv2d(128, 64, 1, 1), nn.BatchNorm2d(64))
    proj1 = nn.Sequential(nn.Conv2d(64, 9, 1, 1), nn.BatchNorm2d(9))

    proj = nn.ModuleDict(
        {
            "16": proj16,
            "8": proj8,
            "4": proj4,
            "2": proj2,
            "1": proj1,
        }
    )

    displacement_dropout_p = 0.0
    gm_warp_dropout_p = 0.0

    decoder = Decoder(
        coordinate_decoder,
        gps,
        proj,
        conv_refiner,
        detach=True,
        scales=["16", "8", "4", "2", "1"],
        displacement_dropout_p=displacement_dropout_p,
        gm_warp_dropout_p=gm_warp_dropout_p,
    )

    h, w = resolutions[resolution]
    encoder = CNNandDinov2(
        cnn_kwargs=dict(pretrained=pretrained_backbone, amp=True),
        amp=True,
        use_vgg=True,
    )

    matcher = RegressionMatcher(encoder, decoder, h=h, w=w, **kwargs)
    return matcher


def train(args):
    warnings.filterwarnings(
        "ignore", category=UserWarning, message="WARNING batched routines are designed for small sizes."
    )
    """Основная функция обучения"""
    dist.init_process_group("nccl")
    gpus = int(os.environ["WORLD_SIZE"])

    rank = dist.get_rank()
    print(f"Start running DDP on rank {rank}")
    device_id = rank % torch.cuda.device_count()
    romatch.LOCAL_RANK = device_id
    torch.cuda.set_device(device_id)

    resolution = args.train_resolution
    experiment_name = os.path.splitext(os.path.basename(__file__))[0]

    wandb.init(
        project="romatch_umbra",
        mode="disabled",
    )

    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    # TensorBoard writer (только на rank 0)
    tb_writer = None
    if rank == 0:
        tb_log_dir = os.path.join(checkpoint_dir, "tensorboard")
        os.makedirs(tb_log_dir, exist_ok=True)
        tb_writer = SummaryWriter(tb_log_dir)
        print(f"TensorBoard logging to: {tb_log_dir}")

    h, w = resolutions[resolution]

    # Использовать planar decoder для SAR
    use_planar_decoder = getattr(args, "use_planar_decoder", True)
    if use_planar_decoder:
        print("Using PlanarDecoder optimized for SAR orthophotos")
        model = get_planar_model(pretrained_backbone=True, resolution=resolution, attenuate_cert=False).to(device_id)
    else:
        print("Using standard RoMa decoder")
        model = get_model(pretrained_backbone=True, resolution=resolution, attenuate_cert=False).to(device_id)

    # Параметры обучения
    global_step = 0
    batch_size = args.gpu_batch_size
    step_size = gpus * batch_size
    romatch.STEP_SIZE = step_size

    # Общее количество шагов
    N = args.total_steps * step_size
    # Сохранение чекпоинта каждые k шагов
    k = args.checkpoint_every // romatch.STEP_SIZE

    # Загрузка данных
    print(f"Loading training data from: {args.train_data}")
    with open(args.train_data, "r") as f:
        scene_info = json.load(f)

    umbra_train_scene = UmbraScene(
        scene_info,
        image_size=h,
        scene_name="umbra_train",
        use_horizontal_flip_aug=True,
        use_vertical_flip_aug=True,
        shake_t=32,
    )
    umbra_train = ConcatDataset([umbra_train_scene])

    # Веса для сэмплирования (равномерные для одной сцены)
    umbra_ws = torch.ones(len(umbra_train))

    depth_loss = RobustLosses(
        ce_weight=0.03,  # было 0.01 → +200% для certainty
        local_dist={1:6, 2:6, 4:12, 8:12},  # было {1:4, 2:4, 4:8, 8:8} → {1:6, 2:6, 4:12, 8:12}
        local_largest_scale=8,  # без изменений
        alpha=0.5,  # без изменений
        c=3e-4,  # было 1e-4 → +200% порог
        scale_weights={
            1: 1.5,
            2: 1,
            4: 0.7,
            8: 0.5,
            16: 0.2,
        },  # новое (SAR имеет меньшую точность на грубых масштабах → уменьшить их вклад в loss)
        planar_mode=args.planar_mode,
    )
    parameters = [
        {"params": model.encoder.parameters(), "lr": romatch.STEP_SIZE * 5e-6},
        {"params": model.decoder.parameters(), "lr": romatch.STEP_SIZE * 1e-4},
    ]

    optimizer = torch.optim.AdamW(parameters, weight_decay=0.01)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, threshold=0.01, threshold_mode="rel"
    )

    # Бенчмарк для валидации (БЕЗ аугментаций!)
    if args.val_data and os.path.exists(args.val_data):
        print(f"Loading validation data from: {args.val_data}")
        with open(args.val_data, "r") as f:
            val_scene_info = json.load(f)

        vis_dir = os.path.join(args.checkpoint_dir, "visualizations")
        umbra_benchmark = UmbraDenseBenchmark(
            scene_info=val_scene_info, image_size=h, vis_dir=vis_dir, planar_mode=args.planar_mode
        )
    else:
        umbra_benchmark = None
        print("No validation data provided, skipping benchmark")

    # Checkpointer
    checkpointer = CheckPoint(checkpoint_dir, experiment_name)
    if args.flush:
        model, _, _, _ = checkpointer.load(model, optimizer, lr_scheduler, n=global_step, postfix="best")
    else:
        model, optimizer, lr_scheduler, global_step = checkpointer.load(
            model, optimizer, lr_scheduler, n=global_step, postfix="best"
        )
    romatch.GLOBAL_STEP = global_step

    def tb_visualize_callback(batch, step):
        """Callback для визуализации одного батча в TensorBoard."""
        if tb_writer is None or rank != 0:
            return

        print("Generating training batch visualization for TensorBoard...")
        model.eval()
        im_A, im_B = batch["im_A"], batch["im_B"]
        batch_size = im_A.shape[0]
        with torch.no_grad():
            matches, certainty = model.match(im_A, im_B, batched=True)
        figures = [
            visualize_total(
                im_A=im_A[b],
                im_B=im_B[b],
                matches=matches[b],
                certainty=certainty[b],
                T_1to2=batch["T_1to2"][b],
                K1=batch["K1"][b],
                K2=batch["K2"][b],
                depth1=batch["im_A_depth"][b],
                depth2=batch["im_B_depth"][b],
                planar_mode=args.planar_mode,
            )[0]
            for b in range(batch_size)
        ]
        for i, fig in enumerate(figures):
            tb_writer.add_figure(f"Train/match_visualization_{i}", fig, global_step=step)

    # DDP модель
    ddp_model = DDP(model, device_ids=[device_id], find_unused_parameters=False, gradient_as_bucket_view=True)

    grad_scaler = torch.amp.GradScaler("cuda", growth_interval=1_000_000)
    grad_clip_norm = 0.01
    best_epe = 1e7  # less means better
    # Цикл обучения
    for n in range(romatch.GLOBAL_STEP, N, k * romatch.STEP_SIZE):
        umbra_sampler = torch.utils.data.WeightedRandomSampler(umbra_ws, num_samples=batch_size * k, replacement=True)
        umbra_dataloader = iter(
            torch.utils.data.DataLoader(
                umbra_train,
                batch_size=batch_size,
                sampler=umbra_sampler,
                num_workers=args.num_workers,
            )
        )

        train_k_steps(
            n,
            k,
            umbra_dataloader,
            ddp_model,
            depth_loss,
            optimizer,
            lr_scheduler,
            grad_scaler,
            grad_clip_norm=grad_clip_norm,
            accumulation_steps=4,  # Эффективный батч-сайз 4*4=16
            writer=tb_writer,
            vis_callback=tb_visualize_callback if rank == 0 else None,
        )

        # Запуск бенчмарка и визуализаций
        if umbra_benchmark is not None and rank == 0:
            print(f"\n=== Running validation at step {romatch.GLOBAL_STEP} ===")
            benchmark_results = umbra_benchmark.benchmark(model, batch_size=args.gpu_batch_size)
            wandb.log(benchmark_results, step=romatch.GLOBAL_STEP)

            # TensorBoard визуализации на фиксированном батче
            if tb_writer is not None:
                print("Generating TensorBoard visualizations...")
                # Логируем метрики из benchmark
                for key, value in benchmark_results.items():
                    if isinstance(value, (int, float)):
                        tb_writer.add_scalar(f"Benchmark/{key}", scalar_value=value, global_step=romatch.GLOBAL_STEP)
                for i, v in enumerate(benchmark_results["visual"]):
                    tb_writer.add_figure(f"Benchmark/visual_{i}", figure=v, global_step=romatch.GLOBAL_STEP)
            lr_scheduler.step(benchmark_results["umbra_epe"])
            if benchmark_results["umbra_epe"] < best_epe:
                best_epe = benchmark_results["umbra_epe"]
                checkpointer.save(model, optimizer, lr_scheduler, romatch.GLOBAL_STEP, postfix="best")

        # Сохранение чекпоинта
        checkpointer.save(model, optimizer, lr_scheduler, romatch.GLOBAL_STEP, postfix="latest")
    print(f"Training completed! Final step: {romatch.GLOBAL_STEP}")


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore", category=UserWarning, message="WARNING batched routines are designed for small sizes."
    )
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"  # For BF16 computations
    os.environ["OMP_NUM_THREADS"] = "16"
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    import romatch

    parser = ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True, help="Путь к JSON файлу с тренировочными данными")
    parser.add_argument("--val_data", type=str, default=None, help="Путь к JSON файлу с валидационными данными")
    parser.add_argument(
        "--checkpoint_dir", type=str, default="workspace/checkpoints_umbra", help="Директория для сохранения чекпоинтов"
    )
    parser.add_argument("--only_test", action="store_true", help="Только тестирование, без обучения")
    parser.add_argument("--debug_mode", action="store_true", help="Режим отладки")
    parser.add_argument("--dont_log_wandb", action="store_true", help="Отключить логирование в wandb")
    parser.add_argument(
        "--train_resolution",
        default="mega",
        choices=["low", "medium", "high", "mega", "ultra"],
        help="Разрешение для обучения (mega=1008 center crop из 1024, ultra=1120 upscale)",
    )
    parser.add_argument("--gpu_batch_size", default=4, type=int, help="Размер батча на одну GPU")
    parser.add_argument("--total_steps", default=50000, type=int, help="Общее количество шагов обучения")
    parser.add_argument("--checkpoint_every", default=5000, type=int, help="Частота сохранения чекпоинтов")
    parser.add_argument("--val_samples", default=200, type=int, help="Количество пар для валидации")
    parser.add_argument(
        "--num_vis_samples", default=4, type=int, help="Количество фиксированных сэмплов для TensorBoard визуализаций"
    )
    parser.add_argument("--num_workers", default=4, type=int, help="Количество worker'ов для DataLoader")
    parser.add_argument("--wandb_entity", required=False, help="WandB entity")
    parser.add_argument("--flush", action="store_true", help="Не использовать global_step и состояние lr_scheduler")
    parser.add_argument(
        "--use_planar_decoder", action="store_true", default=True, help="Использовать PlanarDecoder вместо стандартного"
    )
    parser.add_argument(
        "--planar_mode", action="store_true", default=True, help="Использовать 2D геометрию для planar сцен (SAR)"
    )

    args, _ = parser.parse_known_args()
    romatch.DEBUG_MODE = args.debug_mode

    if not args.only_test:
        train(args)
    else:
        print("Test mode not implemented yet")
