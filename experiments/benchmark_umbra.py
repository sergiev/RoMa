"""
Скрипт для запуска бенчмарка на Umbra датасете.

Использование:
    python benchmark_umbra.py --val_data /path/to/pairs_val.json --checkpoint /path/to/model.pth

Пример:
    python benchmark_umbra.py \
        --val_data /home/sema/radar/datasets/umbra_tiles/pairs_val.json \
        --checkpoint workspace/checkpoints_umbra/train_roma_umbra_latest.pth \
        --image_size 640 \
        --num_samples 500 \
        --vis_dir results/umbra_vis
"""
import os
import sys
import torch
import json
from argparse import ArgumentParser
from pathlib import Path

# Добавляем путь к модулю romatch
sys.path.insert(0, str(Path(__file__).parent.parent))

from romatch.benchmarks.umbra_dense_benchmark import UmbraDenseBenchmark
from romatch.models.matcher import *
from romatch.models.transformer import Block, TransformerDecoder, MemEffAttention
from romatch.models.encoders import *
import romatch

resolutions = {
    "low": (448, 448),               # 14*8*4
    "medium": (14*8*5, 14*8*5),      # 560x560
    "high": (14*8*6, 14*8*6),        # 672x672
    "mega": (14*8*9, 14*8*9),        # 1008x1008 - center crop из 1024
    "ultra": (14*8*10, 14*8*10),     # 1120x1120
}


def get_model(pretrained_backbone=True, resolution="medium", **kwargs):
    """Создает модель RoMa"""
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
    
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
    
    conv_refiner = nn.ModuleDict({
        "16": ConvRefiner(
            2 * 512+128+(2*7+1)**2,
            2 * 512+128+(2*7+1)**2,
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
            2 * 512+64+(2*3+1)**2,
            2 * 512+64+(2*3+1)**2,
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
            2 * 256+32+(2*2+1)**2,
            2 * 256+32+(2*2+1)**2,
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
            2 * 64+16,
            128+16,
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
    })
    
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
    
    proj = nn.ModuleDict({
        "16": proj16,
        "8": proj8,
        "4": proj4,
        "2": proj2,
        "1": proj1,
    })
    
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
        gm_warp_dropout_p=gm_warp_dropout_p
    )
    
    h, w = resolutions[resolution]
    encoder = CNNandDinov2(
        cnn_kwargs=dict(
            pretrained=pretrained_backbone,
            amp=True
        ),
        amp=True,
        use_vgg=True,
    )
    
    matcher = RegressionMatcher(encoder, decoder, h=h, w=w, **kwargs)
    return matcher


def main(args):
    """Основная функция для запуска бенчмарка"""
    # Установка device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Загрузка модели
    print("Loading model...")
    h, w = resolutions[args.resolution]
    model = get_model(pretrained_backbone=True, resolution=args.resolution, attenuate_cert=False)
    
    if args.checkpoint:
        print(f"Loading checkpoint from: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        
        # Извлекаем state_dict из чекпоинта
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Удаляем префикс 'module.' если он есть (из DDP)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        print("Checkpoint loaded successfully")
    else:
        print("Warning: No checkpoint provided, using random initialization")
    
    model = model.to(device)
    model.eval()
    
    # Загрузка данных
    print(f"Loading validation data from: {args.val_data}")
    with open(args.val_data, "r") as f:
        val_scene_info = json.load(f)
    
    # Создание директории для визуализаций
    if args.vis_dir:
        os.makedirs(args.vis_dir, exist_ok=True)
        print(f"Visualizations will be saved to: {args.vis_dir}")
    
    # Создание бенчмарка
    benchmark = UmbraDenseBenchmark(
        val_scene_info,
        image_size=h,
        num_samples=args.num_samples,
        vis_dir=args.vis_dir
    )
    
    # Запуск бенчмарка
    print("\n" + "="*50)
    print("Starting benchmark...")
    print("="*50 + "\n")
    
    romatch.RANK = 0  # Устанавливаем для single-GPU режима
    results = benchmark.benchmark(model, batch_size=args.batch_size, visualize=args.visualize)
    
    # Сохранение результатов
    if args.output:
        print(f"\nSaving results to: {args.output}")
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
    
    print("\n" + "="*50)
    print("Benchmark completed!")
    print("="*50)


if __name__ == "__main__":
    parser = ArgumentParser(description="Umbra SAR Image Matching Benchmark")
    
    parser.add_argument("--val_data", type=str, required=True,
                       help="Путь к JSON файлу с валидационными парами")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Путь к чекпоинту модели (.pth)")
    parser.add_argument("--resolution", type=str, default="medium", 
                       choices=['low', 'medium', 'high'],
                       help="Разрешение для обработки изображений")
    parser.add_argument("--image_size", type=int, default=None,
                       help="Размер изображения (игнорируется, используется resolution)")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Количество пар для тестирования (по умолчанию все)")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Размер батча")
    parser.add_argument("--vis_dir", type=str, default=None,
                       help="Директория для сохранения визуализаций")
    parser.add_argument("--visualize", action="store_true",
                       help="Создавать визуализации соответствий")
    parser.add_argument("--output", type=str, default=None,
                       help="Путь для сохранения результатов в JSON")
    
    args = parser.parse_args()
    
    # Инициализация romatch глобальных переменных
    romatch.RANK = 0
    romatch.DEBUG_MODE = False
    
    main(args)


