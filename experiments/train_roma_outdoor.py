import os
import torch
from argparse import ArgumentParser

from torch import nn
from torch.utils.data import ConcatDataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import json
from torch.utils.tensorboard import SummaryWriter

from romatch.benchmarks import MegadepthDenseBenchmark
from romatch.datasets.custom_dataset import CustomBuilder  # <-- 1. ИМПОРТ
from romatch.losses.robust_loss import RobustLosses
from romatch.benchmarks import MegaDepthPoseEstimationBenchmark
from romatch.benchmarks.custom_benchmark import UmbraBenchmark # <-- ИМПОРТ НОВОГО БЕНЧМАРКА

from romatch.train.train import train_k_steps
from romatch.models.matcher import *
from romatch.models.transformer import Block, TransformerDecoder, MemEffAttention
from romatch.models.encoders import *
from romatch.checkpointing import CheckPoint

resolutions = {"low":(448, 448), "medium":(14*8*5, 14*8*5), "high":(14*8*6, 14*8*6), "mega": (14*8*9, 14*8*9)}

def get_model(pretrained_backbone=True, resolution = "medium", **kwargs):
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
        amp = True,
        pos_enc = False,)
    dw = True
    hidden_blocks = 8
    kernel_size = 5
    displacement_emb = "linear"
    disable_local_corr_grad = True
    
    conv_refiner = nn.ModuleDict(
        {
            "16": ConvRefiner(
                2 * 512+128+(2*7+1)**2,
                2 * 512+128+(2*7+1)**2,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=128,
                local_corr_radius = 7,
                corr_in_other = True,
                amp = True,
                disable_local_corr_grad = disable_local_corr_grad,
                bn_momentum = 0.01,
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
                local_corr_radius = 3,
                corr_in_other = True,
                amp = True,
                disable_local_corr_grad = disable_local_corr_grad,
                bn_momentum = 0.01,
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
                local_corr_radius = 2,
                corr_in_other = True,
                amp = True,
                disable_local_corr_grad = disable_local_corr_grad,
                bn_momentum = 0.01,
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
                amp = True,
                disable_local_corr_grad = disable_local_corr_grad,
                bn_momentum = 0.01,
            ),
            "1": ConvRefiner(
                2 * 9 + 6,
                24,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks = hidden_blocks,
                displacement_emb = displacement_emb,
                displacement_emb_dim = 6,
                amp = True,
                disable_local_corr_grad = disable_local_corr_grad,
                bn_momentum = 0.01,
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
    proj = nn.ModuleDict({
        "16": proj16,
        "8": proj8,
        "4": proj4,
        "2": proj2,
        "1": proj1,
        })
    displacement_dropout_p = 0.0
    gm_warp_dropout_p = 0.0
    decoder = Decoder(coordinate_decoder, 
                      gps, 
                      proj, 
                      conv_refiner, 
                      detach=True, 
                      scales=["16", "8", "4", "2", "1"], 
                      displacement_dropout_p = displacement_dropout_p,
                      gm_warp_dropout_p = gm_warp_dropout_p)
    h,w = resolutions[resolution]
    encoder = CNNandDinov2(
        cnn_kwargs = dict(
            pretrained=pretrained_backbone,
            amp = True),
        amp = True,
        use_vgg = True,
        in_chans = 1, # <--- Указываем 1 входной канал
    )
    matcher = RegressionMatcher(encoder, decoder, h=h, w=w,**kwargs)
    return matcher

def train(args):
    dist.init_process_group('nccl')
    #torch._dynamo.config.verbose=True
    gpus = int(os.environ['WORLD_SIZE'])
    # create model and move it to GPU with id rank
    rank = dist.get_rank()
    print(f"Start running DDP on rank {rank}")
    device_id = rank % torch.cuda.device_count()
    romatch.LOCAL_RANK = device_id
    torch.cuda.set_device(device_id)
    
    resolution = args.train_resolution
    experiment_name = os.path.splitext(os.path.basename(__file__))[0]
    if rank == 0:
        writer = SummaryWriter(f"workspace/runs/{experiment_name}")

    checkpoint_dir = "workspace/checkpoints/"
    h,w = resolutions[resolution]
    model = get_model(pretrained_backbone=True, resolution=resolution, attenuate_cert = False).to(device_id)
    # Num steps
    global_step = 0
    batch_size = args.gpu_batch_size
    step_size = gpus*batch_size
    romatch.STEP_SIZE = step_size
    
    N = (batch_size * 25000)  # 250k steps of batch size
    # checkpoint every
    k = 1000 // romatch.STEP_SIZE


    depth_interpolation_mode = "bilinear"

    train_json_path = "/home/sema/radar/datasets/umbra_tiles/pairs_train.json"
    train_builder = CustomBuilder(data_root=train_json_path)

    # Убедитесь, что image_size соответствует разрешению модели (h, w)
    train_scenes = train_builder.build_scenes(image_size=h)
    
    train_dataset = ConcatDataset(train_scenes)
    dataset_weights = train_builder.weight_scenes(train_dataset, alpha=0.75)

    # Loss and optimizer
    depth_loss = RobustLosses(
        ce_weight=0.01, 
        local_dist={1:4, 2:4, 4:8, 8:8},
        local_largest_scale=8,
        depth_interpolation_mode=depth_interpolation_mode,
        alpha = 0.5,
        c = 1e-4,)
    parameters = [
        {"params": model.encoder.parameters(), "lr": romatch.STEP_SIZE * 5e-6 / 8},
        {"params": model.decoder.parameters(), "lr": romatch.STEP_SIZE * 1e-4 / 8},
    ]
    optimizer = torch.optim.AdamW(parameters, weight_decay=0.01)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[(9*N/romatch.STEP_SIZE)//10])

    val_json_path = "/home/sema/radar/datasets/umbra_tiles/pairs_val.json" 
    umbra_benchmark = UmbraBenchmark(val_json_path, num_samples=1000, h=h, w=w)
    
    checkpointer = CheckPoint(checkpoint_dir, experiment_name)
    model, optimizer, lr_scheduler, global_step = checkpointer.load(model, optimizer, lr_scheduler, global_step)
    romatch.GLOBAL_STEP = global_step
    ddp_model = DDP(model, device_ids=[device_id], find_unused_parameters = False, gradient_as_bucket_view=True)
    grad_scaler = torch.cuda.amp.GradScaler(growth_interval=1_000_000)
    grad_clip_norm = 0.01
    for n in range(romatch.GLOBAL_STEP, N, k * romatch.STEP_SIZE):
        custom_sampler = torch.utils.data.WeightedRandomSampler(
            dataset_weights, num_samples = batch_size * k, replacement=False
        )
        dataloader = iter(
            torch.utils.data.DataLoader(
                train_dataset,
                batch_size = batch_size,
                sampler = custom_sampler,
                num_workers = 8,
            )
        )
        
        training_metrics = train_k_steps(
            n, k, dataloader, ddp_model, depth_loss, optimizer, lr_scheduler, grad_scaler, grad_clip_norm = grad_clip_norm,
        )
        checkpointer.save(model, optimizer, lr_scheduler, romatch.GLOBAL_STEP)
        
        if rank == 0:
            # Логирование тренировочных метрик
            for i, metrics in enumerate(training_metrics):
                step = n + i * romatch.STEP_SIZE
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(f"train/{key}", value, step)

            # Логирование валидационных метрик
            benchmark_results = umbra_benchmark.benchmark(model)
            print(f"Logging to TensorBoard: {benchmark_results}")
            for key, value in benchmark_results.items():
                writer.add_scalar(f"val/{key}", value, romatch.GLOBAL_STEP)


if __name__ == "__main__":
    warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
    warnings.filterwarnings('ignore', category=UserWarning, message=r'.*WARNING batched routines are designed for small sizes.*')
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1" # For BF16 computations
    os.environ["OMP_NUM_THREADS"] = "16"
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    import romatch
    parser = ArgumentParser()
    parser.add_argument("--only_test", action='store_true')
    parser.add_argument("--debug_mode", action='store_true')
    parser.add_argument("--train_resolution", default='mega')
    parser.add_argument("--gpu_batch_size", default=4, type=int)

    args, _ = parser.parse_known_args()
    romatch.DEBUG_MODE = args.debug_mode
    if not args.only_test:
        train(args)
