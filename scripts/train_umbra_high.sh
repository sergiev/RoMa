CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 experiments/train_roma_umbra.py \
    --train_data /home/sema/radar/datasets/umbra_tiles/1024/pairs_train.json \
    --val_data /home/sema/radar/datasets/umbra_tiles/1024/pairs_val.json \
    --checkpoint_dir workspace/20251012_high_downscale \
    --train_resolution high \
    --gpu_batch_size 8 \
    --total_steps 100000 \
    --checkpoint_every 100 \
    --dont_log_wandb
