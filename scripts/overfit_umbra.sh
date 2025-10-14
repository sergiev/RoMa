CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 experiments/train_roma_umbra.py \
    --train_data /home/sema/radar/datasets/umbra_tiles/1024/pairs_val.json \
    --val_data /home/sema/radar/datasets/umbra_tiles/1024/pairs_val.json \
    --checkpoint_dir workspace/umbra_overfit_20251011_onlyval \
    --train_resolution mega \
    --gpu_batch_size 4 \
    --total_steps 10000 \
    --checkpoint_every 100 \
    --dont_log_wandb
