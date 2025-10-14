OUT_DIR=workspace/20251012_medium_planar
EXP_SCRIPT=experiments/train_roma_umbra.py
mkdir -p $OUT_DIR
cp $EXP_SCRIPT $OUT_DIR
cp $0 $OUT_DIR

CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=29601 $EXP_SCRIPT \
    --train_data /home/sema/radar/datasets/umbra_tiles/560/pairs_train.json \
    --val_data /home/sema/radar/datasets/umbra_tiles/560/pairs_val.json \
    --checkpoint_dir $OUT_DIR \
    --train_resolution medium \
    --gpu_batch_size 8 \
    --total_steps 100000 \
    --checkpoint_every 100 \
    --dont_log_wandb \
    --use_planar_decoder
