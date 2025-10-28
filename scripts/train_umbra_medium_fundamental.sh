OUT_DIR=workspace/20251027_medium_fundamental
EXP_SCRIPT=experiments/train_roma_umbra.py
mkdir -p $OUT_DIR
cp $EXP_SCRIPT $OUT_DIR
cp ${BASH_SOURCE[0]} $OUT_DIR

CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 --master_port=29602 $EXP_SCRIPT \
    --train_data /home/sema/radar/datasets/umbra_tiles/560_cleanup/pairs_train.json \
    --val_data /home/sema/radar/datasets/umbra_tiles/560_cleanup/pairs_val.json \
    --checkpoint_dir $OUT_DIR \
    --train_resolution medium \
    --gpu_batch_size 8 \
    --total_steps 100000 \
    --checkpoint_every 100 \
    --dont_log_wandb \
    --flush
