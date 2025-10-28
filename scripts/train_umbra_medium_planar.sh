OUT_DIR=workspace/20251027_medium_planar
EXP_SCRIPT=experiments/train_roma_umbra.py
mkdir -p $OUT_DIR
cp $EXP_SCRIPT $OUT_DIR
cp ${BASH_SOURCE[0]} $OUT_DIR

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29600 $EXP_SCRIPT \
    --train_data /home/sema/radar/datasets/umbra_tiles/560_cleanup/pairs_train.json \
    --val_data /home/sema/radar/datasets/umbra_tiles/560_cleanup/pairs_val.json \
    --checkpoint_dir $OUT_DIR \
    --train_resolution medium \
    --gpu_batch_size 8 \
    --total_steps 100000 \
    --checkpoint_every 100 \
    --dont_log_wandb \
    --use_planar_decoder \
    --planar_mode \
    --flush