conda activate dfar
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="/disk_n/zzf/.cache/huggingface"
# export CUDA_VISIBLE_DEVICES=3

# proxy
export http_proxy="http://127.0.0.1:20171"
export https_proxy="http://127.0.0.1:20171"
export HTTP_PROXY="http://127.0.0.1:20171"
export HTTPS_PROXY="http://127.0.0.1:20171"

# leadtek2

export http_proxy=http://10.20.1.4:20171
export https_proxy=http://10.20.1.4:20171
export HTTP_PROXY=http://10.20.1.4:20171
export HTTPS_PROXY=http://10.20.1.4:20171
export no_proxy=localhost,127.0.0.1,10.20.1.0/24
export NO_PROXY=localhost,127.0.0.1,10.20.1.0/24
export PACKY_API_KEY="sk-Xg2c9fAFxJ2lXhhqeoKwondPCNycymwVgVB9E0r3aX0vtwxZ"

LD_PRELOAD=/home/leadtek/miniconda3/envs/flip/lib/libjpeg.so.8 \
    /home/leadtek/miniconda3/envs/flip/bin/python -m src.pipeline.robot_patch \
    --task inspire --degrade blur --blur-ksize 41 --patch-expand 3 \
    --max-segments 250 --workers 50 --clean

  # train split (900 samples)
LD_PRELOAD=/home/leadtek/miniconda3/envs/flip/lib/libjpeg.so.8 \
    python -m src.pipeline.mitty_cache \
        --pair-dir training_data/pair/1s_patch/train \
        --output   training_data/cache/1s_patch/train \
        --device cuda:0 --no-frames

  # eval split (100 samples)
LD_PRELOAD=/home/leadtek/miniconda3/envs/flip/lib/libjpeg.so.8 \
    python -m src.pipeline.mitty_cache \
        --pair-dir training_data/pair/1s_patch/eval \
        --output   training_data/cache/1s_patch/eval \
        --device cuda:2 --no-frames

LD_PRELOAD=/home/leadtek/miniconda3/envs/flip/lib/libjpeg.so.8 \
    CUDA_VISIBLE_DEVICES=2,3 \
    torchrun --nproc_per_node=2 -m src.pipeline.train_mitty \
    --merge-lora training_data/log/identity-mitty-bs8-2k/ckpt/step-2000.safetensors \
    --merge-lora-rank 96 \
    --lora-target-modules "ffn.0,ffn.2" \
    --lora-rank 128 \
    --cache-train training_data/cache/1s_patch/train \
    --cache-eval  training_data/cache/1s_patch/eval \
    --patch-dir   training_data/pair/1s_patch/train/patch \
    --max-steps 2000 --save-steps 100 --eval-steps 100 \
    --eval-video-steps 200 --eval-video-samples-in-task 4 \
    --lr 1e-4 --warmup-steps 50 \
    --wandb-tags ffn_lora appearance\
    --task-name appearance


LD_PRELOAD=/home/leadtek/miniconda3/envs/flip/lib/libjpeg.so.8 \
  CUDA_VISIBLE_DEVICES=2,3 \
  torchrun --nproc_per_node=2 -m src.pipeline.train_mitty \
    --merge-lora training_data/log/identity-mitty-bs8-2k/ckpt/step-2000.safetensors \
    --merge-lora-rank 96 \
    --lora-target-modules "ffn.0,ffn.2" \
    --lora-rank 192 \
    --cache-train training_data/cache/1s_patch_sam2/train \
    --cache-eval  training_data/cache/1s_patch_sam2/eval \
    --patch-dir   training_data/pair/1s_patch_sam2/train/patch \
    --max-steps 2000 --save-steps 100 --eval-steps 100 \
    --eval-video-steps 200 --eval-video-samples-in-task 4 \
    --lr 1e-4 --warmup-steps 50 \
    --wandb-tags ffn_lora appearance sam2 \
    --task-name appearance