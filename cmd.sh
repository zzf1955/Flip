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


scripts/flip_run.sh train --cuda 0,2 --nproc 2 -- \
  --task-name R2H \
  --loss uniform \
  --cache-train training_data/cache/vae/pair_1s_r2h/train \
  --cache-eval training_data/cache/vae/pair_1s_r2h/eval \
  --cache-ood training_data/cache/vae/pair_1s_r2h/ood_eval \
  --t5-cache-dir training_data/cache/t5 \
  --merge-lora training_data/log/identity-mitty-bs8-5k/ckpt/step-4850.safetensors \
  --merge-lora-rank 96 \
  --lora-rank 128 \
  --lora-target-modules q,k,v,o \
  --batch-size 4 \
  --max-steps 500 \
  --save-steps 100 \
  --eval-steps 100 \
  --eval-video-steps 100 \
  --eval-video-samples-in-task 4 \
  --eval-video-samples-ood 4 \
  --num-inference-steps 30 \
  --wandb-tags attn_lora r2h r128 bs8 train3_518 identity_mitty_bs8_5k_merge

 LD_PRELOAD=/disk_n/zzf_tmp/conda_envs/flip/lib/libjpeg.so.8 \
  CUDA_VISIBLE_DEVICES=0,2 \
  HF_HOME=/disk_n/zzf/.cache/huggingface \
  TORCH_HOME=/disk_n/zzf/.cache/torch \
  PIP_CACHE_DIR=/disk_n/zzf/.pip_cache \
  FFMPEG_BIN=/disk_n/zzf_tmp/conda_envs/flip/bin/ffmpeg \
  SSL_CERT_FILE=/disk_n/zzf_tmp/conda_envs/flip/ssl/cert.pem \
  REQUESTS_CA_BUNDLE=/disk_n/zzf_tmp/conda_envs/flip/ssl/cert.pem \
  CURL_CA_BUNDLE=/disk_n/zzf_tmp/conda_envs/flip/ssl/cert.pem \
  WANDB_X_DISABLE_SERVICE=true \
  WANDB_CORE=disabled \
  /disk_n/zzf_tmp/conda_envs/flip/bin/python -m torch.distributed.run \
    --standalone \
    --nproc_per_node=2 \
    -m src.pipeline.train \
    --task-name transfer \
    --loss uniform \
    --cache-train /disk_n/zzf/flip/training_data/cache/vae/pair_1s_train3/train \
    --cache-eval /disk_n/zzf/flip/training_data/cache/vae/pair_1s_train3/eval \
    --cache-ood /disk_n/zzf/flip/training_data/cache/vae/pair_1s_train3/ood_eval \
    --t5-cache-dir /disk_n/zzf/flip/training_data/cache/t5 \
    --output-dir /disk_n/zzf/flip/training_data/log \
    --lora-rank 32 \
    --lora-target-modules q,k,v,o \
    --lr 1e-4 \
    --lr-min 1e-6 \
    --warmup-steps 50 \
    --weight-decay 0.01 \
    --max-steps 500 \
    --batch-size 4 \
    --save-steps 100 \
    --eval-steps 100 \
    --eval-t-samples 5 \
    --eval-video-steps 100 \
    --eval-video-samples-in-task 4 \
    --eval-video-samples-ood 4 \
    --max-eval-files 0 \
    --num-inference-steps 30 \
    --wandb-project Flip \
    --wandb-tags attn_lora transfer r32 cuda02 bs8 eval100 video100 train3_518 16eval 32ood max500 no_merge_lora


scripts/flip_run_2.sh train --cuda 0,2 --nproc 2 -- \
  --task-name R2H \
  --loss uniform \
  --cache-train training_data/cache/vae/pair_1s_r2h/train \
  --cache-eval training_data/cache/vae/pair_1s_r2h/eval \
  --cache-ood training_data/cache/vae/pair_1s_r2h/ood_eval \
  --t5-cache-dir training_data/cache/t5 \
  --lora-rank 128 \
  --lora-target-modules q,k,v,o \
  --batch-size 4 \
  --max-steps 500 \
  --save-steps 100 \
  --eval-steps 100 \
  --eval-video-steps 100 \
  --eval-video-samples-in-task 4 \
  --eval-video-samples-ood 4 \
  --num-inference-steps 30 \
  --wandb-tags attn_lora r2h r128 bs8 train3_518 no_merge_lora


scripts/flip_run_2.sh train --cuda 0,2 --nproc 2 -- \
--task-name identity \
--loss uniform \
--cache-train output/mitty_cache_1s/train \
--cache-eval  output/mitty_cache_1s/eval \
--cache-ood   output/mitty_cache_1s/ood_eval \
--t5-cache-dir training_data/cache/t5 \
--output-dir training_data/log \
--init-lora training_data/log/identity-mitty-bs8-5k/ckpt/step-4850.safetensors \
--lora-rank 16 \
--lora-target-modules q,k,v,o \
--batch-size 8 \
--max-steps 2000 \
--save-steps 100 \
--eval-steps 100 \
--eval-t-samples 5 \
--eval-video-steps 100 \
--eval-video-samples-in-task 4 \
--eval-video-samples-ood 2 \
--num-inference-steps 30 \
--lr 1e-4 \
--lr-min 1e-6 \
--warmup-steps 50 \
--weight-decay 0.01 \
--wandb-project flip \
--wandb-tags identity attn_lora r16 bs8 from_identity_mitty_bs8_5k