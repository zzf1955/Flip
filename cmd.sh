conda activate dfar
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="/disk_n/zzf/.cache/huggingface"
# export CUDA_VISIBLE_DEVICES=3

# proxy
export http_proxy="http://127.0.0.1:20171"
export https_proxy="http://127.0.0.1:20171"
export HTTP_PROXY="http://127.0.0.1:20171"
export HTTPS_PROXY="http://127.0.0.1:20171"

export http_proxy=""
export https_proxy=""
export HTTP_PROXY=""
export HTTPS_PROXY=""

# leadtek2

export http_proxy=http://10.20.1.4:20171
export https_proxy=http://10.20.1.4:20171
export HTTP_PROXY=http://10.20.1.4:20171
export HTTPS_PROXY=http://10.20.1.4:20171
export no_proxy=localhost,127.0.0.1,10.20.1.0/24
export NO_PROXY=localhost,127.0.0.1,10.20.1.0/24
export PACKY_API_KEY="sk-Xg2c9fAFxJ2lXhhqeoKwondPCNycymwVgVB9E0r3aX0vtwxZ"

scripts/flip_run_2.sh train --cuda 0,2 --nproc 2 -- \
    --task-name identity_r2r_1s \
    --lora-rank 32 \
    --lora-target-modules q,k,v,o \
    --max-steps 1000 \
    --save-steps 100 \
    --eval-steps 100 \
    --eval-video-steps 100 \
    --train-size 1000 \
    --in-task-eval-size 16 \
    --ood-eval-size 16 \
    --in-task-video-size 4 \
    --ood-video-size 4 \
    --data-seed 42

scripts/flip_run_2.sh mitty_cache --cuda 0 -- \
  --pair-dir training_data/pair/identity_r2r/1s/Inspire_Put_Clothes_into_Washing_Machine\
  --output training_data/cache/vae/identity_r2r/1s/Inspire_Put_Clothes_into_Washing_Machine \
  --t5-cache-dir training_data/cache/t5/identity_r2r/1s \
  --device cuda:0 \
  --batch-size 4 \
  --prefetch-workers 8 \
  --prefetch-batches 2 \
  --save-workers 1


scripts/flip_run_2.sh mitty_cache --cuda 2 -- \
  --pair-dir training_data/pair/identity_r2r/1s/Inspire_Pickup_Pillow_MainCamOnly \
  --output training_data/cache/vae/identity_r2r/1s/Inspire_Pickup_Pillow_MainCamOnly \
  --t5-cache-dir training_data/cache/t5/identity_r2r/1s \
  --device cuda:0 \
  --batch-size 4 \
  --prefetch-workers 8 \
  --prefetch-batches 2 \
  --save-workers 1

scripts/flip_run_2.sh mitty_cache --cuda 2 -- \
  --pair-dir training_data/pair/h2r/1s/Inspire_Pickup_Pillow_MainCamOnly \
  --output training_data/cache/vae/h2r/1s/Inspire_Pickup_Pillow_MainCamOnly \
  --t5-cache-dir training_data/cache/t5/h2r/1s \
  --device cuda:0 \
  --batch-size 4 \
  --prefetch-workers 8 \
  --prefetch-batches 2 \
  --save-workers 1

  scripts/flip_run_2.sh mitty_cache --cuda 2 -- \
    --pair-dir training_data/pair/h2r/1s/Inspire_Put_Clothes_Into_Basket \
    --output training_data/cache/vae/h2r/1s/Inspire_Put_Clothes_Into_Basket \
    --t5-cache-dir training_data/cache/t5/h2r/1s \
    --device cuda:0 \
    --batch-size 4 \
    --prefetch-workers 8 \
    --prefetch-batches 2 \
    --save-workers 1

  scripts/flip_run_2.sh mitty_cache --cuda 2 -- \
    --pair-dir training_data/pair/h2r/1s/Inspire_Put_Clothes_into_Washing_Machine \
    --output training_data/cache/vae/h2r/1s/Inspire_Put_Clothes_into_Washing_Machine \
    --t5-cache-dir training_data/cache/t5/h2r/1s \
    --device cuda:0 \
    --batch-size 4 \
    --prefetch-workers 8 \
    --prefetch-batches 2 \
    --save-workers 1


  for task in \
    Inspire_Pickup_Pillow_MainCamOnly \
    Inspire_Put_Clothes_Into_Basket \
    Inspire_Put_Clothes_into_Washing_Machine
  do
    scripts/flip_run_2.sh mitty_cache --cuda 2 -- \
      --pair-dir training_data/pair/blur_r2r/1s/${task} \
      --output training_data/cache/vae/blur_r2r/1s/${task} \
      --t5-cache-dir training_data/cache/t5/blur_r2r/1s \
      --device cuda:0 \
      --batch-size 4 \
      --prefetch-workers 8 \
      --prefetch-batches 2 \
      --save-workers 1
  done



scripts/flip_run.sh train --cuda 0,2 --nproc 2 -- \
  --task-name identity_r2r_1s \
  --lora-rank 32 \
  --lora-target-modules q,k,v,o \
  --max-steps 1000 \
  --save-steps 100 \
  --eval-steps 100 \
  --eval-video-steps 100 \
  --train-size 1000 \
  --in-task-eval-size 16 \
  --ood-eval-size 16 \
  --in-task-video-size 4 \
  --ood-video-size 4 \
  --data-seed 42

# 恒等映射 leadtek 10.20.1.2

scripts/flip_run_2.sh train --cuda 0 -- \
  --task-name identity_r2r_1s \
  --lora-rank 64 \
  --train-size 10000 \
  --in-task-eval-size 16 \
  --ood-eval-size 16 \
  --in-task-video-size 8 \
  --ood-video-size 8 \
  --batch-size 4

scripts/flip_run_2.sh train --cuda 2 -- \
  --task-name identity_r2r_1s \
  --lora-rank 64 \
  --lora-target-modules q,k,v,o,ffn.0,ffn.2 \
  --batch-size 4 \
  --train-size 10000 \
  --in-task-eval-size 16 \
  --ood-eval-size 16 \
  --in-task-video-size 8 \
  --ood-video-size 8

scripts/flip_run_2.sh train --cuda 1 -- \
  --task-name identity_r2r_1s \
  --lora-rank 64 \
  --lora-target-modules ffn.0,ffn.2 \
  --batch-size 4 \
  --train-size 10000 \
  --in-task-eval-size 16 \
  --ood-eval-size 16 \
  --in-task-video-size 8 \
  --ood-video-size 8

# LoRA Ablation

scripts/flip_run_2.sh train --cuda 0 -- \
  --task-name identity_r2r_1s \
  --lora-rank 64 \
  --lora-attn-types self \
  --lora-attn-projections q,k,v,o \
  --batch-size 4 \
  --train-size 10000 \
  --in-task-eval-size 16 \
  --ood-eval-size 16 \
  --in-task-video-size 8 \
  --ood-video-size 8

scripts/flip_run_2.sh train --cuda 1 -- \
  --task-name identity_r2r_1s \
  --lora-rank 64 \
  --lora-attn-types self \
  --lora-attn-projections q,k \
  --batch-size 4 \
  --train-size 10000 \
  --in-task-eval-size 16 \
  --ood-eval-size 16 \
  --in-task-video-size 8 \
  --ood-video-size 8

scripts/flip_run_2.sh train --cuda 2 -- \
  --task-name identity_r2r_1s \
  --lora-rank 64 \
  --lora-attn-types self \
  --lora-attn-projections v,o \
  --batch-size 4 \
  --train-size 10000 \
  --in-task-eval-size 16 \
  --ood-eval-size 16 \
  --in-task-video-size 8 \
  --ood-video-size 8

# mitty

scripts/train_lora_grid.py \
  --cuda 2 \
  --task-name h2r_1s \
  --train-size 400 \
  --layouts self_qkv \
  --ranks 32 \
  --name-prefix baseline_mitty_h2r

scripts/train_lora_grid.py \
  --cuda 1 \
  --task-name h2r_1s \
  --train-size 400 \
  --layouts self_qkv_cross_qkv \
  --ranks 96 \
  --name-prefix baseline_mitty_h2r

# step2

scripts/train_lora_grid.py \
  --cuda 3 \
  --task-name blur_r2r_1s \
  --train-size 10000 \
  --merge-lora "/disk_n/zzf/flip/training_data/log/archive.5.2 high rank search/Mitty-identity_r2r_1s-10000d_r32_qkvoffn0ffn2_1000s_0428_195227/ckpt/step-1000.safetensors" \
  --layouts self_qkvo_ffn \
  --ranks 96 \
  --name-prefix step2_blur_r2r

scripts/train_lora_grid.py \
  --cuda 2 \
  --task-name blur_r2r_1s \
  --train-size 10000 \
  --merge-lora "/disk_n/zzf/flip/training_data/log/archive.5.2 high rank search/Mitty-identity_r2r_1s-10000d_r32_qkvoffn0ffn2_1000s_0428_195227/ckpt/step-1000.safetensors" \
  --layouts self_qkv \
  --ranks 96 \
  --name-prefix step2_blur_r2r

# step3

scripts/train_lora_grid.py \
  --cuda 3 \
  --task-name h2r_1s \
  --train-size 400 \
  --merge-lora \
    "/disk_n/zzf/flip/training_data/log/archive.5.2 high rank search/Mitty-identity_r2r_1s-10000d_r32_qkvoffn0ffn2_1000s_0428_195227/ckpt/step-1000.safetensors" \
    "/disk_n/zzf/flip/training_data/log/Mitty-h2r_1s-400d_r96_self_qkv_cross_qkv_1000s_0503_154803/ckpt/step-1000.safetensors" \
  --layouts self_qkvo_ffn \
  --ranks 64 \
  --name-prefix step3_h2r

# step1/2 merge

scripts/train_lora_grid.py \
  --cuda 3 \
  --task-name blur_r2r_1s \
  --train-size 10000 \
  --layouts self_qkvo_ffn \
  --ranks 96 \
  --name-prefix step12_blur_r2r

scripts/train_lora_grid.py \
  --cuda 2 \
  --task-name h2r_1s \
  --train-size 400 \
  --merge-lora "/disk_n/zzf/flip/training_data/log/Mitty-step12-blur_r2r_1s-10000d_r96_self_qkvo_ffn_1000s_0503_202508/ckpt/step-1000.safetensors" \
  --layouts self_qkvo_ffn \
  --ranks 32 \
  --name-prefix step3_h2r

cd /disk_n/zzf/flip

mkdir -p training_data/eval/h2r_step1000_16x16_logs

scripts/flip_run.sh eval_mitty --cuda 2 -- \
  --runs Mitty-h2r_1s-400d_r32_self_qkvo_ffn_1000s_0503_201038 \
  --checkpoint step-1000.safetensors \
  --splits eval ood_eval \
  --cache-root training_data/eval/h2r_step1000_16x16_flat/cache \
  --pair-root training_data/eval/h2r_step1000_16x16_flat/pair \
  --t5-cache-dir training_data/cache/t5/h2r/1s \
  --output-dir training_data/eval/h2r_step1000_16x16_qkvo_ffn \
  --samples-per-split 16 \
  --device cuda:0 \
  > training_data/eval/h2r_step1000_16x16_logs/3step_qkvo_ffn.log 2>&1 &

PID_QKVO_FFN=$!

scripts/flip_run.sh eval_mitty --cuda 0 -- \
  --runs Mitty-h2r_1s-400d_r32_self_qkv_1000s_0503_202021 \
  --checkpoint step-1000.safetensors \
  --splits eval ood_eval \
  --cache-root training_data/eval/h2r_step1000_16x16_flat/cache \
  --pair-root training_data/eval/h2r_step1000_16x16_flat/pair \
  --t5-cache-dir training_data/cache/t5/h2r/1s \
  --output-dir training_data/eval/h2r_step1000_16x16_qkv \
  --samples-per-split 16 \
  --device cuda:0 \
  > training_data/eval/h2r_step1000_16x16_logs/mitty.log 2>&1 &

PID_QKV=$!

wait "$PID_QKVO_FFN"
wait "$PID_QKV"