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
  --cuda 3 \
  --task-name h2r_1s \
  --train-size 50 \
  --layouts self_qkv \
  --ranks 96 \
  --name-prefix baseline_mitty_h2r

# step 1

scripts/flip_run.sh train --cuda 2 --nproc 1 -- \
  --task-name identity_r2r_1s \
  --train-tasks Inspire_Put_Clothes_into_Washing_Machine,Inspire_Put_Clothes_Into_Basket \
  --ood-tasks Inspire_Pickup_Pillow_MainCamOnly \
  --run-prefix Final_identity_r2r_1s_r96_self_qkvo_ffn \
  --train-lora-rank 96 \
  --train-lora-target-modules self_attn.q,self_attn.k,self_attn.v,self_attn.o,ffn.0,ffn.2 \
  --train-size 0 \
  --in-task-eval-size 16 \
  --ood-eval-size 16 \
  --in-task-video-size 4 \
  --ood-video-size 2 \
  --max-steps 1000 \
  --save-steps 100 \
  --eval-steps 100 \
  --eval-video-steps 100 \
  --batch-size 4 \
  --lr 1e-4 \
  --lr-min 1e-6 \
  --warmup-steps 50 \
  --weight-decay 0.01 \
  --wandb-project Flip \
  --wandb-tags single_lora stage:1 task:identity_r2r layout:self_qkvo_ffn rank:96

# step2

scripts/train_lora_grid.py \
  --cuda 3 \
  --task-name blur_r2r_1s \
  --train-size 10000 \
  --merge-lora "/disk_n/zzf/flip/training_data/log/archive.5.2 high rank search/Mitty-identity_r2r_1s-10000d_r32_qkvoffn0ffn2_1000s_0428_195227/ckpt/step-1000.safetensors" \
  --layouts self_qkvo_ffn \
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
  --ranks 96 \
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

#

scripts/flip_run.sh train --cuda 2 --nproc 1 -- \
  --task-name identity_r2r_1s \
  --train-tasks Inspire_Put_Clothes_into_Washing_Machine,Inspire_Put_Clothes_Into_Basket \
  --ood-tasks Inspire_Pickup_Pillow_MainCamOnly \
  --run-prefix single_lora_s1_identity_r2r \
  --train-lora-rank 96 \
  --train-lora-target-modules self_attn.q,self_attn.k,self_attn.v,self_attn.o,ffn.0,ffn.2 \
  --train-size 0 \
  --in-task-eval-size 16 \
  --ood-eval-size 16 \
  --in-task-video-size 4 \
  --ood-video-size 2 \
  --max-steps 1000 \
  --save-steps 100 \
  --eval-steps 100 \
  --eval-video-steps 100 \
  --batch-size 4 \
  --lr 1e-4 \
  --lr-min 1e-6 \
  --warmup-steps 50 \
  --weight-decay 0.01 \
  --wandb-project Flip \
  --wandb-tags single_lora stage:1 task:identity_r2r layout:self_qkvo_ffn rank:96

scripts/flip_run.sh train --cuda 3 --nproc 1 -- \
  --task-name blur_r2r_1s \
  --run-prefix single_lora_s2_blur_r2r \
  --train-lora /disk_n/zzf/flip/training_data/log/single_lora_s1_identity_r2r-identity_r2r_1s-22592d_r96_self_qkvo_ffn_1000s_0506_170245/ckpt/step-1000.safetensors \
  --train-size 0 \
  --in-task-eval-size 16 \
  --ood-eval-size 16 \
  --in-task-video-size 4 \
  --ood-video-size 2 \
  --max-steps 1000 \
  --save-steps 100 \
  --eval-steps 100 \
  --eval-video-steps 100 \
  --batch-size 4 \
  --lr 1e-4 \
  --lr-min 1e-6 \
  --warmup-steps 50 \
  --weight-decay 0.01 \
  --wandb-project Flip \
  --wandb-tags single_lora stage:2 task:blur_r2r layout:self_qkvo_ffn rank:96

scripts/flip_run.sh train --cuda 0 --nproc 1 -- \
  --task-name h2r_1s \
  --run-prefix single_lora_s3_h2r \
  --train-lora /disk_n/zzf/flip/training_data/log/single_lora_s2_blur_r2r-blur_r2r_1s-17944d_r96_self_qkvo_ffn_1000s_0506_185729/ckpt/step-0500.safetensors \
  --train-size 400 \
  --in-task-eval-size 16 \
  --ood-eval-size 16 \
  --in-task-video-size 4 \
  --ood-video-size 2 \
  --max-steps 1000 \
  --save-steps 100 \
  --eval-steps 100 \
  --eval-video-steps 100 \
  --batch-size 4 \
  --lr 1e-4 \
  --lr-min 1e-6 \
  --warmup-steps 50 \
  --weight-decay 0.01 \
  --wandb-project Flip \
  --wandb-tags single_lora stage:3 task:h2r layout:self_qkvo_ffn rank:96


scripts/flip_run_2.sh train --cuda 0 --nproc 1 -- \
  --task-name h2r_1s \
  --train-tasks Inspire_Collect_Clothes_MainCamOnly,Inspire_Put_Clothes_into_Washing_Machine \
  --ood-tasks Inspire_Pickup_Pillow_MainCamOnly \
  --run-prefix test_leadtek3 \
  --lora-rank 96 \
  --lora-target-modules self_attn.q,self_attn.k,self_attn.v \
  --train-size 400 \
  --in-task-eval-size 16 \
  --ood-eval-size 16 \
  --in-task-video-size 4 \
  --ood-video-size 2 \
  --data-seed 42 \
  --max-steps 1000 \
  --save-steps 100 \
  --eval-steps 100 \
  --eval-video-steps 100 \
  --batch-size 4 \
  --lr 1e-4 \
  --lr-min 1e-6 \
  --warmup-steps 50 \
  --weight-decay 0.01 \
  --wandb-project Flip \
  --wandb-tags final mitty h2r layout:self_qkv rank:96
  

scripts/flip_run.sh train --cuda 0 --nproc 1 -- \
  --task-name h2r_1s \
  --train-tasks Inspire_Collect_Clothes_MainCamOnly,Inspire_Put_Clothes_into_Washing_Machine \
  --ood-tasks Inspire_Pickup_Pillow_MainCamOnly \
  --run-prefix final_ours_step3_continue_stack_r128_0507 \
  --merge-lora "/disk_n/zzf/flip/training_data/log/final_ours_step1_0507_004839-identity_r2r_1s-22592d_r32_self_qkvo_ffn_1000s_0507_004911/ckpt/step-1000.safetensors" \
  --merge-lora "/disk_n/zzf/flip/training_data/log/final_ours_step2_0507_004839-blur_r2r_1s-17944d_r256_self_qkvo_ffn_1000s_0507_030557/ckpt/step-1000.safetensors" \
  --lora-rank 128 \
  --lora-target-modules self_attn.q,self_attn.k,self_attn.v,self_attn.o,ffn.0,ffn.2 \
  --train-size 400 \
  --in-task-eval-size 16 \
  --ood-eval-size 16 \
  --in-task-video-size 4 \
  --ood-video-size 2 \
  --data-seed 42 \
  --max-steps 1000 \
  --save-steps 100 \
  --eval-steps 100 \
  --eval-video-steps 100 \
  --batch-size 4 \
  --lr 1e-4 \
  --lr-min 1e-6 \
  --warmup-steps 50 \
  --weight-decay 0.01 \
  --wandb-project Flip \
  --wandb-tags final ours step3 h2r continue_stack merge_step1_step2 new_lora_r128 layout:self_qkvo_ffn rank:128

# run r2h

scripts/flip_run_2.sh train --cuda 2 --nproc 1 -- \
  --task-name r2h_1s \
  --train-tasks Inspire_Collect_Clothes_MainCamOnly,Inspire_Pickup_Pillow_MainCamOnly,Inspire_Put_Clothes_into_Washing_Machine \
  --ood-tasks "," \
  --run-prefix final_data_r2h_all_in_task \
  --lora-rank 96 \
  --lora-target-modules self_attn.q,self_attn.k,self_attn.v,self_attn.o,ffn.0,ffn.2 \
  --train-size 0 \
  --in-task-eval-size 59 \
  --ood-eval-size 0 \
  --in-task-video-size 4 \
  --ood-video-size 0 \
  --data-seed 42 \
  --batch-size 4 \
  --max-steps 1000 \
  --save-steps 100 \
  --eval-steps 100 \
  --eval-video-steps 100 \
  --lr 1e-4 \
  --lr-min 1e-6 \
  --warmup-steps 50 \
  --weight-decay 0.01 \
  --wandb-project Flip \
  --wandb-tags final_data_r2h all_in_task split_9_1 layout:self_qkvo_ffn rank:96



# mixed h2r
scripts/flip_run_2.sh train_mitty_mixed_h2r --cuda 0 --nproc 1 -- \
    --task-name mixed_h2r_1s_400orig_400syn \
    --original-train-tasks Inspire_Collect_Clothes_MainCamOnly,Inspire_Put_Clothes_into_Washing_Machine \
    --syn-train-tasks Inspire_Collect_Clothes_MainCamOnly_syn,Inspire_Put_Clothes_into_Washing_Machine_syn \
    --ood-eval-tasks Inspire_Pickup_Pillow_MainCamOnly \
    --run-prefix final_mitty_mixed_h2r_r96_0508 \
    --lora-rank 96 \
    --lora-target-modules self_attn.q,self_attn.k,self_attn.v \
    --original-train-size 400 \
    --syn-train-size 400 \
    --in-task-eval-size 16 \
    --ood-eval-size 16 \
    --eval-video-samples-in-task 4 \
    --eval-video-samples-ood 2 \
    --data-seed 42 \
    --t5-cache-dir training_data/cache/t5/h2r/1s \
    --max-steps 1000 \
    --save-steps 100 \
    --eval-steps 100 \
    --eval-video-steps 100 \
    --batch-size 4 \
    --lr 1e-4 \
    --lr-min 1e-6 \
    --warmup-steps 50 \
    --weight-decay 0.01 \
    --wandb-project Flip \
    --wandb-tags final mitty h2r mixed layout:self_qkv rank:96 orig:400 syn:400


LD_PRELOAD=/home/leadtek/miniconda3/envs/flip/lib/libjpeg.so.8 \
no_proxy=localhost,127.0.0.1 \
CUDA_VISIBLE_DEVICES=0 \
/home/leadtek/miniconda3/envs/flip/bin/python -m src.pipeline.r2h_synthesize \
  --source-task Inspire_Collect_Clothes_MainCamOnly,Inspire_Put_Clothes_into_Washing_Machine \
  --duration 1s \
  --run final_data_r2h_all_in_task-r2h_1s-529d_r96_self_qkvo_ffn_1000s_0507_124705 \
  --checkpoint step-1000.safetensors \
  --num-samples 4000 \
  --allocate-by-task proportional \
  --device cuda:0 \
  --resume-existing

scripts/eval_final_step1000_missing.py \
--runner flip_run_2 \
--cuda-list 0,1,2,3 \
--execute


scripts/flip_run_2.sh train_mitty_mixed_h2r --cuda 2 --nproc 1 -- \
    --task-name mixed_h2r_1s_400orig_2800syn \
    --original-train-tasks Inspire_Collect_Clothes_MainCamOnly,Inspire_Put_Clothes_into_Washing_Machine \
    --syn-train-tasks Inspire_Collect_Clothes_MainCamOnly_syn,Inspire_Put_Clothes_into_Washing_Machine_syn \
    --ood-eval-tasks Inspire_Pickup_Pillow_MainCamOnly \
    --run-prefix final_ours_mixed_h2r_r96_0508 \
    --merge-lora training_data/log/final_ours_step1_0507_004839-identity_r2r_1s-22592d_r32_self_qkvo_ffn_1000s_0507_004911/ckpt/step-1000.safetensors \
    --merge-lora training_data/log/final_ours_step2_0507_004839-blur_r2r_1s-17944d_r256_self_qkvo_ffn_1000s_0507_030557/ckpt/step-1000.safetensors \
    --lora-rank 96 \
    --lora-target-modules self_attn.q,self_attn.k,self_attn.v,self_attn.o,ffn.0,ffn.2 \
    --original-train-size 400 \
    --syn-train-size 2800 \
    --in-task-eval-size 16 \
    --ood-eval-size 16 \
    --eval-video-samples-in-task 4 \
    --eval-video-samples-ood 2 \
    --data-seed 42 \
    --t5-cache-dir training_data/cache/t5/h2r/1s \
    --max-steps 1000 \
    --save-steps 100 \
    --eval-steps 100 \
    --eval-video-steps 100 \
    --batch-size 4 \
    --lr 1e-4 \
    --lr-min 1e-6 \
    --warmup-steps 50 \
    --weight-decay 0.01 \
    --wandb-project Flip \
    --wandb-tags final ours h2r mixed merge_step1_step2 layout:self_qkvo_ffn rank:96 orig:400 syn:2800
  

scripts/flip_run_2.sh mitty_cache --cuda 0 -- \
  --pair-dir training_data/pair/h2r/1s/Inspire_Collect_Clothes_MainCamOnly_syn \
  --output training_data/cache/vae/h2r/1s/Inspire_Collect_Clothes_MainCamOnly_syn \
  --t5-cache-dir training_data/cache/t5/h2r/1s \
  --device cuda:0 \
  --resume \
  --batch-size 4 \
  --prefetch-workers 8 \
  --prefetch-batches 2 \
  --save-workers 1

scripts/flip_run_2.sh mitty_cache --cuda 1 -- \
  --pair-dir training_data/pair/h2r/1s/Inspire_Put_Clothes_into_Washing_Machine_syn \
  --output training_data/cache/vae/h2r/1s/Inspire_Put_Clothes_into_Washing_Machine_syn \
  --t5-cache-dir training_data/cache/t5/h2r/1s \
  --device cuda:0 \
  --resume \
  --batch-size 4 \
  --prefetch-workers 8 \
  --prefetch-batches 2 \
  --save-workers 1

#ablation study

scripts/flip_run_2.sh train_mitty_mixed_h2r --cuda 2 --nproc 1 -- \
    --task-name mixed_h2r_1s_400orig_0syn \
    --original-train-tasks Inspire_Collect_Clothes_MainCamOnly,Inspire_Put_Clothes_into_Washing_Machine \
    --syn-train-tasks Inspire_Collect_Clothes_MainCamOnly_syn,Inspire_Put_Clothes_into_Washing_Machine_syn \
    --ood-eval-tasks Inspire_Pickup_Pillow_MainCamOnly \
    --run-prefix final_ours_ablation_only_step2_h2r_r96_0508 \
    --merge-lora training_data/log/final_ours_step2_0507_004839-blur_r2r_1s-17944d_r256_self_qkvo_ffn_1000s_0507_030557/ckpt/step-1000.safetensors \
    --lora-rank 96 \
    --lora-target-modules self_attn.q,self_attn.k,self_attn.v,self_attn.o,ffn.0,ffn.2 \
    --original-train-size 400 \
    --syn-train-size 0 \
    --in-task-eval-size 16 \
    --ood-eval-size 16 \
    --eval-video-samples-in-task 4 \
    --eval-video-samples-ood 2 \
    --data-seed 42 \
    --t5-cache-dir training_data/cache/t5/h2r/1s \
    --max-steps 1000 \
    --save-steps 100 \
    --eval-steps 100 \
    --eval-video-steps 100 \
    --batch-size 4 \
    --lr 1e-4 \
    --lr-min 1e-6 \
    --warmup-steps 50 \
    --weight-decay 0.01 \
    --wandb-project Flip \
    --wandb-tags final ours h2r mixed ablation_study step2 only layout:self_qkvo_ffn rank:96 orig:400 syn:0

scripts/flip_run.sh train_mitty_mixed_h2r --cuda 1 --nproc 1 -- \
    --task-name mixed_h2r_1s_400orig_0syn \
    --original-train-tasks Inspire_Collect_Clothes_MainCamOnly,Inspire_Put_Clothes_into_Washing_Machine \
    --syn-train-tasks Inspire_Collect_Clothes_MainCamOnly_syn,Inspire_Put_Clothes_into_Washing_Machine_syn \
    --ood-eval-tasks Inspire_Pickup_Pillow_MainCamOnly \
    --run-prefix final_ours_ablation_only_step1_h2r_r96_0508 \
    --merge-lora training_data/log/final_ours_step1_0507_004839-identity_r2r_1s-22592d_r32_self_qkvo_ffn_1000s_0507_004911/ckpt/step-1000.safetensors \
    --lora-rank 96 \
    --lora-target-modules self_attn.q,self_attn.k,self_attn.v,self_attn.o,ffn.0,ffn.2 \
    --original-train-size 400 \
    --syn-train-size 0 \
    --in-task-eval-size 16 \
    --ood-eval-size 16 \
    --eval-video-samples-in-task 4 \
    --eval-video-samples-ood 2 \
    --data-seed 42 \
    --t5-cache-dir training_data/cache/t5/h2r/1s \
    --max-steps 1000 \
    --save-steps 100 \
    --eval-steps 100 \
    --eval-video-steps 100 \
    --batch-size 4 \
    --lr 1e-4 \
    --lr-min 1e-6 \
    --warmup-steps 50 \
    --weight-decay 0.01 \
    --wandb-project Flip \
    --wandb-tags final ours h2r mixed ablation_study step1 only layout:self_qkvo_ffn rank:96 orig:400 syn:0