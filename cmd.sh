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


scripts/flip_run_2.sh train --cuda 0 -- \
  --task-name identity_r2r_1s \
  --lora-rank 32 \
  --train-size 10000 \
  --in-task-eval-size 16 \
  --ood-eval-size 16 \
  --in-task-video-size 8 \
  --ood-video-size 8 \
  --batch-size 4


scripts/flip_run_2.sh train --cuda 2 -- \
  --task-name identity_r2r_1s \
  --lora-rank 32 \
  --lora-target-modules q,k,v,o,ffn.0,ffn.2 \
  --batch-size 4 \
  --train-size 10000 \
  --in-task-eval-size 16 \
  --ood-eval-size 16 \
  --in-task-video-size 8 \
  --ood-video-size 8

  scripts/flip_run.sh train --cuda 3 -- \
  --task-name identity_r2r_1s \
  --lora-rank 32 \
  --lora-target-modules ffn.0,ffn.2 \
  --batch-size 4 \
  --train-size 10000 \
  --in-task-eval-size 16 \
  --ood-eval-size 16 \
  --in-task-video-size 8 \
  --ood-video-size 8