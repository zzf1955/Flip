给出指令
1. Task 恒等映射
2. lora Attention rank = 32
3. Step = 1000
4. train data 1000
5. eval data 16+16
6. eval Video 4+4

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

  scripts/flip_run.sh train --cuda 3 -- \
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

python -m src.pipeline.make_robot_pair \
--task all \
--workers 64 \
--clean

python -m src.pipeline.make_pair \
--task all \
--second 1s \
--data-type h2r \
--human-source seedance_direct \
--workers 64 \
--clean

python -m src.pipeline.make_pair \
--task all \
--second 1s \
--data-type blur_r2r \
--human-source seedance_direct \
--workers 64 \
--clean