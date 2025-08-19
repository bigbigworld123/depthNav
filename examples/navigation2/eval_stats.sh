#!/bin/bash

python skinny_VisFly/scripts/eval_logger.py \
    --cfg_file "skinny_examples/navigation2/eval_cfg/nav2_ring.yaml" \
    --weight skinny_examples/navigation2/logs/curriculum_2/level1_small_3_iteration_4000.pth \
    --policy_cfg_file "skinny_examples/navigation2/policy_cfg/small_yaw.yaml" \
    --num_envs 4 \
