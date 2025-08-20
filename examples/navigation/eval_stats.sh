#!/bin/bash

python depthnav/scripts/eval_logger.py \
    --cfg_file "examples/navigation/eval_cfg/nav_ring.yaml" \
    --weight examples/navigation/logs/curriculum_2/level1_small_3_iteration_4000.pth \
    --policy_cfg_file "examples/navigation/policy_cfg/small_yaw.yaml" \
    --num_envs 4 \
