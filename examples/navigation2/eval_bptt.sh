#!/bin/bash

python skinny_examples/navigation2/eval_visual.py \
    --cfg_file "skinny_examples/navigation2/eval_cfg/nav2_empty.yaml" \
    --policy_cfg_file "skinny_examples/navigation2/policy_cfg/small_yaw.yaml" \
    --weight "skinny_examples/navigation2/logs/geodesic_test/level1_geodesic_collision_no_slerp_target_distance_2_iteration_10000.pth" \
    --render \
    --num_envs 1 \
    --plot 1 \
