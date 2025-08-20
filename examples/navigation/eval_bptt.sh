#!/bin/bash

python examples/navigation/eval_visual.py \
    --cfg_file "examples/navigation/eval_cfg/nav2_empty.yaml" \
    --policy_cfg_file "examples/navigation/policy_cfg/small_yaw.yaml" \
    --weight "examples/navigation/logs/geodesic_test/level1_geodesic_collision_no_slerp_target_distance_2_iteration_10000.pth" \
    --render \
    --num_envs 1 \
    --plot 1 \
