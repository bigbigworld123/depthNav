#!/bin/bash

python depthnav/scripts/train_bptt.py \
    --logging_root examples/navigation/logs/debug \
    --cfg_file examples/navigation/cfg/nav2_baseline.yaml \
    --run_name debugging \
    --render