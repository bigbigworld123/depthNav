#!/bin/bash

python skinny_VisFly/scripts/train_bptt.py \
    --logging_root skinny_examples/navigation2/logs/debug \
    --cfg_file skinny_examples/navigation2/cfg/nav2_baseline.yaml \
    --run_name debugging \
    --render