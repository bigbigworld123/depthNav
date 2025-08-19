from skinny_VisFly.scripts.runner import run_experiment

if __name__ == "__main__":

    config_keys = (
        "env.scene_kwargs.load_geodesics",
        "env.scene_kwargs.path",
        "train_bptt.iterations",
    )

    # small model
    run_params = {
        "level0_small_safe_yaw": (False, "configs/box_2", 500),
        "level1_small_safe_yaw": (True, "configs/level_1", 10000),
        "level2_small_safe_yaw": (True, "configs/level_2", 10000),
        "level3_small_safe_yaw": (True, "configs/level_3", 10000),
        "level4_small_safe_yaw": (True, "configs/level_4", 10000),
    }
    base_config_files=[
        "skinny_examples/navigation2/train_cfg/nav2_empty.yaml",
        "skinny_examples/navigation2/train_cfg/nav2_levelX.yaml",
        "skinny_examples/navigation2/train_cfg/nav2_levelX.yaml",
        "skinny_examples/navigation2/train_cfg/nav2_levelX.yaml",
        "skinny_examples/navigation2/train_cfg/nav2_levelX.yaml",
    ]
    run_experiment(
        script="skinny_VisFly/scripts/train_bptt.py",
        experiment_dir="skinny_examples/navigation2/logs/curriculum",
        config_keys=config_keys,
        run_params=run_params,
        base_config_files=base_config_files,
        policy_config_file="skinny_examples/navigation2/policy_cfg/small_yaw.yaml",
        eval_configs=[
            "skinny_examples/navigation2/eval_cfg/nav2_level1.yaml",
            "skinny_examples/navigation2/eval_cfg/nav2_level2.yaml",
            "skinny_examples/navigation2/eval_cfg/nav2_level3.yaml",
            "skinny_examples/navigation2/eval_cfg/nav2_level4.yaml",
        ],
        eval_csvs=[
            "skinny_examples/navigation2/logs/curriculum/nav2_level_1.csv",
            "skinny_examples/navigation2/logs/curriculum/nav2_level_2.csv",
            "skinny_examples/navigation2/logs/curriculum/nav2_level_3.csv",
            "skinny_examples/navigation2/logs/curriculum/nav2_level_4.csv",
        ],
        curriculum=True,
        max_retries=5,
    )