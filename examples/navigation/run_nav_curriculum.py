from depthnav.scripts.runner import run_experiment

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
    base_config_files = [
        "examples/navigation/train_cfg/nav_empty.yaml",
        "examples/navigation/train_cfg/nav_levelX.yaml",
        "examples/navigation/train_cfg/nav_levelX.yaml",
        "examples/navigation/train_cfg/nav_levelX.yaml",
        "examples/navigation/train_cfg/nav_levelX.yaml",
    ]
    run_experiment(
        script="depthnav/scripts/train_bptt.py",
        experiment_dir="examples/navigation/logs/curriculum",
        config_keys=config_keys,
        run_params=run_params,
        base_config_files=base_config_files,
        policy_config_file="examples/navigation/policy_cfg/small_yaw.yaml",
        eval_configs=[
            "examples/navigation/eval_cfg/nav_level1.yaml",
            "examples/navigation/eval_cfg/nav_level2.yaml",
            "examples/navigation/eval_cfg/nav_level3.yaml",
            "examples/navigation/eval_cfg/nav_level4.yaml",
        ],
        eval_csvs=[
            "examples/navigation/logs/curriculum/nav_level_1.csv",
            "examples/navigation/logs/curriculum/nav_level_2.csv",
            "examples/navigation/logs/curriculum/nav_level_3.csv",
            "examples/navigation/logs/curriculum/nav_level_4.csv",
        ],
        curriculum=True,
        max_retries=5,
    )
