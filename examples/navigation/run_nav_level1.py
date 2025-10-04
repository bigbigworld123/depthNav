from depthnav.scripts.runner import run_experiment

if __name__ == "__main__":
    config_keys = (
        "env.scene_kwargs.load_geodesics",
        "env.scene_kwargs.path",
        "train_bptt.iterations",
    )

    # for tensorboard
    run_params = {
        "level0": (False, "configs/box_2", 500),
        "level1": (True, "configs/level_1", 12000),
    }
    base_config_files = [
        "examples/navigation/train_cfg/nav_empty.yaml",
        "examples/navigation/train_cfg/nav_levelX.yaml",
    ]
    run_experiment(
        script="depthnav/scripts/train_bptt.py",
        # experiment_dir="examples/navigation/logs/level1_resnet",
        experiment_dir="examples/navigation/logs/asymmetric_fusion_experiment",
        config_keys=config_keys,
        run_params=run_params,
        base_config_files=base_config_files,
        # policy_config_file="examples/navigation/policy_cfg/small_yaw.yaml",
        # policy_config_file="examples/navigation/policy_cfg/large_yaw.yaml",
        policy_config_file="examples/navigation/policy_cfg/asymmetric_fusion.yaml",
        eval_configs=[
            "examples/navigation/eval_cfg/nav_level1.yaml",
        ],
        eval_csvs=[
            "examples/navigation/logs/level1/nav_level_1_resnet.csv",
        ],
        curriculum=True,
        max_retries=5,
    )
