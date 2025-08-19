from skinny_VisFly.scripts.runner import run_experiment

if __name__ == "__main__":
    config_keys = (
        "env.scene_kwargs.path",
        "env.random_kwargs.position.class",
        "env.reward_kwargs.lambda_c",
        "env.reward_kwargs.beta_2",
        "env.reward_kwargs.safe_view_degrees",
        "train_bptt.iterations"
    )

    run_params = {
        "level0_yaw_exp_avg_half_exp": ("../VisFly-datasets/datasets/skinny_dataset/configs/box_2", "uniform", 0, -6, 0.0, 500),
        "level1_yaw_exp_avg_half_exp": ("../VisFly-datasets/datasets/skinny_dataset/configs/ring_level0", "cylinder", 6, -6, 0.0, 5000),
        "level2_yaw_exp_avg_half_exp": ("../VisFly-datasets/datasets/skinny_dataset/configs/ring_level1", "cylinder", 6, -6, 0.0, 5000),
        "level3_yaw_exp_avg_half_exp": ("../VisFly-datasets/datasets/skinny_dataset/configs/ring_level2", "cylinder", 6, -6, 0.0, 5000),
    }
    run_experiment(
        script="skinny_VisFly/scripts/train_bptt.py",
        experiment_dir="skinny_examples/navigation2/logs/drone_dome",
        base_config_file="skinny_examples/navigation2/train_cfg/nav2_ring.yaml",
        policy_config_file="skinny_examples/navigation2/policy_cfg/small_yaw.yaml",
        config_keys=config_keys,
        run_params=run_params,
        curriculum=True,
        max_retries=3,
    )
