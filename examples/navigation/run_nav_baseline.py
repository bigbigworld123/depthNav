from depthnav.scripts.runner import run_experiment

if __name__ == "__main__":
    config_keys = (
        "env.scene_kwargs.path",
        "env.random_kwargs.position.class",
        "env.reward_kwargs.lambda_c",
        "env.reward_kwargs.beta_2",
        "env.reward_kwargs.safe_view_degrees",
        "train_bptt.iterations",
    )

    run_params = {
        "level0_yaw_exp_avg_half_exp": (
            "../datasets/depthnav_dataset/configs/box_2",
            "uniform",
            0,
            -6,
            0.0,
            500,
        ),
        "level1_yaw_exp_avg_half_exp": (
            "../datasets/depthnav_dataset/configs/ring_level0",
            "cylinder",
            6,
            -6,
            0.0,
            5000,
        ),
        "level2_yaw_exp_avg_half_exp": (
            "../datasets/depthnav_dataset/configs/ring_level1",
            "cylinder",
            6,
            -6,
            0.0,
            5000,
        ),
        "level3_yaw_exp_avg_half_exp": (
            "../datasets/depthnav_dataset/configs/ring_level2",
            "cylinder",
            6,
            -6,
            0.0,
            5000,
        ),
    }
    run_experiment(
        script="depthnav/scripts/train_bptt.py",
        experiment_dir="examples/navigation/logs/drone_dome",
        base_config_file="examples/navigation/train_cfg/nav_ring.yaml",
        policy_config_file="examples/navigation/policy_cfg/small_yaw.yaml",
        config_keys=config_keys,
        run_params=run_params,
        curriculum=True,
        max_retries=3,
    )
