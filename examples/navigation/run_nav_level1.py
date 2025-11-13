# run_nav_level1.py (修改版)

from depthnav.scripts.runner import run_experiment

if __name__ == "__main__":
    config_keys = (
        "env.scene_kwargs.load_geodesics",
        "env.scene_kwargs.path",
        "train_bptt.iterations",
    )

    # for tensorboard
    run_params = {
        "level0": (False, "configs/box_2", 200),
        "level1": (False, "configs/level_1", 13000),
    }
    base_config_files = [
        "examples/navigation/train_cfg/nav_empty.yaml",
        "examples/navigation/train_cfg/nav_levelX.yaml",
    ]
    run_experiment(
        script="depthnav/scripts/train_bptt.py",
        experiment_dir="examples/navigation/logs/level1_no_toa_no_yaw", # 建议使用新的日志目录
        config_keys=config_keys,
        run_params=run_params,
        base_config_files=base_config_files,

        # ==========================================================
        # !! 核心修改：将策略配置文件切换为 "small_no_yaw.yaml" !!
        # ==========================================================
        policy_config_file="examples/navigation/policy_cfg/spatial_memory_injection.yaml",

        eval_configs=[
            "examples/navigation/eval_cfg/nav_level1.yaml",
        ],
        eval_csvs=[
            "examples/navigation/logs/level1_no_toa_no_yaw/nav_level_1.csv",
        ],
        curriculum=True,
        max_retries=5,
    )