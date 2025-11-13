# 文件: examples/navigation/run_nav_level1.py
# (修改版)

from depthnav.scripts.runner import run_experiment

if __name__ == "__main__":
    
    # --- 关键修改：更新 runner.py 用来覆盖的键 ---
    # 我们现在用 'train_shac.iterations' 替换 'train_bptt.iterations'
    config_keys = (
        "env.scene_kwargs.load_geodesics",
        "env.scene_kwargs.path",
        "train_shac.iterations", # <-- 修改
    )

    # for tensorboard
    run_params = {
        "level0": (False, "configs/box_2", 500),
        "level1": (False, "configs/level_1", 13000),
    }

    # --- 关键修改：使用新的 SHAC 配置文件 ---
    base_config_files = [
        "examples/navigation/train_cfg/nav_shac_level0.yaml", # <-- 修改
        "examples/navigation/train_cfg/nav_shac_levelX.yaml", # <-- 修改
    ]

    # --- 关键修改：定义日志目录和策略文件 ---
    experiment_log_dir = "examples/navigation/logs/shac_level1_run" # <-- 新的日志目录
    policy_config = "examples/navigation/policy_cfg/small_no_yaw_shac.yaml" # <-- 新的策略配置
    # --- 结束修改 ---
    
    run_experiment(
        # --- 关键修改：调用新的训练脚本 ---
        script="depthnav/scripts/train_shac.py", # <-- 修改
        # --- 结束修改 ---
        
        experiment_dir=experiment_log_dir,
        config_keys=config_keys,
        run_params=run_params,
        base_config_files=base_config_files,

        policy_config_file=policy_config,

        eval_configs=[
            "examples/navigation/eval_cfg/nav_level1.yaml",
        ],
        eval_csvs=[
            f"{experiment_log_dir}/nav_level_1_eval.csv", # <-- 确保 CSV 在日志目录中
        ],
        curriculum=True,
        max_retries=5,
    )