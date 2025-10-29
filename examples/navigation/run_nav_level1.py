# examples/navigation/run_nav_level1.py

from depthnav.scripts.runner import run_experiment

if __name__ == "__main__":
    config_keys = (
        "env.scene_kwargs.load_geodesics",
        "env.scene_kwargs.path",
        "train_bptt.iterations",
    )

    run_params = {
        "level0": (False, "configs/box_2", 500),
        "level1": (True, "configs/level_1", 20000),
    }
    base_config_files = [
        "examples/navigation/train_cfg/nav_empty.yaml",
        "examples/navigation/train_cfg/nav_levelX.yaml",
    ]
    
    # ===================== 核心修改 =====================
    experiment_name = "topological_sru_v1" # 為您的新實驗命名
    
    run_experiment(
        script="depthnav/scripts/train_bptt.py",
        experiment_dir=f"examples/navigation/logs/{experiment_name}",
        config_keys=config_keys,
        run_params=run_params,
        base_config_files=base_config_files,
        # ！！！ 加載您的新策略配置 ！！！
        policy_config_file="examples/navigation/policy_cfg/topological_sru_policy.yaml",
        eval_configs=[
            "examples/navigation/eval_cfg/nav_level1.yaml",
        ],
        # ！！！ 將評估結果保存到新文件 ！！！
        eval_csvs=[
            f"examples/navigation/logs/{experiment_name}/nav_level_1.csv",
        ],
        curriculum=True,
        max_retries=5,
    )
    # ====================================================