from depthnav.scripts.runner import run_experiment

if __name__ == "__main__":
    
    # 1. 定义要覆盖的参数
    config_keys = (
        "train_bptt.iterations",
    )

    # 2. 定义您的 3 阶段课程计划
    run_params = {
        # 阶段1: 空环境 (500 步)
        "level0_empty": (500,),
        
        # 阶段2: 新 L2 双环 (6000 步)
        "level2_dual_ring": (6000,),
        
        # 阶段3: 新 L3 三环 (9000 步)
        "level3_tri_ring": (8500,),
    }
    
    # 3. 定义与课程计划顺序对应的配置文件
    base_config_files = [
        "examples/navigation/train_cfg/nav_empty.yaml",
        "examples/navigation/train_cfg/nav_level_2_dual_ring.yaml",
        "examples/navigation/train_cfg/nav_level_3_tri_ring.yaml",
    ]
    
    # <<< START MODIFICATION: 更改日誌目錄和策略文件 >>>
    # 4. 设置日志和评估 (评估已启用)
    experiment_log_dir = "examples/navigation/logs/my_maze_temporal_attention" # 新的實驗日誌
    
    # L2 评估文件
    eval_config_file_L2 = "examples/navigation/eval_cfg/nav_level1.yaml"

    # 5. 运行实验
    run_experiment(
        script="depthnav/scripts/train_bptt.py",
        experiment_dir=experiment_log_dir,
        config_keys=config_keys,
        run_params=run_params,
        base_config_files=base_config_files,
        
        policy_config_file="examples/navigation/policy_cfg/temporal_attention.yaml", # <--- 換上新的策略
        
        # --- 评估已开启 ---
        eval_configs=[
            eval_config_file_L2, # 使用 L2 地图评估
        ], 
        eval_csvs=[
            f"{experiment_log_dir}/L2_eval_log.csv", # <--- 更新 CSV 路徑
        ],
        
        curriculum=True, 
        max_retries=1,
    )