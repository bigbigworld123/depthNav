# 文件: depthnav/scripts/train_shac.py

#!/usr/bin/env python3
import faulthandler

faulthandler.enable()

import sys
import yaml
import torch as th
import argparse
from copy import deepcopy

# 导入 SHAC
from depthnav.policies.shac_algorithm import SHAC

from depthnav.envs.env_aliases import env_aliases
from depthnav.policies.policy_aliases import policy_aliases
from depthnav.policies.multi_input_policy import MultiInputPolicy
from depthnav.common import ExitCode
from depthnav.scripts.eval_logger import Evaluate


def main(args):
    
    # runner.py 已经将所有 config 合并到了 args.cfg_file 中
    with open(args.cfg_file, "r") as file:
        config = yaml.safe_load(file)

    # --- 关键：为训练环境启用梯度 ---
    config["env"]["requires_grad"] = True
    
    # 应用 update_env_kwargs
    if "update_env_kwargs" in config:
        print("Applying 'update_env_kwargs' from policy config to main env...")
        config["env"].update(config["update_env_kwargs"])
    
    env_class = env_aliases[config["env_class"]]
    env = env_class(**config["env"])

    # 加载评估环境
    eval_envs = []
    if args.eval_configs is not None:
        for cfg_file in args.eval_configs:
            with open(cfg_file, "r") as file:
                eval_config = yaml.safe_load(file)
            
            # (确保评估环境也应用了 update_env_kwargs)
            if "update_env_kwargs" in config:
                print(f"Applying 'update_env_kwargs' to eval env: {cfg_file}")
                if "env" not in eval_config:
                    eval_config["env"] = {}
                eval_config["env"].update(config["update_env_kwargs"])

            if args.render:
                eval_config["env"].setdefault("scene_kwargs", {})["render_settings"] = {
                    "mode": "follow",
                    "view": "back",
                    "sensor_type": "color",
                    "resolution": [512, 512],
                    "axes": True,
                    "trajectory": False,
                    "object_path": "./datasets/depthnav_dataset/configs/agents/DJI_Mavic_Mini_2.object_config.json",
                    "line_width": 2.0,
                }
            
            # 评估环境不需要梯度
            eval_config["env"]["requires_grad"] = False

            eval_env_class = env_aliases[eval_config["env_class"]]
            eval_env = eval_env_class(**eval_config["env"])
            eval_envs.append(eval_env)

    # 加载策略 (config 中已包含 policy_class 和 policy)
    policy_class = policy_aliases[config["policy_class"]] 
    policy_kwargs = config["policy"] # 这是一个包含 'critic_mlp' 的字典

    # --- 关键修复：将 critic_mlp 从 Actor 的参数中分离 ---
    # 1. 复制一份给 Actor
    actor_kwargs = policy_kwargs.copy() 
    # 2. 从 Actor 的参数中移除 critic_mlp
    if "critic_mlp" in actor_kwargs:
        actor_kwargs.pop("critic_mlp") 
        print("Separated 'critic_mlp' config from Actor (MultiInputPolicy) kwargs.")
    else:
        print("Warning: 'critic_mlp' not found in policy config. Critic may use default settings.")
    # --- 结束修复 ---

    if policy_class == MultiInputPolicy:
        # --- 关键修复：使用清理过的 actor_kwargs ---
        policy = policy_class(env.observation_space, **actor_kwargs)
        # --- 结束修复 ---
    else:
        policy = policy_class(**actor_kwargs)

    # 加载权重
    if args.weight is not None:
        try:
            # 尝试加载 bptt 权重 (仅 state_dict)
            policy.load(args.weight) 
            print(f"Successfully loaded policy weights (BPTT-style) from {args.weight}")
        except Exception as e:
            print(f"Could not load BPTT-style weights, trying SHAC checkpoint... Error: {e}")
            try:
                # 尝试加载 SHAC 检查点 (字典)
                checkpoint = th.load(args.weight, map_location=policy.device)
                policy.load_state_dict(checkpoint['actor_state_dict'])
                print(f"Successfully loaded 'actor_state_dict' from SHAC checkpoint {args.weight}")
            except Exception as e2:
                print(f"Failed to load checkpoint. Error: {e2}. Proceeding with uninitialized policy.")

    elif config.get("weights_file", None):
        policy.load(config["weights_file"])

    # --- 实例化 SHAC 训练器 ---
    trainer = SHAC(
        env=env,
        eval_envs=eval_envs,
        eval_csvs=args.eval_csvs,
        policy=policy,
        policy_kwargs=policy_kwargs, # <-- 传入 *原始* kwargs (包含 critic_mlp)
        run_name=args.run_name,
        logging_dir=args.logging_root,
        **config["train_shac"] 
    )
    
    # 训练器加载 RMS (如果存在于检查点中)
    if args.weight is not None:
        try:
            checkpoint = th.load(args.weight, map_location=trainer.device)
            if 'obs_rms' in checkpoint and checkpoint['obs_rms'] is not None:
                trainer.obs_rms = checkpoint['obs_rms'].to(trainer.device)
                print("Successfully loaded obs_rms from checkpoint.")
            if 'privilege_obs_rms' in checkpoint and checkpoint['privilege_obs_rms'] is not None:
                trainer.privilege_obs_rms = checkpoint['privilege_obs_rms'].to(trainer.device)
                print("Successfully loaded privilege_obs_rms from checkpoint.")
        except Exception as e:
            print(f"Could not load RMS stats from {args.weight}. Starting with fresh stats. (This is normal for BPTT weights).")

    # 训练
    exit_code = trainer.learn(args.render, args.start_iter)

    # 保存模型
    print("Done training. Saving model")
    trainer.save(filepath=os.path.join(trainer.run_path, "final_policy.pth"))
    sys.exit(exit_code.value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg_file", type=str, default=None 
    )
    parser.add_argument("--logging_root", type=str, default="logs") 
    parser.add_argument("--run_name", type=str) 
    parser.add_argument(
        "--start_iter", type=int, default=0, help="start index to log to df"
    )
    parser.add_argument(
        "--weight", type=str, default=None, help="pre-trained model weights file"
    )
    parser.add_argument("--render", action="store_true", help="Show observations")
    parser.add_argument(
        "--eval_configs",
        nargs="+",
        type=str,
        default=None,
        help="list of eval env paths",
    )
    parser.add_argument(
        "--eval_csvs",
        nargs="+",
        type=str,
        default=None,
        help="list of paths to write eval stats",
    )
    
    args = parser.parse_args()
    
    if args.cfg_file is None:
        print("Error: This script is intended to be called by 'runner.py' (e.g., via 'run_nav_level1_shac.py').")
        print("Please run 'examples/navigation/run_nav_level1.py' (which you modified) instead.")
        sys.exit(ExitCode.ERROR.value)
    
    main(args)