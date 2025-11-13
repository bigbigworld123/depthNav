#!/usr/bin/env python3
import faulthandler

faulthandler.enable()

import sys
import yaml
import torch as th
import argparse
from copy import deepcopy
from depthnav.policies.shac_algorithm import SHAC # (!!!) 替换 BPTT
from depthnav.policies.critic import CriticMLP   # (!!!) 导入
from depthnav.policies.vae import VAE             # (!!!) 导入
from depthnav.policies.bptt_algorithm import BPTT
from depthnav.envs.env_aliases import env_aliases
from depthnav.policies.policy_aliases import policy_aliases
from depthnav.policies.multi_input_policy import MultiInputPolicy
from depthnav.common import ExitCode
from depthnav.scripts.eval_logger import Evaluate


def main(args):
    # load training env
    with open(args.cfg_file, "r") as file:
        config = yaml.safe_load(file)
    env_class = env_aliases[config["env_class"]]
    env = env_class(requires_grad=True, **config["env"])

    # load eval envs
    eval_envs = []
    if args.eval_configs is not None:
        for cfg_file in args.eval_configs:
            with open(cfg_file, "r") as file:
                eval_config = yaml.safe_load(file)

            if args.render:
                eval_config["scene_kwargs"]["render_settings"] = {
                    "mode": "follow",
                    "view": "back",
                    "sensor_type": "color",
                    "resolution": [512, 512],
                    "axes": True,
                    "trajectory": False,
                    "object_path": "./datasets/depthnav_dataset/configs/agents/DJI_Mavic_Mini_2.object_config.json",
                    "line_width": 2.0,
                }

            env_class = env_aliases[config["env_class"]]
            eval_env = env_class(requires_grad=False, **eval_config["env"])
            eval_envs.append(eval_env)

    # load policy
    policy_class = policy_aliases[config["policy_class"]]
    policy_kwargs = config["policy"]
    if policy_class == MultiInputPolicy:
        policy = policy_class(env.observation_space, **policy_kwargs)
    else:
        policy = policy_class(**policy_kwargs)

    # (!!!) 初始化 Critic 网络
    # 您需要从 gradNav 的 config (e.g., drone_long_traj.yaml) 中获取 critic_mlp 的配置
    #
    critic_config = config.get("critic_network", { # 假设您会把配置加到 yaml 中
        "critic_mlp": {"units": [512, 256, 128], "activation": "elu"}
    })
    critic = CriticMLP(
        privilege_obs_dim=env.privilege_obs_dim, # (!!!) 使用 env 属性
        cfg_network=critic_config,
        device=policy_kwargs.get("device", "cuda")
    )

    # (!!!) 初始化 VAE (CENet) 网络
    # 同样，从 gradNav 的 config 中获取 vae 的配置
    #
    vae_config = config.get("vae_network", { # 假设您会把配置加到 yaml 中
        "kld_weight": 1.0, 
        "encoder_hidden_dims": [256, 256, 256], # <--- (!!!) 修正 (!!!)
        "decoder_hidden_dims": [32, 64, 128, 256]  # <--- (!!!) 修正 (!!!)
    })
    vae = VAE(
        num_obs=env.vae_obs_dim, # (!!!) 使用 env 属性
        num_history=env.num_history,  # 添加缺失的num_history参数
        num_latent=env.num_latent,
        device=policy_kwargs.get("device", "cuda"),
        **vae_config
    )

    if args.weight is not None:
        # (!!!) 注意：SHAC 的权重需要分别加载
        # gradNav 的 save/load 在一个 .pt 文件中保存了所有网络
        # (save/load 方法)
        print("--- Loading Checkpoint for SHAC ---")
        checkpoint = th.load(args.weight)
        policy.load_state_dict(checkpoint[0].state_dict()) # 索引 0 是 Actor
        critic.load_state_dict(checkpoint[1].state_dict()) # 索引 1 是 Critic
        vae.load_state_dict(checkpoint[4].state_dict())    # 索引 4 是 VAE
        # 您还需要加载 RMS 状态
    elif config.get("weights_file", None):
        policy.load(config["weights_file"])

    # setup trainer
    # 使用默认配置以防配置文件中没有train_shac部分
    shac_config = config.get("train_shac", {
        "horizon": 32,
        "gamma": 0.99,
        "lambda": 0.95,
        "critic_iterations": 16,
        "learning_rate_init": 1e-4,
        "learning_rate_critic": 1e-4,
        "learning_rate_vae": 5e-4,
        "device": "cuda",
        "max_epochs": 500  # 添加默认的max_epochs参数
    })

    trainer = SHAC(
        env=env,
        eval_envs=eval_envs,
        eval_csvs=args.eval_csvs,
        policy=policy,
        critic=critic,
        vae=vae,
        run_name=args.run_name,
        logging_dir=args.logging_root,
        # 从 shac_config 中读取 SHAC 特定参数
        steps_num=shac_config.get("horizon", 32), # SHAC 的 horizon 是短时窗
        gamma=shac_config.get("gamma", 0.99),
        lam=shac_config.get("lambda", 0.95),
        critic_iterations=shac_config.get("critic_iterations", 16),
        learning_rate_actor=shac_config.get("learning_rate_init", 1e-4),
        learning_rate_critic=shac_config.get("learning_rate_critic", 1e-4),
        learning_rate_vae=shac_config.get("learning_rate_vae", 5e-4),
        max_epochs=shac_config.get("max_epochs", 500),  # 添加max_epochs参数
        device=shac_config.get("device", "cuda")
    )

    # train
    exit_code = trainer.learn(args.render, args.start_iter)

    # (!!!) 保存模型 (模仿 gradNav.save)
    print("Done training. Saving model")
    save_path = os.path.join(args.logging_root, args.run_name + ".pth")
    th.save([
        trainer.policy,          # Actor (包含您的 CNN)
        trainer.critic,          # Critic
        trainer.target_critic,   # Target Critic
        trainer.obs_rms,         # 观测归一化
        trainer.vae              # VAE (CENet)
    ], save_path)

    sys.exit(exit_code.value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg_file", type=str, default="examples/hovering/bptt_hover_1.yaml"
    )
    parser.add_argument("--logging_root", type=str, default="examples/hovering/saved")
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
    main(args)
