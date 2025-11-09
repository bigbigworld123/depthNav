#!/usr/bin/env python3
import faulthandler

faulthandler.enable()

import sys
import yaml
import torch as th
import argparse
from copy import deepcopy

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

    # <<< START MODIFICATION 3.11: 傳遞 policy_kwargs >>>
    # 為了支持時間注意力，我們需要將完整的 policy_kwargs 傳遞給構造函數
    # 並從中提取 MultiInputPolicy 需要的特定參數。
    if policy_class == MultiInputPolicy:
        policy = policy_class(
            env.observation_space, 
            net_arch=policy_kwargs["net_arch"],
            activation_fn=policy_kwargs["activation_fn"],
            output_activation_fn=policy_kwargs["output_activation_fn"],
            feature_extractor_class=policy_kwargs["feature_extractor_class"], # 確保傳遞
            policy_kwargs=policy_kwargs, # <--- 傳遞完整的 policy 字典
            output_activation_kwargs=policy_kwargs.get("output_activation_kwargs"),
            feature_extractor_kwargs=policy_kwargs.get("feature_extractor_kwargs"),
            device=policy_kwargs.get("device", "cuda")
        )
    # <<< END MODIFICATION 3.11 >>>
    else:
        policy = policy_class(**policy_kwargs)

    if args.weight is not None:
        policy.load(args.weight)
    elif config.get("weights_file", None):
        policy.load(config["weights_file"])

    # setup trainer
    trainer = BPTT(
        env=env,
        eval_envs=eval_envs,
        eval_csvs=args.eval_csvs,
        policy=policy,
        run_name=args.run_name,
        logging_dir=args.logging_root,
        **config["train_bptt"],
    )

    # train
    exit_code = trainer.learn(args.render, args.start_iter)

    # save model
    print("Done training. Saving model")
    trainer.save()
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