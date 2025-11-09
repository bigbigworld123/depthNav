#!/usr/bin/env python3
import faulthandler

faulthandler.enable()

import cv2
import os
import yaml
import time
import torch as th
import torch.nn.functional as F
import numpy as np
import pandas as pd
import argparse
from tqdm import trange
from copy import deepcopy
from pylogtools import timerlog

timerlog.timer.print_logs(False)

from depthnav.envs.env_aliases import env_aliases
from depthnav.policies.policy_aliases import policy_aliases
from depthnav.policies.multi_input_policy import MultiInputPolicy
from depthnav.common import observation_to_device, rgba2rgb


def main(args):
    with open(args.cfg_file, "r") as file:
        config = yaml.safe_load(file)

    if args.policy_cfg_file:
        with open(args.policy_cfg_file, "r") as file:
            policy_config = yaml.safe_load(file)
        config.update(policy_config)

        if "update_env_kwargs" in policy_config:
            for k, v in policy_config["update_env_kwargs"].items():
                config["env"][k] = v

    eval_config = deepcopy(config["env"])
    eval_config["num_envs"] = args.num_envs

    env_class = env_aliases[config["env_class"]]
    env = env_class(requires_grad=False, **eval_config)

    policy_class = policy_aliases[config["policy_class"]]
    policy_kwargs = config["policy"]
    
    # <<< START MODIFICATION 3.11: (Fix) 傳遞 policy_kwargs (同 train_bptt.py) >>>
    # 這是為了修復加載時間注意力模型
    if policy_class == MultiInputPolicy:
        policy = policy_class(
            env.observation_space, 
            net_arch=policy_kwargs["net_arch"],
            activation_fn=policy_kwargs["activation_fn"],
            output_activation_fn=policy_kwargs["output_activation_fn"],
            feature_extractor_class=policy_kwargs["feature_extractor_class"],
            policy_kwargs=policy_kwargs, # <--- 傳遞完整的 policy 字典
            output_activation_kwargs=policy_kwargs.get("output_activation_kwargs"),
            feature_extractor_kwargs=policy_kwargs.get("feature_extractor_kwargs"),
            device=policy_kwargs.get("device", "cuda")
        )
    # <<< END MODIFICATION 3.11 >>>
    else:
        policy = policy_class(**policy_kwargs)
        
    policy.load(args.weight)
    policy.eval()

    def create_name(run_path, ext=".mp4"):
        index = 1
        while True:
            path = f"{run_path}_{index}"
            if not os.path.exists(path + ext):
                break
            index += 1
        return path

    if args.save_path is None:
        save_path = create_name(args.cfg_file.split(".")[0])

    e = Evaluate(env, policy)

    if args.run_name is None:
        run_name = create_name(args.weight.split(".")[0])
    df = e.run_rollouts(args.num_rollouts, run_name)

    write_header = not os.path.exists(save_path + ".csv")
    df.to_csv(save_path + ".csv", float_format="%.3f", mode="a", header=write_header)
    print(f"wrote stats to {save_path}.csv")


class Evaluate:
    def __init__(self, env, policy, render_kwargs=None):
        self.env = env
        self.policy = policy
        self.render_kwargs = render_kwargs or {}

    @th.no_grad()
    def run_rollouts(self, num_rollouts=1, run_name=0, render=False):
        all_stats = {
            "avg_speed": [],
            "max_speed": [],
            "max_acceleration": [],
            "collision_count": [],
            "success_count": [],
            "timeout_count": [],
            "avg_reward": [],
            "duration": [],
            "steps": [],
            "path_length": [],
            "avg_yaw_rate": [],
            "max_yaw_rate": [],
            "avg_min_obstacle_distance": [],
            "avg_control_effort": [],
            "total_energy_proxy": [], # (已保留)
            "last_action_x": [],
            "last_action_y": [],
            "last_action_z": [],
            "last_action_yaw": [],
            "last_position_x": [],
            "last_position_y": [],
            "last_position_z": [],
            "last_velocity_x": [],
            "last_velocity_y": [],
            "last_velocity_z": [],
        }

        for _ in trange(num_rollouts):
            self.env.reset()
            batch_stats = self.single_rollout(render=render)
            for key, value in all_stats.items():
                # 確保 batch_stats[key] 是一個列表
                if isinstance(batch_stats[key], list):
                    value.extend(batch_stats[key])
                else:
                    # 如果不是列表（例如，如果 num_envs=1 時的錯誤），則包裝它
                    value.append(batch_stats[key])


        self.env.close()

        # <<< START MODIFICATION 3.20: (Fix) 強制轉換為 float32 >>>
        # convert lists to tensors
        all_stats = {key: th.tensor(value, dtype=th.float32) for key, value in all_stats.items()}
        # <<< END MODIFICATION 3.20 >>>

        # summarize the rollout stats in a dataframe
        num_trials = num_rollouts * self.env.num_envs
        rows = [
            {
                "success_rate": all_stats["success_count"].sum().item() / num_trials,
                "collision_rate": all_stats["collision_count"].sum().item()
                / num_trials,
                "timeout_rate": all_stats["timeout_count"].sum().item() / num_trials,
                "num_trials": num_trials,
                "avg_speed": all_stats["avg_speed"].mean().item(),
                "max_speed": all_stats["max_speed"].mean().item(),
                "max_acceleration": all_stats["max_acceleration"].mean().item(),
                "avg_reward": all_stats["avg_reward"].mean().item(),
                "avg_duration": all_stats["duration"].mean().item(),
                "std_duration": all_stats["duration"].std().item(),
                "avg_steps": all_stats["steps"].mean().item(),
                "std_steps": all_stats["steps"].std().item(),
                "avg_path_length": all_stats["path_length"].mean().item(),
                "std_path_length": all_stats["path_length"].std().item(),
                "avg_control_effort": all_stats["avg_control_effort"].mean().item(),
                "total_energy_proxy": all_stats["total_energy_proxy"].mean().item(), # (已保留)
                "avg_yaw_rate": all_stats["avg_yaw_rate"].mean().item(),
                "max_yaw_rate": all_stats["max_yaw_rate"].mean().item(),
                "avg_min_obstacle_distance": all_stats["avg_min_obstacle_distance"]
                .mean()
                .item(),
                "last_action_x": all_stats["last_action_x"].mean().item(),
                "last_action_y": all_stats["last_action_y"].mean().item(),
                "last_action_z": all_stats["last_action_z"].mean().item(),
                "last_action_yaw": all_stats["last_action_yaw"].mean().item(),
                "last_position_x": all_stats["last_position_x"].mean().item(),
                "last_position_y": all_stats["last_position_y"].mean().item(), # <<< 錯誤發生行
                "last_position_z": all_stats["last_position_z"].mean().item(),
                "last_velocity_x": all_stats["last_velocity_x"].mean().item(),
                "last_velocity_y": all_stats["last_velocity_y"].mean().item(),
                "last_velocity_z": all_stats["last_velocity_z"].mean().item(),
            }
        ]
        df = pd.DataFrame(rows, index=[run_name])
        return df

    @th.no_grad()
    def single_rollout(self, render=False):
        # For each agent in the batch
        # <<< START MODIFICATION 3.19: (Fix) 將所有 0 改為 0.0 >>>
        agent_logs = [
            {
                "position": [],
                "velocity": [],
                "speed": [],
                "acceleration": [],
                "jerk": [],
                "yaw_rate": [],
                "obstacle_distance": [],
                "success": 0.0,
                "collision": 0.0,
                "timeout": 0.0,
                "avg_reward": 0.0,
                "duration": 0.0,
                "steps": 0.0,
                "path_length": 0.0,
                "avg_control_effort": 0.0,
                "total_energy_proxy": 0.0,
                "last_action_x": 0.0,
                "last_action_y": 0.0,
                "last_action_z": 0.0,
                "last_action_yaw": 0.0,
                "last_position_x": 0.0,
                "last_position_y": 0.0,
                "last_position_z": 0.0,
                "last_velocity_x": 0.0,
                "last_velocity_y": 0.0,
                "last_velocity_z": 0.0,
            }
            for _ in range(self.env.num_envs)
        ]
        # <<< END MODIFICATION 3.19 >>>

        eval_info_id_list = [i for i in range(self.env.num_envs)]

        # <<< START MODIFICATION 3.16: 初始化隱藏狀態元組 >>>
        if hasattr(self.policy, "is_temporal_attention") and self.policy.is_temporal_attention:
            K, H = self.policy.attention_history_shape
            h_t = th.zeros((self.env.num_envs, H), device=self.policy.device)
            history_buffer = th.zeros((self.env.num_envs, K, H), device=self.policy.device)
            latent_tuple = (h_t, history_buffer)
        elif self.policy.is_recurrent:
            latent_state = th.zeros(
                (self.env.num_envs, self.policy.latent_dim), device=self.policy.device
            )
        # <<< END MODIFICATION 3.16 >>>
        
        while True:
            obs = observation_to_device(self.env.get_observation(), self.policy.device)
            
            # <<< START MODIFICATION 3.17: 更新評估時的 Policy 調用 >>>
            if type(self.policy) == MultiInputPolicy:
                if hasattr(self.policy, "is_temporal_attention") and self.policy.is_temporal_attention:
                    action, latent_tuple = self.policy(obs, latent_tuple)
                elif self.policy.is_recurrent:
                    action, latent_state = self.policy(obs, latent_tuple)
                else:
                    action = self.policy(obs)
            else:
                action = self.policy(obs["state"])
            # <<< END MODIFICATION 3.17 >>>
            
            obs, reward, terminated, infos = self.env.step(action, is_test=True)

            if render:
                self.render_kwargs["points"] = th.cat(
                    [self.env.position.unsqueeze(1), self.env.target.unsqueeze(1)],
                    dim=1,
                )
                # ... (render 邏輯) ...
                cv2.waitKey(1)

            # log
            for index in reversed(eval_info_id_list):
                if not terminated[index]:
                    # metrics to log at every step
                    agent_logs[index]["speed"].append(
                        self.env.velocity[index].norm().item()
                    )
                    agent_logs[index]["acceleration"].append(
                        self.env.acceleration[index].norm().item()
                    )
                    agent_logs[index]["jerk"].append(self.env.jerk[index].norm().item())
                    agent_logs[index]["yaw_rate"].append(
                        self.env.omega[index][2].item()
                    )
                    agent_logs[index]["obstacle_distance"].append(
                        self.env.collision_dis[index].item()
                    )

                    # these metrics aren't logged directly, but are used to compute other metrics
                    agent_logs[index]["position"].append(self.env.position[index])

                else:
                    # metrics to log only once at the end of the episode
                    
                    # <<< START MODIFICATION 3.18: (可選) 重置已終止的隱藏狀態 >>>
                    if hasattr(self.policy, "is_temporal_attention") and self.policy.is_temporal_attention:
                        h_t, history_buffer = latent_tuple
                        h_t[index] = 0.
                        history_buffer[index] = 0.
                        latent_tuple = (h_t, history_buffer)
                    elif self.policy.is_recurrent:
                        latent_state[index] = 0.
                    # <<< END MODIFICATION 3.18 >>>
                    
                    eval_info_id_list.remove(index)

                    agent_logs[index]["collision"] = (
                        self.env.is_collision[index].int().item()
                    )
                    agent_logs[index]["success"] = int(infos[index]["is_success"])
                    agent_logs[index]["timeout"] = int(
                        not agent_logs[index]["collision"]
                        and not agent_logs[index]["success"]
                    )
                    agent_logs[index]["avg_reward"] = infos[index][
                        "episode_avg_step_reward"
                    ].item()
                    agent_logs[index]["duration"] = infos[index][
                        "episode_duration"
                    ].item()
                    agent_logs[index]["steps"] = float(
                        infos[index]["episode_length"].item()
                    )

                    # last state action
                    action = action.cpu()
                    agent_logs[index]["last_action_x"] = action[index][0].item()
                    agent_logs[index]["last_action_y"] = action[index][1].item()
                    agent_logs[index]["last_action_z"] = action[index][2].item()
                    agent_logs[index]["last_action_yaw"] = (
                        action[index][3].item() if action.shape[1] >= 4 else 0.0
                    )
                    agent_logs[index]["last_position_x"] = self.env.position[index][
                        0
                    ].item()
                    agent_logs[index]["last_position_y"] = self.env.position[index][
                        1
                    ].item()
                    agent_logs[index]["last_position_z"] = self.env.position[index][
                        2
                    ].item()
                    agent_logs[index]["last_velocity_x"] = self.env.velocity[index][
                        0
                    ].item()
                    agent_logs[index]["last_velocity_y"] = self.env.velocity[index][
                        1
                    ].item()
                    agent_logs[index]["last_velocity_z"] = self.env.velocity[index][
                        2
                    ].item()

                    # path length
                    if agent_logs[index]["position"]: # 確保列表不為空
                        points = th.stack(agent_logs[index]["position"])
                        if len(points) > 1: # 確保至少有兩個點
                            path_length = (points[1:] - points[:-1]).norm(dim=1).sum()
                            agent_logs[index]["path_length"] = path_length.item()

                    # (已保留)
                    if agent_logs[index]["acceleration"]: # 確保列表不為空
                        acceleration = th.tensor(agent_logs[index]["acceleration"])
                        energy_proxy = (acceleration**2).sum() * self.env.dynamics.ctrl_dt
                        agent_logs[index]["total_energy_proxy"] = energy_proxy.item()

                    # control effort
                    if agent_logs[index]["jerk"] and len(agent_logs[index]["jerk"]) > 0: # 確保列表不為空
                        jerk = th.tensor(agent_logs[index]["jerk"])
                        total_control_effort = (jerk**2).sum() * self.env.dynamics.ctrl_dt
                        avg_control_effort = total_control_effort / len(jerk)
                        agent_logs[index]["avg_control_effort"] = avg_control_effort.item()

            if terminated.all():
                break

        batch_stats = {
            "avg_speed": [
                th.tensor(agent["speed"]).mean().item() if agent["speed"] else 0.0
                for agent in agent_logs
            ],
            "max_speed": [
                th.tensor(agent["speed"]).max().item() if agent["speed"] else 0.0
                for agent in agent_logs
            ],
            "max_acceleration": [
                th.tensor(agent["acceleration"]).max().item() if agent["acceleration"] else 0.0
                for agent in agent_logs
            ],
            "avg_yaw_rate": [
                th.tensor(agent["yaw_rate"]).mean().item() if agent["yaw_rate"] else 0.0
                for agent in agent_logs
            ],
            "max_yaw_rate": [
                th.tensor(agent["yaw_rate"]).max().item() if agent["yaw_rate"] else 0.0
                for agent in agent_logs
            ],
            "avg_min_obstacle_distance": [
                th.tensor(agent["obstacle_distance"]).min().item() if agent["obstacle_distance"] else 0.0
                for agent in agent_logs
            ],
            "collision_count": [agent["collision"] for agent in agent_logs],
            "success_count": [agent["success"] for agent in agent_logs],
            "timeout_count": [agent["timeout"] for agent in agent_logs],
            "avg_reward": [agent["avg_reward"] for agent in agent_logs],
            "duration": [agent["duration"] for agent in agent_logs],
            "steps": [agent["steps"] for agent in agent_logs],
            "path_length": [agent["path_length"] for agent in agent_logs],
            "avg_control_effort": [agent["avg_control_effort"] for agent in agent_logs],
            "total_energy_proxy": [agent["total_energy_proxy"] for agent in agent_logs], # (已保留)
            "last_action_x": [agent["last_action_x"] for agent in agent_logs],
            "last_action_y": [agent["last_action_y"] for agent in agent_logs],
            "last_action_z": [agent["last_action_z"] for agent in agent_logs],
            "last_action_yaw": [agent["last_action_yaw"] for agent in agent_logs],
            "last_position_x": [agent["last_position_x"] for agent in agent_logs],
            "last_position_y": [agent["last_position_y"] for agent in agent_logs],
            "last_position_z": [agent["last_position_z"] for agent in agent_logs],
            "last_velocity_x": [agent["last_velocity_x"] for agent in agent_logs],
            "last_velocity_y": [agent["last_velocity_y"] for agent in agent_logs],
            "last_velocity_z": [agent["last_velocity_z"] for agent in agent_logs],
        }
        return batch_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", type=str, default="examples/navigation/eval_cfg/nav_level1.yaml")
    parser.add_argument("--policy_cfg_file", type=str, default="examples/navigation/policy_cfg/small_yaw.yaml")
    parser.add_argument("--weight", type=str, default=None, help="trained weight name")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--num_rollouts", type=int, default=5)
    args = parser.parse_args()
    main(args)