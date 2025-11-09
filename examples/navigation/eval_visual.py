#!/usr/bin/env python3

import os
import cv2
import yaml
import time
import torch as th
import torch.nn.functional as F
import numpy as np
import argparse
from copy import deepcopy
import matplotlib.pyplot as plt
from pylogtools import timerlog

timerlog.timer.print_logs(False)

from depthnav.envs.env_aliases import env_aliases
from depthnav.policies.policy_aliases import policy_aliases
from depthnav.policies.multi_input_policy import MultiInputPolicy
from depthnav.common import observation_to_device
from depthnav.common import std_to_habitat, obs_list2array, rgba2rgb


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
    eval_config["single_env"] = False
    eval_config["scene_kwargs"]["load_geodesics"] = False

    render_kwargs = {}

    # ==========================================================
    # !! 您的自定義渲染設置 (已保留) !!
    # ==========================================================
    eval_config["scene_kwargs"]["render_settings"] = {
        "mode": "fix",
        "view": "top",
        # 1. 添加自定义相机高度，您可以调整这个值 (越小越近)
        "camera_height": 45.0,
        # 2. 开启轨迹绘制功能
        "trajectory": True,
        "sensor_type": "color",
        "resolution": [args.res, args.res],
        "axes": True,
        "object_path": "./datasets/depthnav_dataset/configs/agents/DJI_Mavic_Mini_2.object_config.json",
        "line_width": 2.0,
    }
    # ==========================================================

    env_class = env_aliases[config["env_class"]]
    env = env_class(requires_grad=False, **eval_config)

    policy_class = policy_aliases[config["policy_class"]]
    policy_kwargs = config["policy"]

    # <<< START MODIFICATION 3.11: 傳遞 policy_kwargs (同 train_bptt.py) >>>
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

    def create_name(run_path):
        index = 1
        while True:
            path = f"{run_path}_{index}"
            if not os.path.exists(path + ".mp4"):
                break
            index += 1
        return path

    if args.save_name is None:
        save_name = create_name(args.weight.split(".")[0] + "_eval")

    evaluate = Evaluate(
        env=env,
        policy=policy,
        save_name=save_name,
        save=True,
        show=args.render,
        render_kwargs=render_kwargs,
        res=args.res,
        num_cols=args.num_envs,
    )
    evaluate.run_rollouts(args.num_rollouts)


class Evaluate:
    def __init__(
        self,
        env,
        policy,
        save_name,
        save,
        show=True,
        render_kwargs={},
        res=512,
        num_rows=1,
        num_cols=4,
    ):
        assert num_rows * num_cols == env.num_envs

        self.env = env
        self.policy = policy

        self.save_name = save_name
        self.save = save
        self.show = show
        self.render_kwargs = render_kwargs
        self.res = res
        self.num_rows = num_rows
        self.num_cols = num_cols

        self.render_grid = np.zeros((num_rows * res, num_cols * res, 3), dtype=np.uint8)
        self.collided = th.zeros(self.env.num_envs, dtype=th.bool)
        self.first_collision = th.zeros(self.env.num_envs, dtype=th.bool)
        self.done = th.zeros(self.env.num_envs, dtype=th.bool)
        self.all_frames = []

    @th.no_grad()
    def run_rollouts(self, num_rollouts=1):
        for _ in range(num_rollouts):
            self.env.reset()
            self.render_kwargs["points"] = th.cat(
                [self.env.position.unsqueeze(1), self.env.target.unsqueeze(1)], dim=1
            )
            self.single_rollout()

        if self.save:
            self.save_video()
        self.env.close()
        cv2.destroyAllWindows()

    def save_video(self):
        print("Processing video")
        fps = int(1.0 / self.env.dynamics.ctrl_dt)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4
        video_width = self.num_cols * self.res
        video_height = self.num_rows * self.res
        out = cv2.VideoWriter(
            f"{self.save_name}.mp4", fourcc, fps, (video_width, video_height)
        )
        for img in self.all_frames:
            out.write(img)
        out.release()
        print(f"Saved video to {self.save_name}.mp4")

    @th.no_grad()
    def single_rollout(self):
        self.render_grid = np.zeros(
            (self.num_rows * self.res, self.num_cols * self.res, 3), dtype=np.uint8
        )
        self.collided = th.zeros(self.env.num_envs, dtype=th.bool)
        self.first_collision = th.zeros(self.env.num_envs, dtype=th.bool)
        self.done = th.zeros(self.env.num_envs, dtype=th.bool)
        
        # <<< START MODIFICATION 3.19: 初始化可視化評估時的隱藏狀態元組 >>>
        if hasattr(self.policy, "is_temporal_attention") and self.policy.is_temporal_attention:
            K, H = self.policy.attention_history_shape
            h_t = th.zeros((self.env.num_envs, H), device=self.policy.device)
            history_buffer = th.zeros((self.env.num_envs, K, H), device=self.policy.device)
            latent_tuple = (h_t, history_buffer)
        elif self.policy.is_recurrent:
            latent_state = th.zeros(
                (self.env.num_envs, self.policy.latent_dim), device=self.policy.device
            )
        # <<< END MODIFICATION 3.19 >>>
        
        while True:
            start = time.time()
            obs = observation_to_device(self.env.get_observation(), self.policy.device)
            
            # <<< START MODIFICATION 3.20: 更新可視化評估時的 Policy 調用 >>>
            if type(self.policy) == MultiInputPolicy:
                if hasattr(self.policy, "is_temporal_attention") and self.policy.is_temporal_attention:
                    action, latent_tuple = self.policy(obs, latent_tuple)
                elif self.policy.is_recurrent:
                    action, latent_state = self.policy(obs, latent_state)
                else:
                    action = self.policy(obs)
            else:
                action = self.policy(obs["state"])
            # <<< END MODIFICATION 3.20 >>>
            
            obs, reward, terminated, infos = self.env.step(action, is_test=True)
            # 确保terminated张量与模型在同一设备上
            terminated = terminated.to(self.policy.device)
            self.first_collision = self.env.is_collision & ~self.collided
            self.collided = self.collided | self.env.is_collision

            if self.show or self.save:
                render_list = rgba2rgb(
                    self.env.scene_manager.render(**self.render_kwargs)
                )
                self.render(render_list, terminated)

                if self.show:
                    cv2.imshow("render cams", self.render_grid)
                    elapsed = time.time() - start
                    wait_ms = max(int(1000 * (self.env.dynamics.ctrl_dt - elapsed)), 1)
                    cv2.waitKey(wait_ms)
            
            # <<< START MODIFICATION 3.21: 在可視化時重置隱藏狀態元組(掩碼) >>>
            # 確保terminated在正確的設備上
            terminated = terminated.to(self.policy.device)
            if hasattr(self.policy, "is_temporal_attention") and self.policy.is_temporal_attention:
                h_t, history_buffer = latent_tuple
                h_t = h_t * ~terminated.unsqueeze(1)
                history_buffer = history_buffer * ~terminated.unsqueeze(1).unsqueeze(2)
                latent_tuple = (h_t, history_buffer)
            elif self.policy.is_recurrent:
                 latent_state = latent_state * ~terminated.unsqueeze(1)
            # <<< END MODIFICATION 3.21 >>>

            if terminated.all():
                break

    # (您的自定義 render 方法已保留)
    def render(self, render_list, dones):
        for i, (render, done) in enumerate(zip(render_list, dones)):
            row = i // self.num_cols
            col = i % self.num_cols

            if self.first_collision[i]:
                img = self.tint_red(render)
                self.render_grid[
                    row * self.res : (row + 1) * self.res,
                    col * self.res : (col + 1) * self.res,
                ] = img

            if not done:
                self.render_grid[
                    row * self.res : (row + 1) * self.res,
                    col * self.res : (col + 1) * self.res,
                ] = render
        self.all_frames.append(np.copy(self.render_grid))

    # (您的自定義 tint_red 方法已保留)
    @staticmethod
    def tint_red(image, alpha=0.3):
        red_overlay = np.zeros_like(image)
        red_overlay[..., 2] = 255
        tinted = (1 - alpha) * image + alpha * red_overlay
        return np.clip(tinted, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", type=str, default="examples/navigation/eval_cfg/nav_level1.yaml")
    parser.add_argument("--policy_cfg_file", type=str, default="examples/navigation/policy_cfg/small_yaw.yaml")
    parser.add_argument("--weight", type=str, default=None, help="trained weight name")
    parser.add_argument("--render", action="store_true", help="Show observations")
    parser.add_argument("--save_name", type=str, default=None)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_rollouts", type=int, default=1)
    parser.add_argument("--res", type=int, default=720) # 提高默认分辨率
    args = parser.parse_args()
    main(args)