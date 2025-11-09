#!/usr/bin/env python3
import os
import sys
import time
from typing import Any, Union, Optional, Type, Dict, TypeVar, ClassVar, List

import torch as th
import numpy as np
from tqdm import tqdm

from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common import logger

from pylogtools import timerlog

timerlog.timer.print_logs(False)

from .debug import check_none_parameters, get_network_statistics, compute_gradient_norm
from .mlp_policy import MlpPolicy
from ..common import observation_to_device, rgba2rgb, ExitCode
from depthnav.scripts.eval_logger import Evaluate


class BPTT:
    """
    Back Propagation Through Time (BPTT).
    Pass in an env, eval_env, and policy
    Call learn() and get back the trained policy
    Gradients are computed across multiple timesteps and loss is computed at final time step.
    """

    def __init__(
        self,
        policy: Union[MlpPolicy],
        env: GymEnv,
        eval_envs: List[GymEnv] = None,
        eval_csvs: List[str] = None,
        learning_rate_init: float = 3e-4,
        learning_rate_final: float = 0.0,
        weight_decay: float = 0.0,
        run_name: Optional[str] = "BPTT",
        logging_dir: Optional[str] = "./saved",
        horizon: int = 32,
        gamma: float = 0.99,
        iterations: int = 1000,
        log_interval: int = 1,
        early_stop_reward_threshold: float = -th.inf,
        checkpoint_interval: int = 1000,
        device: Optional[Union[str, th.device]] = "cpu",
    ):
        self.logging_dir = os.path.abspath(logging_dir)
        self.run_path = self._create_run_path(run_name)

        self.env = env
        self.eval_envs = eval_envs or []
        self.eval_csvs = eval_csvs or []
        assert len(self.eval_envs) == len(self.eval_csvs)
        self.device = th.device(device)
        self.policy = policy.to(self.device)

        # training parameters
        self.iterations = iterations
        self.horizon = horizon
        self.gamma = gamma
        self.optimizer = th.optim.AdamW(
            self.policy.parameters(), lr=learning_rate_init, weight_decay=weight_decay
        )
        self.lr_schedule = th.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=iterations, eta_min=learning_rate_final
        )

        # training buffers used for logging
        self.log_interval = log_interval
        self.early_stop_reward_threshold = early_stop_reward_threshold
        self.checkpoint_interval = checkpoint_interval

        self.whitelisted_tensorboard_keys = [
            "avg_reward",
            "success_rate",
            "collision_rate",
            "timeout_rate",
            "avg_steps",
            "avg_path_length",
            "avg_speed",
            "max_speed",
            "max_acceleration",
            "max_yaw_rate",
            "avg_min_obstacle_distance",
            "total_energy_proxy", # <<< 確保能耗指標被記錄 >>>
            "reward_avoid", # <<< 確保避障獎勵被記錄 >>>
        ]

        self.whitelisted_csv_keys = [
            "success_rate",
            "collision_rate",
            "timeout_rate",
            "num_trials",
            "avg_speed",
            "max_speed",
            "avg_path_length",
            "avg_control_effort",
            "total_energy_proxy", # <<< 確保能耗指標被記錄 >>>
            "max_acceleration",
            "avg_yaw_rate",
            "max_yaw_rate",
            "avg_reward",
            "avg_min_obstacle_distance",
            "scene",
        ]

    def _create_run_path(self, run_name="BPTT"):
        index = 1
        run_name = run_name or "BPTT"
        while True:
            path = f"{self.logging_dir}/{run_name}_{index}"
            if not os.path.exists(path):
                break
            index += 1
        return path

    def save(self, filepath=None):
        filepath = filepath or self.run_path + ".pth"
        print(f"Saving to {filepath}")
        self.policy.save(filepath)

    def learn(self, render=False, start_iter=0) -> ExitCode:
        """
        Train policy using BPTT, return True when finished training
        """
        assert self.horizon >= 1, "horizon must be greater than 1"
        assert self.env is not None

        self.policy.train()
        check_none_parameters(self.policy)
        self.start_time = time.time_ns()
        exit_code = ExitCode.ERROR
        self._logger = logger.configure(self.run_path, ["stdout", "tensorboard"])

        try:
            self.env.reset()
            episode_steps = 0
            
            # <<< START MODIFICATION 3.12: 初始化隱藏狀態元組 >>>
            if hasattr(self.policy, "is_temporal_attention") and self.policy.is_temporal_attention:
                K, H = self.policy.attention_history_shape
                h_t = th.zeros((self.env.num_envs, H), device=self.policy.device)
                history_buffer = th.zeros((self.env.num_envs, K, H), device=self.policy.device)
                latent_tuple = (h_t, history_buffer)
            elif self.policy.is_recurrent:
                latent_state = th.zeros(
                    (self.env.num_envs, self.policy.latent_dim), device=self.policy.device
                )
            # <<< END MODIFICATION 3.12 >>>

            for iter in tqdm(range(self.iterations)):
                timerlog.timer.tic("iteration")

                self.policy.train()
                loss = 0.0
                fps_start = time.time_ns()
                discount_factor = th.ones(
                    self.env.num_envs, dtype=th.float32, device=self.device
                )

                # reset agents once they have max_episode_steps experience
                if episode_steps >= self.env.max_episode_steps:
                    self.env.reset()
                    episode_steps = 0
                    
                    # <<< START MODIFICATION 3.12.1: 重置隱藏狀態元組 >>>
                    if hasattr(self.policy, "is_temporal_attention") and self.policy.is_temporal_attention:
                        h_t.zero_()
                        history_buffer.zero_()
                        latent_tuple = (h_t, history_buffer)
                    elif self.policy.is_recurrent:
                        latent_state.zero_()
                    # <<< END MODIFICATION 3.12.1 >>>

                # rollout policy over horizon steps
                for t in range(self.horizon):
                    obs = self.env.get_observation()
                    obs = observation_to_device(obs, self.policy.device)
                    
                    # <<< START MODIFICATION 3.13: 更新 Policy 調用 >>>
                    if type(self.policy) == MlpPolicy:
                        actions = self.policy(obs["state"])
                    else:
                        if hasattr(self.policy, "is_temporal_attention") and self.policy.is_temporal_attention:
                            actions, latent_tuple = self.policy(obs, latent_tuple)
                        elif self.policy.is_recurrent:
                            actions, latent_state = self.policy(obs, latent_state)
                        else:
                            actions = self.policy(obs)
                    # <<< END MODIFICATION 3.13 >>>

                    # step
                    obs, reward, done, info = self.env.step(actions, is_test=False)
                    reward = reward.to(self.device)
                    done = done.to(self.device).to(th.bool)
                    loss = loss + -1.0 * reward * discount_factor

                    # if done, reset discount factor and latents
                    discount_factor = discount_factor * self.gamma * ~done + done
                    
                    # <<< START MODIFICATION: 修复隐藏状态的in-place操作 >>>
                    # 修复隐藏状态元组的重置操作，避免in-place修改
                    if hasattr(self.policy, "is_temporal_attention") and self.policy.is_temporal_attention:
                        h_t, history_buffer = latent_tuple
                        # 使用非in-place操作重置已完成episode的隐藏状态
                        done_mask = done.unsqueeze(1)
                        h_t = th.where(done_mask, th.zeros_like(h_t), h_t)
                        done_mask_history = done.unsqueeze(1).unsqueeze(2)
                        history_buffer = th.where(done_mask_history, th.zeros_like(history_buffer), history_buffer)
                        latent_tuple = (h_t, history_buffer)
                    elif self.policy.is_recurrent:
                        # 使用非in-place操作重置已完成episode的隐藏状态
                        done_mask = done.unsqueeze(1)
                        latent_state = th.where(done_mask, th.zeros_like(latent_state), latent_state)
                    # <<< END MODIFICATION >>>
                    
                    # <<< START MODIFICATION 3.15: 移除錯誤的 detach >>>
                    # (您在 BPTT 循環內部添加的 detach() 調用已被移除)
                    # (例如: actions = actions.detach(), reward = reward.detach(), 等)
                    # (這些調用會切斷 BPTT 的計算圖，必須移除)
                    # <<< END MODIFICATION 3.15 >>>

                episode_steps += self.horizon
                total_steps = self.env.num_envs * self.horizon
                time_elapsed = max(
                    (time.time_ns() - fps_start) / 1e9, sys.float_info.epsilon
                )
                fps = int(total_steps / time_elapsed)

                # backprop
                loss = loss / self.horizon  # average across the rollout
                loss = loss.mean()  # average across the batch
                self.optimizer.zero_grad()
                loss.backward()

                # log total gradient magnitude and clip to prevent exploding gradients
                max_norm = 5.0
                grad_norm = th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), max_norm=max_norm
                )
                # (移除了 print 語句以保持日誌清潔，您可以加回來)
                # print(f"grad norm = {grad_norm:.4f}")

                # update policy
                self.optimizer.step()
                self.lr_schedule.step()

                # detach gradients properly (在 backward() 之後)
                self.env.detach()
                
                # <<< START MODIFICATION: 修复隐藏状态的detach操作 >>>
                # 修复隐藏状态元组的detach操作，避免in-place修改
                if hasattr(self.policy, "is_temporal_attention") and self.policy.is_temporal_attention:
                    h_t, history_buffer = latent_tuple
                    h_t = h_t.detach()
                    history_buffer = history_buffer.detach()
                    latent_tuple = (h_t, history_buffer)
                elif self.policy.is_recurrent:
                    latent_state = latent_state.detach()
                # <<< END MODIFICATION >>>

                # (保留您添加的正確的 detach)
                loss = loss.detach()
                discount_factor = discount_factor.detach()
                
                # Clear CUDA cache periodically to prevent memory accumulation
                if iter % 50 == 0 and th.cuda.is_available():
                    th.cuda.empty_cache()
                    
                timerlog.timer.toc("iteration")

                timerlog.timer.tic("eval")
                if iter % self.log_interval == 0:
                    self.policy.eval()
                    for i, (eval_env, csv_file) in enumerate(
                        zip(self.eval_envs, self.eval_csvs)
                    ):
                        # evaluate policy in eval_env with multiple rollouts
                        e = Evaluate(eval_env, self.policy)
                        index = start_iter + iter
                        df = e.run_rollouts(
                            num_rollouts=5, run_name=index, render=render
                        )

                        # add a column to log the scene
                        # <<< MODIFICATION 3.17: 添加 hasattr 檢查 >>>
                        if hasattr(self.env, "scene_manager") and self.env.scene_manager is not None and hasattr(self.env.scene_manager, "scene_path"):
                             df["scene"] = os.path.basename(
                                 self.env.scene_manager.scene_path
                             )
                        else:
                             df["scene"] = "N/A" # 處理沒有 scene_manager 的情況
                        # <<< END MODIFICATION 3.17 >>>


                        # log the df to tensorboard
                        basename = os.path.basename(csv_file).split(".")[0]
                        self.df_to_tensorboard(self._logger, df, prefix=basename)

                        # save df to csv
                        write_header = not os.path.exists(csv_file)
                        df.to_csv(
                            csv_file,
                            float_format="%.3f",
                            mode="a",
                            header=write_header,
                            columns=self.whitelisted_csv_keys,
                        )
                        print(f"wrote stats to {csv_file}")

                        # check if we should early stop
                        last_avg_reward = df["avg_reward"].iloc[-1]
                        if last_avg_reward < self.early_stop_reward_threshold:
                            print("REWARD FELL BELOW EARLY STOP THRESHOLD")
                            print(f"Last reward: {last_avg_reward}")
                            exit_code = ExitCode.EARLY_STOP
                            for eval_env in self.eval_envs:
                                eval_env.close()
                            self.env.close()
                            return exit_code

                    # log and dump iter to tensorboard
                    self._logger.record(
                        "train/learning_rate", self.lr_schedule.get_last_lr()[0]
                    )
                    self._logger.record("train/loss", float(loss))
                    self._logger.record("train/grad_norm", float(grad_norm))
                    self._logger.record("train/steps_per_second", fps)
                    self._logger.dump(start_iter + iter)

                    timerlog.timer.toc("eval")
                    timerlog.timer.print_summary()
                    timerlog.timer.clear_history()
                if iter > 0 and iter % self.checkpoint_interval == 0:
                    self.save(self.run_path + "_iteration_" + str(iter) + ".pth")

            exit_code = ExitCode.SUCCESS
        except KeyboardInterrupt:
            self.save(self.run_path + "_iteration_" + str(iter) + ".pth")
            exit_code = ExitCode.KEYBOARD_INTERRUPT
        finally:
            for eval_env in self.eval_envs:
                eval_env.close()
            self.env.close()
        return exit_code

    def df_to_tensorboard(self, logger, df, prefix="eval"):
        keys = df.columns.tolist()
        for key in keys:
            if key not in self.whitelisted_tensorboard_keys:
                continue
            logger.record(f"{prefix}/{key}", df[key].iloc[-1])