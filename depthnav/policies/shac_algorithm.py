#
# 完整文件: depthNav/policies/shac_algorithm.py
# (这是 qianzhong-chen/grad_nav/algorithms/gradnav.py 的适配版)
#

import sys, os
import time
import numpy as np
import copy
import torch as th
import torch.nn as nn
import math
import yaml
from torch.nn.utils.clip_grad import clip_grad_norm_

# 导入 depthNav 的组件
from depthnav.envs.navigation_env import NavigationEnv
from depthnav.policies.multi_input_policy import MultiInputPolicy
from depthnav.common import observation_to_device
from depthnav.scripts.eval_logger import Evaluate # 用于评估

# 导入我们新添加的组件
from depthnav.policies.critic import CriticMLP
from depthnav.policies.running_mean_std import RunningMeanStd
from depthnav.policies.critic_dataset import CriticDataset

# 导入 stable_baselines3 的日志工具 (depthNav 已依赖)
from stable_baselines3.common import logger
from tqdm import tqdm
# 文件: depthnav/policies/shac_algorithm.py

# ... (所有 import 保持不变) ...

class SHAC:
    def __init__(self, 
                 policy: MultiInputPolicy,
                 policy_kwargs: dict,
                 env: NavigationEnv,
                 eval_envs: list = None,
                 eval_csvs: list = None,
                 run_name: str = "SHAC",
                 logging_dir: str = "./logs",
                 
                 # --- 关键修复：在这里添加 'iterations' ---
                 iterations: int = 10000, # 对应 grad_nav 的 max_epochs
                 # --- 结束修复 ---
                 
                 # --- 从 train_shac 配置中读取 (其他参数) ---
                 actor_learning_rate: float = 1e-4,
                 critic_learning_rate: float = 1e-4,
                 lr_schedule: str = 'cosine',
                 weight_decay: float = 0.01,
                 steps_num: int = 32,
                 gamma: float = 0.99,
                 lambda_val: float = 0.95, 
                 critic_iterations: int = 16,
                 num_batch: int = 4,
                 grad_norm: float = 1.0,
                 log_interval: int = 100,
                 checkpoint_interval: int = 1000,
                 early_stop_reward_threshold: float = -150.0,
                 device: str = "cuda",
                 # ... (LPF_train, obs_rms 等其他参数保持不变) ...
                 LPF_train: bool = False,
                 LPF_val: float = 0.5,
                 obs_rms: bool = True,
                 ret_rms: bool = False,
                 target_critic_alpha: float = 0.2,
                 betas: list = [0.7, 0.95]
                 ):
        
        self.env = env
        self.actor = policy 
        self.device = th.device(device)
        
        # ... (self.num_envs, self.num_obs, ... 等不变) ...
        self.num_envs = self.env.num_envs
        self.num_obs = self.env.observation_space["state"].shape[0] 
        self.num_privilege_obs = self.env.privileged_observation_space.shape[0]
        self.num_actions = self.env.action_space.shape[0]
        self.max_episode_length = self.env.max_episode_steps

        # 算法超参数
        self.gamma = gamma
        self.lam = lambda_val
        self.critic_method = 'td-lambda' 
        self.steps_num = steps_num
        
        # --- 关键修复：正确设置 self.iterations ---
        self.max_epochs = self.iterations = iterations
        # (我们删除了之前错误的 self.max_epochs = ... 1e10 的那一行)
        # --- 结束修复 ---

        self.actor_lr = actor_learning_rate
        self.critic_lr = critic_learning_rate
        self.lr_schedule = lr_schedule
        self.init_actor_lr = self.actor_lr
        self.init_critic_lr = self.critic_lr
        self.target_critic_alpha = target_critic_alpha
        self.critic_iterations = critic_iterations
        self.num_batch = num_batch
        self.batch_size = self.num_envs * self.steps_num // self.num_batch
        self.truncate_grad = True 
        self.grad_norm = grad_norm

        # ... (日志, LPF, Critic, 优化器, 缓冲区, ... 的定义保持不变) ...
        
        # 日志和评估
        self.logging_dir = os.path.abspath(logging_dir)
        self.run_path = self._create_run_path(run_name)
        self._logger = logger.configure(self.run_path, ["stdout", "tensorboard"])
        self.eval_envs = eval_envs or []
        self.eval_csvs = eval_csvs or []
        self.log_interval = log_interval
        self.checkpoint_interval = checkpoint_interval
        self.early_stop_reward_threshold = early_stop_reward_threshold
        self.whitelisted_tensorboard_keys = [
            "avg_reward", "success_rate", "collision_rate", "timeout_rate", 
            "avg_steps", "avg_path_length", "avg_speed", "max_speed",
            "max_acceleration", "max_yaw_rate", "avg_min_obstacle_distance",
        ]

        # LPF (可选, 来自 grad_nav)
        self.train_LPF = LPF_train
        if self.train_LPF:
            self.train_r_filter = BatchLowPassFilter(alpha=LPF_val, batch_size=self.num_envs)
            self.train_p_filter = BatchLowPassFilter(alpha=LPF_val, batch_size=self.num_envs)
            self.train_y_filter = BatchLowPassFilter(alpha=LPF_val, batch_size=self.num_envs)
            self.train_thrust_filter = BatchLowPassFilter(alpha=LPF_val, batch_size=self.num_envs)

        # Critic (来自 grad_nav)
        critic_network_cfg = policy_kwargs.get("critic_mlp", {"units": [192, 192], "activation": "leaky_relu"})
        self.critic = CriticMLP(
            privilege_obs_dim=self.num_privilege_obs,
            cfg_network={'critic_mlp': critic_network_cfg},
            device=self.device
        )
        self.target_critic = copy.deepcopy(self.critic)

        # 优化器
        self.actor_optimizer = th.optim.AdamW(self.actor.parameters(), betas = betas, lr = self.actor_lr, weight_decay=weight_decay)
        self.critic_optimizer = th.optim.AdamW(self.critic.parameters(), betas = betas, lr = self.critic_lr, weight_decay=weight_decay)

        # 缓冲区
        self.obs_buf = th.zeros((self.steps_num, self.num_envs, self.num_obs), dtype = th.float32, device = self.device)
        self.privilege_obs_buf = th.zeros((self.steps_num, self.num_envs, self.num_privilege_obs), dtype = th.float32, device = self.device)
        self.rew_buf = th.zeros((self.steps_num, self.num_envs), dtype = th.float32, device = self.device)
        self.done_mask = th.zeros((self.steps_num, self.num_envs), dtype = th.float32, device = self.device)
        self.next_values = th.zeros((self.steps_num, self.num_envs), dtype = th.float32, device = self.device)
        self.target_values = th.zeros((self.steps_num, self.num_envs), dtype = th.float32, device = self.device)
        
        self.iter_count = 0
        self.step_count = 0
        self.best_policy_loss = -np.inf # <-- 修复：BPTT 用的是 loss(越小越好)，我们用 reward (越大越好)
        self.actor_loss = np.inf
        self.value_loss = np.inf
        
        # 内部状态
        self.episode_loss = th.zeros(self.num_envs, dtype = th.float32, device = self.device)
        self.episode_discounted_loss = th.zeros(self.num_envs, dtype = th.float32, device = self.device)
        self.episode_gamma = th.ones(self.num_envs, dtype = th.float32, device = self.device)
        self.episode_length = th.zeros(self.num_envs, dtype = int, device = self.device)
        
        # RMS (可选, 来自 grad_nav)
        self.obs_rms = None
        if obs_rms:
            self.obs_rms = RunningMeanStd(shape = (self.num_obs), device = self.device)
            self.privilege_obs_rms = RunningMeanStd(shape = (self.num_privilege_obs), device = self.device)
        self.ret_rms = None
        if ret_rms:
            self.ret_rms = RunningMeanStd(shape = (), device = self.device)
            
    # ... (所有其他函数: _create_run_path, save, load, learn, compute_actor_loss, etc. 保持不变) ...
    # (确保 'learn' 循环使用 'self.iterations' 或 'self.max_epochs')
    # (确保 'learn' 方法中检查最佳奖励的逻辑是 > self.best_policy_loss)
    
    def _create_run_path(self, run_name="SHAC"):
        index = 1
        run_name = run_name or "SHAC"
        while True:
            path = f"{self.logging_dir}/{run_name}_{index}"
            if not os.path.exists(path):
                break
            index += 1
        os.makedirs(path, exist_ok=True) # 确保目录被创建
        return path

    def save(self, filepath=None):
        filepath = filepath or os.path.join(self.run_path, "default_policy.pth") # 确保在 run_path 目录内
        print(f"Saving SHAC checkpoint to {filepath}")
        # 保存 Actor (MultiInputPolicy) 和 RMS
        th.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'obs_rms': self.obs_rms,
            'privilege_obs_rms': self.privilege_obs_rms,
        }, filepath)
        
    def load(self, filepath):
        print(f"Loading SHAC checkpoint from {filepath}")
        checkpoint = th.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        # 兼容 BPTT 权重 (只有 actor)
        if 'critic_state_dict' in checkpoint:
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.target_critic = copy.deepcopy(self.critic)
            print("Loaded Critic weights.")
        else:
            print("Warning: Critic weights not found in checkpoint. Initializing Critic from scratch.")
        self.obs_rms = checkpoint.get('obs_rms')
        self.privilege_obs_rms = checkpoint.get('privilege_obs_rms')
        if self.obs_rms:
            print("Loaded RMS stats.")

    def learn(self, render=False, start_iter=0): # 兼容 bptt_algorithm.py
        
        self.initialize_env()
        self.episode_loss = th.zeros(self.num_envs, dtype = th.float32, device = self.device)
        self.episode_discounted_loss = th.zeros(self.num_envs, dtype = th.float32, device = self.device)
        self.episode_length = th.zeros(self.num_envs, dtype = int, device = self.device)
        self.episode_gamma = th.ones(self.num_envs, dtype = th.float32, device = self.device)
        
        # 兼容 bptt_algorithm.py 的 latent state
        latent_state = th.zeros(
            (self.env.num_envs, self.actor.latent_dim), device=self.actor.device
        )
        
        def actor_closure():
            self.actor_optimizer.zero_grad()
            
            # --- 核心：计算 Actor 损失 ---
            actor_loss, latent_state_out = self.compute_actor_loss(latent_state)
            
            actor_loss.backward()

            with th.no_grad():
                # --- 修复：检查参数是否有 grad ---
                valid_params = [p for p in self.actor.parameters() if p.grad is not None]
                if valid_params:
                    self.grad_norm_before_clip = th.nn.utils.clip_grad_norm_(valid_params, self.grad_norm)
                else:
                    self.grad_norm_before_clip = th.tensor(0.0, device=self.device)
                # --- 结束修复 ---
            
            return actor_loss, latent_state_out

        # --- 主训练循环 (来自 grad_nav.py) ---
        for epoch in tqdm(range(start_iter, self.iterations)): # 确保从 start_iter 开始
            time_start_epoch = time.time()

            # --- 学习率调度 (来自 grad_nav.py) ---
            if self.lr_schedule == 'linear':
                actor_lr = (1e-5 - self.init_actor_lr) * float(epoch / self.iterations) + self.init_actor_lr
                critic_lr = (1e-5 - self.init_critic_lr) * float(epoch / self.iterations) + self.init_critic_lr
            elif self.lr_schedule == 'cosine':
                actor_lr = 1e-5 + (self.init_actor_lr - 1e-5) * 0.5 * (1 + math.cos(math.pi * epoch / self.iterations))
                critic_lr = 1e-5 + (self.init_critic_lr - 1e-5) * 0.5 * (1 + math.cos(math.pi * epoch / self.iterations))
            else:
                actor_lr = self.init_actor_lr
                critic_lr = self.init_critic_lr

            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] = actor_lr
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] = critic_lr

            # --- 训练 Actor ---
            actor_loss, latent_state = actor_closure()
            self.actor_optimizer.step()
            
            # 分离 latent_state 的梯度
            latent_state = latent_state.detach()
            
            # --- 训练 Critic ---
            with th.no_grad():
                self.compute_target_values()
                dataset = CriticDataset(self.batch_size, self.privilege_obs_buf, self.target_values, shuffle=True, drop_last = False)

            self.value_loss = 0.
            for j in range(self.critic_iterations):
                total_critic_loss = 0.
                batch_cnt = 0
                for i in range(len(dataset)):
                    batch_sample = dataset[i]
                    self.critic_optimizer.zero_grad()
                    training_critic_loss = self.compute_critic_loss(batch_sample)
                    training_critic_loss.backward()
                    
                    if self.truncate_grad:
                        clip_grad_norm_(self.critic.parameters(), self.grad_norm)
                    self.critic_optimizer.step()
                    total_critic_loss += training_critic_loss.item()
                    batch_cnt += 1
                
                self.value_loss = (total_critic_loss / batch_cnt)

            # --- 更新 Target Critic (来自 grad_nav.py) ---
            with th.no_grad():
                alpha = self.target_critic_alpha
                for param, param_targ in zip(self.critic.parameters(), self.target_critic.parameters()):
                    param_targ.data.mul_(alpha)
                    param_targ.data.add_((1. - alpha) * param.data)
            
            self.iter_count += 1
            time_end_epoch = time.time()
            fps = self.steps_num * self.num_envs / (time_end_epoch - time_start_epoch)

            # --- 日志记录 (来自 bptt_algorithm.py 和 grad_nav.py) ---
            if epoch % self.log_interval == 0:
                self.actor.eval()
                
                # --- 使用 depthNav 的评估器 ---
                all_rewards = []
                for i, (eval_env, csv_file) in enumerate(
                    zip(self.eval_envs, self.eval_csvs)
                ):
                    e = Evaluate(eval_env, self.actor)
                    index = epoch # 使用当前 epoch
                    df = e.run_rollouts(
                        num_rollouts=5, run_name=index, render=render
                    )
                    
                    try:
                        # 尝试获取场景路径
                        scene_path = self.env.scene_manager.scene_paths[0]
                        df["scene"] = os.path.basename(scene_path)
                    except Exception:
                        df["scene"] = "unknown"

                    
                    # 记录到 Tensorboard
                    basename = os.path.basename(csv_file).split(".")[0]
                    self.df_to_tensorboard(self._logger, df, prefix=basename)
                    
                    # 写入 CSV
                    write_header = not os.path.exists(csv_file)
                    df.to_csv(
                        csv_file,
                        float_format="%.3f",
                        mode="a",
                        header=write_header,
                    )
                    print(f"Wrote eval stats to {csv_file}")
                    
                    last_avg_reward = df["avg_reward"].iloc[-1]
                    all_rewards.append(last_avg_reward)
                    
                self.actor.train() # 切换回训练模式
                
                # --- 检查是否提前停止 ---
                if all_rewards:
                    mean_eval_reward = np.mean(all_rewards)
                    
                    # --- 修复：BPTT 的 early stop 是负数 (loss)，这里是 reward (正数) ---
                    # 假设 early_stop_reward_threshold 是 -150
                    # 我们需要 reward > -150
                    # 如果 mean_eval_reward < -150，则停止
                    if mean_eval_reward < self.early_stop_reward_threshold:
                        print(f"REWARD {mean_eval_reward:.2f} FELL BELOW EARLY STOP THRESHOLD {self.early_stop_reward_threshold}")
                        return ExitCode.EARLY_STOP
                    
                    # --- 修复：BPTT 存的是 loss (越小越好)，我们存 reward (越大越好) ---
                    if mean_eval_reward > self.best_policy_loss: 
                        print(f"Saving best policy with reward {mean_eval_reward:.2f}")
                        self.save(os.path.join(self.run_path, "best_policy.pth"))
                        self.best_policy_loss = mean_eval_reward

                # 记录训练指标
                self._logger.record("train/actor_learning_rate", actor_lr)
                self._logger.record("train/critic_learning_rate", critic_lr)
                self._logger.record("train/actor_loss", self.actor_loss)
                self._logger.record("train/critic_loss", self.value_loss)
                self._logger.record("train/grad_norm", self.grad_norm_before_clip.item())
                self._logger.record("train/steps_per_second", fps)
                self._logger.dump(epoch) # 使用当前 epoch

            if epoch > 0 and epoch % self.checkpoint_interval == 0:
                self.save(os.path.join(self.run_path, f"policy_iter_{epoch}.pth"))
        
        return ExitCode.SUCCESS
        
    def compute_actor_loss(self, latent_state_in):
        # (来自 grad_nav/algorithms/gradnav.py -> compute_actor_loss)
        
        rew_acc = th.zeros((self.steps_num + 1, self.num_envs), dtype = th.float32, device = self.device)
        gamma = th.ones(self.num_envs, dtype = th.float32, device = self.device)
        next_values = th.zeros((self.steps_num + 1, self.num_envs), dtype = th.float32, device = self.device)

        actor_loss = th.tensor(0., dtype = th.float32, device = self.device)
        
        # 复制 RMS
        obs_rms = None
        privilege_obs_rms = None
        ret_var = 1.0
        
        if self.obs_rms is not None:
            with th.no_grad():
                obs_rms = copy.deepcopy(self.obs_rms)
                privilege_obs_rms = copy.deepcopy(self.privilege_obs_rms)
        if self.ret_rms is not None:
            with th.no_grad():
                ret_var = self.ret_rms.var.clone() + 1e-5
        
        # 重新初始化环境以切断梯度
        self.initialize_env()
        obs_dict = self.env.get_observation()
        obs = observation_to_device(obs_dict, self.device)
        
        # 处理 RMS
        if self.obs_rms is not None:
            with th.no_grad():
                self.obs_rms.update(obs["state"])
                self.privilege_obs_rms.update(obs["privileged_state"])
            obs["state"] = obs_rms.normalize(obs["state"])
            obs["privileged_state"] = privilege_obs_rms.normalize(obs["privileged_state"])
            
        # 兼容 bptt 的 recurrent state
        latent_state = latent_state_in

        for i in range(self.steps_num):
            with th.no_grad():
                obs["state"] = th.nan_to_num(obs["state"], nan=0.0, posinf=1e3, neginf=-1e3)
                obs["privileged_state"] = th.nan_to_num(obs["privileged_state"], nan=0.0, posinf=1e3, neginf=-1e3)
                # 修复：确保 obs_buf 维度匹配
                self.obs_buf[i] = obs["state"][:, :self.num_obs].clone() 
                self.privilege_obs_buf[i] = obs["privileged_state"].clone()

            # --- 关键的 Actor-Env 步骤 ---
            actions, latent_state = self.actor(obs, latent_state)
            
            # (可选的 LPF)
            if self.train_LPF:
                actions_filtered = th.stack([
                    self.train_r_filter.filter(actions[:, 0]),
                    self.train_p_filter.filter(actions[:, 1]),
                    self.train_thrust_filter.filter(actions[:, 2]) 
                ], dim=1)
                obs_dict_next, rew, done, info = self.env.step(actions_filtered, is_test=True) # 用 is_test=True 模式
            else:
                obs_dict_next, rew, done, info = self.env.step(actions, is_test=True) # 用 is_test=True 模式
            # -----------------------------

            obs_next = observation_to_device(obs_dict_next, self.device)
            rew = rew.to(self.device)
            done = done.to(self.device)
            
            with th.no_grad():
                raw_rew = rew.clone()
            
            # (可选的 RMS)
            if self.obs_rms is not None:
                with th.no_grad():
                    self.obs_rms.update(obs_next["state"])
                    self.privilege_obs_rms.update(obs_next["privileged_state"])
                obs_next["state"] = obs_rms.normalize(obs_next["state"])
                obs_next["privileged_state"] = privilege_obs_rms.normalize(obs_next["privileged_state"])
            
            if self.ret_rms is not None:
                with th.no_grad():
                    self.ret = self.ret * self.gamma + rew
                    self.ret_rms.update(self.ret)
                rew = rew / th.sqrt(ret_var)


            self.episode_length += 1
            done_env_ids = done.nonzero(as_tuple = False).squeeze(-1)

            # --- 计算 Actor 损失 (来自 grad_nav.py) ---
            next_values[i + 1] = self.target_critic(obs_next["privileged_state"]).squeeze(-1)
            
            # 处理 done_env_ids 的 next_values
            for id in done_env_ids:
                if info[id]["TimeLimit.truncated"]:
                    # 如果是超时
                    with th.no_grad():
                         term_obs_priv = info[id]["terminal_observation"]["privileged_state"]
                         if self.obs_rms is not None:
                             term_obs_priv = privilege_obs_rms.normalize(term_obs_priv)
                         next_values[i + 1, id] = self.target_critic(term_obs_priv).squeeze(-1)
                else:
                    # 如果是碰撞或成功 (真实终止)
                    next_values[i + 1, id] = 0.

            rew_acc[i + 1, :] = rew_acc[i, :] + gamma * rew

            # 累积 Actor 损失
            if i < self.steps_num - 1:
                if len(done_env_ids) > 0: # 确保 done_env_ids 不是空的
                    actor_loss = actor_loss + (- rew_acc[i + 1, done_env_ids] - self.gamma * gamma[done_env_ids] * next_values[i + 1, done_env_ids]).sum()
            else:
                actor_loss = actor_loss + (- rew_acc[i + 1, :] - self.gamma * gamma * next_values[i + 1, :]).sum()
            
            gamma = gamma * self.gamma
            gamma[done_env_ids] = 1.
            rew_acc[i + 1, done_env_ids] = 0.
            
            # 为 Critic 训练存储数据
            with th.no_grad():
                self.rew_buf[i] = rew.clone()
                self.done_mask[i] = done.clone().to(th.float32)
                self.next_values[i] = next_values[i + 1].clone()
                
            # (省略 grad_nav 中的 episode loss 记录逻辑)
            
            obs = obs_next # 准备下一次循环

        actor_loss /= (self.steps_num * self.num_envs)
        
        if self.ret_rms is not None:
            actor_loss = actor_loss * th.sqrt(ret_var)
            
        self.actor_loss = actor_loss.detach().cpu().item()
        self.step_count += self.steps_num * self.num_envs
        
        return actor_loss, latent_state
        
    def initialize_env(self):
        # (来自 grad_nav.py)
        # self.env.clear_grad() # 您的 BaseEnv 没有 clear_grad，但 PointMassDynamics 有 detach
        self.env.detach()
        obs = self.env.reset()
        
    @th.no_grad()
    def compute_target_values(self):
        # (来自 grad_nav.py)
        if self.critic_method == 'one-step':
            self.target_values = self.rew_buf + self.gamma * self.next_values
        elif self.critic_method == 'td-lambda':
            Ai = th.zeros(self.num_envs, dtype = th.float32, device = self.device)
            Bi = th.zeros(self.num_envs, dtype = th.float32, device = self.device)
            lam = th.ones(self.num_envs, dtype = th.float32, device = self.device)
            for i in reversed(range(self.steps_num)):
                lam = lam * self.lam * (1. - self.done_mask[i]) + self.done_mask[i]
                term1 = self.lam * self.gamma * Ai
                term2 = self.gamma * self.next_values[i]
                term3 = (1. - lam) / (1. - self.lam + 1e-8) * self.rew_buf[i] # 避免除以0
                
                Ai = (1.0 - self.done_mask[i]) * (term1 + term2 + term3)
                
                Bi = self.gamma * (self.next_values[i] * self.done_mask[i] + Bi * (1.0 - self.done_mask[i])) + self.rew_buf[i]
                self.target_values[i] = (1.0 - self.lam) * Ai + lam * Bi
        else:
            raise NotImplementedError
            
    def compute_critic_loss(self, batch_sample):
        # (来自 grad_nav.py)
        # 注意：batch_sample['obs'] 对应的是 self.privilege_obs_buf
        predicted_values = self.critic(batch_sample['obs']).squeeze(-1) 
        target_values = batch_sample['target_values']
        critic_loss = ((predicted_values - target_values) ** 2).mean()
        return critic_loss

    def df_to_tensorboard(self, logger_instance, df, prefix="eval"):
        # (来自 bptt_algorithm.py)
        keys = df.columns.tolist()
        for key in keys:
            if key not in self.whitelisted_tensorboard_keys:
                continue
            logger_instance.record(f"{prefix}/{key}", df[key].iloc[-1])