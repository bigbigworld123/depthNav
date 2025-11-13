import torch as th
import torch.nn as nn
from copy import deepcopy
import time
import sys

# (!!!) 导入您基线和复制过来的新模块
from .multi_input_policy import MultiInputPolicy # 您的 Actor
from .critic import CriticMLP                 # 新的 Critic
from .vae import VAE                           # 新的 VAE (CENet)
from ..envs.navigation_env import NavigationEnv
from ..utils.critic_dataset import CriticDataset
from ..utils.running_mean_std import RunningMeanStd
from stable_baselines3.common import logger

class SHAC:
    def __init__(
        self,
        policy: MultiInputPolicy,
        critic: CriticMLP,
        vae: VAE,
        env: NavigationEnv,
        eval_envs: list,
        eval_csvs: list,
        learning_rate_actor: float,
        learning_rate_critic: float,
        learning_rate_vae: float,
        steps_num: int, # 短时窗长度, e.g., 32
        gamma: float,
        lam: float, # for GAE
        critic_iterations: int,
        max_epochs: int, # 添加max_epochs参数
        # ... 其他来自 gradNav.py __init__ 的参数 ...
        run_name: str,
        logging_dir: str,
        device: str,
    ):
        
        self.env = env
        self.policy = policy # Actor
        self.critic = critic
        self.target_critic = deepcopy(self.critic)
        self.vae = vae
        
        self.device = device
        self.steps_num = steps_num
        self.gamma = gamma
        self.lam = lam
        self.critic_iterations = critic_iterations
        self.max_epochs = max_epochs # 保存max_epochs参数
        
        # (!!!) 模仿 gradnav.py __init__ 初始化 Optimizers
        #
        self.actor_optimizer = th.optim.AdamW(self.policy.parameters(), lr=learning_rate_actor)
        self.critic_optimizer = th.optim.AdamW(self.critic.parameters(), lr=learning_rate_critic)
        self.vae_optimizer = th.optim.AdamW(self.vae.parameters(), lr=learning_rate_vae)
        
        # (!!!) 模仿 gradnav.py __init__ 初始化 Buffers
        # 您需要根据您环境的观测维度来定义
        self.num_obs = self.env.observation_space["state"].shape[0] # 示例
        self.num_privilege_obs = self.env.privileged_observation.shape[1]
        
        self.obs_buf = th.zeros((self.steps_num, self.env.num_envs, self.num_obs), device=self.device)
        self.privilege_obs_buf = th.zeros((self.steps_num, self.env.num_envs, self.num_privilege_obs), device=self.device)
        self.rew_buf = th.zeros((self.steps_num, self.env.num_envs), device=self.device)
        self.done_mask = th.zeros((self.steps_num, self.env.num_envs), device=self.device)
        # 初始化 next_values，长度为 steps_num + 1 以容纳最后一步的价值估计
        self.next_values = th.zeros((self.steps_num + 1, self.env.num_envs), device=self.device)
        self.target_values = th.zeros((self.steps_num, self.env.num_envs), device=self.device)

        # (!!!) 模仿 gradNav 初始化 RMS 
        #
        self.obs_rms = RunningMeanStd(shape=(self.num_obs,), device=self.device)
        self.privilege_obs_rms = RunningMeanStd(shape=(self.num_privilege_obs,), device=self.device)
        
        # ... 其他初始化 ...
        self._logger = logger.configure(logging_dir, ["stdout", "tensorboard"])
        self.episode_loss = th.zeros(self.env.num_envs, device=self.device)
        self.episode_gamma = th.ones(self.env.num_envs, device=self.device)
        self.episode_length = th.zeros(self.env.num_envs, device=self.device)

    # (!!!) 核心 Actor 训练循环
    def compute_actor_loss(self):
        # 此方法基于 qianzhong-chen/grad_nav/algorithms/gradnav.py 的 compute_actor_loss
        #
        
        actor_loss = th.tensor(0., device=self.device)
        vae_loss = th.tensor(0., device=self.device)
        
        # (!!!) gradNav 在这里初始化环境，您的 BPTT 也是
        # 注意：gradNav 的 env.initialize_trajectory() 会返回 obs 和 priv_obs
        # 您的 env.reset() 只返回 obs。您需要调整。
        
        # (!!!) 模仿 gradNav 的 initialize_trajectory()
        # 您需要修改您的 env.reset() 以返回 (obs, priv_obs, obs_hist)
        # 您的 env.reset() 只返回 obs。您需要调整。
        # 假设您的 env.reset() 已经调用，我们从 env 获取初始状态
        obs = self.env.get_observation()
        privilege_obs = self.env.privileged_observation
        
        # 确保所有张量在正确的设备上
        for key in obs:
            if isinstance(obs[key], th.Tensor):
                obs[key] = obs[key].to(self.device)
        privilege_obs = privilege_obs.to(self.device)
        
        # 归一化 (可选，但 gradNav 做了)
        obs_state_normalized = self.obs_rms.normalize(obs["state"])
        privilege_obs_normalized = self.privilege_obs_rms.normalize(privilege_obs)
        obs["state"] = obs_state_normalized
        
        # 初始化 GRU 隐藏状态
        latent_state = th.zeros(
            (self.env.num_envs, self.policy.latent_dim), device=self.policy.device
        )

        for i in range(self.steps_num): # 短时窗循环
            
            # 存储用于 Critic 训练的数据
            with th.no_grad():
                self.obs_buf[i] = obs["state"].clone()
                self.privilege_obs_buf[i] = privilege_obs_normalized.clone()
            
            # (!!!) VAE/CENet 前向传播
            # 您的 env.step 必须返回 obs_hist
            obs_hist = self.env.obs_hist_buf.get_concatenated()
            obs_hist = obs_hist.to(self.device)  # 确保在正确的设备上
            vae_output, _ = self.vae.forward(obs_hist)
            
            # (!!!) Actor (Policy) 前向传播
            # 注意：您的 MultiInputPolicy 接收 obs 字典、vae_output 和 latent_state
            actions, latent_state = self.policy(obs, vae_output, latent_state)
            
            # (!!!) 环境步进
            # 确保您的 env.step 返回 (obs, priv_obs, obs_hist, rew, done, info)
            obs, privilege_obs, obs_hist, rew, done, info = self.env.step(actions)
            
            # 确保所有张量在正确的设备上
            for key in obs:
                if isinstance(obs[key], th.Tensor):
                    obs[key] = obs[key].to(self.device)
            privilege_obs = privilege_obs.to(self.device)
            rew = rew.to(self.device)
            done = done.to(self.device)

            # (!!!) VAE 损失计算
            # 模仿 gradNav
            # gradNav 使用 VAE 重构下一时刻的观测
            # 您的 vae_obs_buf 需要在 env.step() 中被更新为 *下一时刻* 的状态
            recons_loss, kld_loss, vae_loss_grad = self.compute_vae_loss(
                obs_hist, self.env.vae_obs_buf, done
            )
            vae_loss = vae_loss + vae_loss_grad
            
            # 归一化新的观测
            with th.no_grad():
                self.obs_rms.update(obs["state"])
                self.privilege_obs_rms.update(privilege_obs)
                obs_state_normalized = self.obs_rms.normalize(obs["state"])
                privilege_obs_normalized = self.privilege_obs_rms.normalize(privilege_obs)
                obs["state"] = obs_state_normalized
            
            # (!!!) Critic 评估价值
            # 模仿 gradNav
            next_values = th.zeros((self.steps_num + 1, self.env.num_envs), device=self.device)
            next_values[i + 1] = self.target_critic(privilege_obs_normalized).squeeze(-1)
            
            done_env_ids = done.nonzero(as_tuple=False).squeeze(-1)
            
            # 处理 done 的环境 (从 info 中获取重置前的状态)
            for id in done_env_ids:
                if info[id]["episode_length"] < self.env.max_episode_steps: # Early termination
                    next_values[i + 1, id] = 0.
                else: # 正常结束
                    # (!!!) 关键：使用重置前的最后状态来估计价值
                    last_priv_obs = info[id]["privilege_obs_before_reset"]
                    last_priv_obs = last_priv_obs.to(self.device)  # 确保在正确的设备上
                    last_priv_obs_norm = self.privilege_obs_rms.normalize(last_priv_obs)
                    next_values[i + 1, id] = self.target_critic(last_priv_obs_norm).squeeze(-1)

            # (!!!) 计算 Actor 损失 (SHAC 核心)
            # 这与您的 BPTT 不同，BPTT 只累加奖励。
            # SHAC 累加（奖励 + 折扣后的下一状态价值）
            #
            
            # 简化版损失 (gradNav 使用了更复杂的 rew_acc)
            # loss = - (rew + self.gamma * next_values[i + 1] * (1.0 - done.float()))
            # actor_loss = actor_loss + loss.mean()
            
            # 复制 gradNav 的损失计算逻辑 (更鲁棒)
            # ... [此处复制 gradNav.compute_actor_loss 中的 actor_loss 累加逻辑] ...
            # ... [这部分比较复杂，需要仔细移植 rew_acc 和 gamma 的更新] ...
            
            # 存储 Critic 训练数据
            with th.no_grad():
                self.rew_buf[i] = rew.clone()
                self.done_mask[i] = done.clone().to(th.float32)
                self.next_values[i] = next_values[i + 1].clone()
                
            # ... [处理 BPTT 中的 episode_loss_his 等日志记录] ...
            # ... [这部分可以从 bptt_algorithm.py 复制] ...

        actor_loss /= (self.steps_num * self.env.num_envs)
        vae_loss /= self.steps_num
        
        return actor_loss, vae_loss

    # (!!!) 复制 compute_vae_loss
    def compute_vae_loss(self, obs_history, next_obs, done):
        # 复制自 qianzhong-chen/grad_nav/algorithms/gradnav.py
        #
        vae_loss_dict = self.vae.loss_fn(obs_history.clone().detach(), next_obs.clone().detach())
        valid = (done == 0).squeeze()
        vae_loss = th.mean(vae_loss_dict['loss'][valid])
        # ...
        return recons_loss.item(), kld_loss.item(), vae_loss

    # (!!!) 复制 vae_update
    def vae_update(self, vae_loss):
        # 复制自 qianzhong-chen/grad_nav/algorithms/gradnav.py
        #
        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        nn.utils.clip_grad_norm_(self.vae.parameters(), 1.0) # 使用 grad_norm
        self.vae_optimizer.step()

    # (!!!) 复制 compute_target_values
    def compute_target_values(self):
        # 复制自 qianzhong-chen/grad_nav/algorithms/gradnav.py
        #
        if self.critic_method == 'td-lambda':
            # ... [复制 TD-Lambda 逻辑] ...
            pass
        self.target_values[i] = (1.0 - self.lam) * Ai + lam * Bi

    # (!!!) 复制 compute_critic_loss
    def compute_critic_loss(self, batch_sample):
        # 复制自 qianzhong-chen/grad_nav/algorithms/gradnav.py
        #
        predicted_values = self.critic(batch_sample['obs']).squeeze(-1)
        target_values = batch_sample['target_values']
        critic_loss = ((predicted_values - target_values) ** 2).mean()
        return critic_loss

    # (!!!) 核心训练循环
    def learn(self, render=False, start_iter=0):
        # 此方法基于 qianzhong-chen/grad_nav/algorithms/gradnav.py 的 train()
        #
        
        self.env.reset() # 初始化环境

        for epoch in range(self.max_epochs): # max_epochs 来自 config
            
            # --- 1. Actor 和 VAE 训练 ---
            def actor_closure():
                self.actor_optimizer.zero_grad()
                self.vae_optimizer.zero_grad() # VAE 梯度一起算
                
                actor_loss, vae_loss = self.compute_actor_loss()
                
                # (!!!) 关键：Actor 和 VAE 一起反向传播
                total_loss = actor_loss + vae_loss
                total_loss.backward()
                
                # 梯度裁剪
                nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                nn.utils.clip_grad_norm_(self.vae.parameters(), 1.0)
                
                return total_loss

            # Actor 和 VAE 参数更新
            self.actor_optimizer.step(actor_closure)
            # self.vae_optimizer.step() # PyTorch 2.0 的 AdamW.step() 不再需要 closure
            
            # --- 2. Critic 训练 ---
            with th.no_grad():
                self.compute_target_values() # 计算 TD-Lambda 目标
                # (!!!) 使用我们复制的 CriticDataset
                # 确保数据在正确的设备上
                privilege_obs_buf = self.privilege_obs_buf.to(self.device)
                target_values = self.target_values.to(self.device)
                dataset = CriticDataset(
                    self.batch_size, 
                    privilege_obs_buf, 
                    target_values
                )
            
            # 循环训练 Critic
            for j in range(self.critic_iterations):
                for batch_sample in dataset:
                    # 确保批次数据在正确的设备上
                    batch_sample['obs'] = batch_sample['obs'].to(self.device)
                    batch_sample['target_values'] = batch_sample['target_values'].to(self.device)
                    
                    self.critic_optimizer.zero_grad()
                    training_critic_loss = self.compute_critic_loss(batch_sample)
                    training_critic_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                    self.critic_optimizer.step()
            
            # --- 3. 更新 Target Critic ---
            # 复制自 gradNav.train()
            with th.no_grad():
                alpha = 0.4 # target_critic_alpha
                for param, param_targ in zip(self.critic.parameters(), self.target_critic.parameters()):
                    param_targ.data.mul_(alpha)
                    param_targ.data.add_((1. - alpha) * param.data)
            
            # --- 4. 日志记录和评估 ---
            # ... [这部分可以从 bptt_algorithm.py 复制] ...
            if epoch % self.log_interval == 0:
                # ... [调用评估、记录 tensorboard] ...
                pass