from habitat_sim import SensorType
import numpy as np
from typing import Union, Tuple, List, Optional, Dict
from enum import Enum
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from pylogtools import timerlog, colorlog
from ..utils.type import TensorDict
from .base_env import BaseEnv
from .scene_manager import Bounds
from ..utils.type import Uniform, Normal
from ..utils import Rotation3
from ..utils.hist_obs_buffer import ObsHistBuffer


class ActionType(Enum):
    THRUST_FIXED_YAW = 0
    THRUST_TARGET_YAW = 1
    THRUST_YAW = 2
    THRUST_YAW_RATE = 3


class Frame(Enum):
    WORLD = 0
    START = 1
    BODY = 2


class TargetType(Enum):
    TARGET_VELOCITY = 0
    TARGET_VELOCITY_TARGET_DISTANCE = 1


def get_enum(enum, value_str):
    if type(value_str) == str:
        try:
            return enum[value_str]
        except KeyError:
            raise ValueError(f"Invalid enum value: {value_str}")
    else:
        raise NotImplementedError


class NavigationEnv(BaseEnv):
    def __init__(
        self,
        num_envs: int = 1,
        seed: int = 42,
        visual: bool = False,
        single_env: bool = False,
        max_episode_steps: int = 256,
        device: Optional[th.device] = th.device("cpu"),
        requires_grad: bool = False,
        robot_radius: float = 0.1,
        dynamics_kwargs=None,
        random_kwargs=None,
        base_action=[0.0, 0.0, 0.0],
        action_type="THRUST_YAW",
        inertial_frame="START",
        target_type="TARGET_VELOCITY_TARGET_DISTANCE",
        target_kwargs=None,
        reward_kwargs=None,
        bounds: Optional[Union[Bounds, Dict]] = None,
        scene_kwargs: Optional[Dict] = None,
        sensor_kwargs: Optional[List] = None,
    ):
        # 允许策略配置自己的动作、惯性系和观测
        self.action_type = get_enum(ActionType, action_type)
        self.inertial_frame = get_enum(Frame, inertial_frame)
        self.target_type = get_enum(TargetType, target_type)

        # VAE/CENet 参数 (可以从 config 中读取)
        # (!!!) 我们必须在 super().__init__ 之前定义这些
        self.num_history = 5 # 示例：同 gradNav
        self.num_latent = 24 # 示例：同 gradNav
        # self.visual_feature_size = 192 # 示例：这个值应来自您的 CNN 配置

        super().__init__(
            num_envs=num_envs,
            seed=seed,
            visual=visual,
            single_env=single_env,
            max_episode_steps=max_episode_steps,
            device=device,
            requires_grad=requires_grad,
            robot_radius=robot_radius,
            dynamics_kwargs=dynamics_kwargs,
            random_kwargs=random_kwargs,
            base_action=base_action,
            bounds=bounds,
            scene_kwargs=scene_kwargs,
            sensor_kwargs=sensor_kwargs,
        )

        # 目标生成器 (Target generator)
        self.target_kwargs = target_kwargs or {}
        self.target_rng = self._create_rng(
            "target",
            self.random_kwargs,
            default_rng=Uniform([9.0, 0.0, 5.0], [0.0, 2.5, 2.5]),
        )
        self.target_speed_rng = self._create_rng(
            "target_speed",
            self.random_kwargs,
            default_rng=Uniform([self.target_kwargs.get("target_speed", 5.0)], [0.0]),
        )
        self.success_radius = self.target_kwargs.get("success_radius", 0.5)
        self.reward_kwargs = reward_kwargs or {}

        # 我们公开为只读的属性
        self._target = th.zeros((self.num_envs, 3), device=self.device)
        self._target_speed = th.zeros((self.num_envs, 1), device=self.device)

        # 将 target 添加到 super 的 observation_space
        if self.target_type == TargetType.TARGET_VELOCITY:
            self.observation_space.spaces["target"] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
            )
        elif self.target_type == TargetType.TARGET_VELOCITY_TARGET_DISTANCE:
            self.observation_space.spaces["target"] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
            )

        # 已弃用 - action_space 可能不正确，因为它取决于策略
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )

        # (!!!) VAE/CENet 缓冲区的初始化 (!!!)
        # (!!!) 必须在 __init__ 的末尾，确保所有依赖项 (如 self._target) 都已初始化
        
        # 1. 定义维度
        self.state_dim = self.state_size # self.state_size 在 super().__init__ 中计算
        self.privilege_obs_dim = 17        
        # 假设 VAE 观测就是 state
        self.vae_obs_dim = self.state_dim 

        # 2. 初始化缓冲区
        self.obs_hist_buf = ObsHistBuffer(batch_size=self.num_envs,
                                        vector_dim=self.vae_obs_dim,
                                        buffer_size=self.num_history,
                                        device=self.device)
        self.vae_obs_buf = th.zeros((self.num_envs, self.vae_obs_dim), device=self.device)

    @property
    def privileged_observation(self):
        # 这是 Critic 才能看到的 "作弊" 信息
        # 模仿 gradNav (lines 443-459)

        # 示例：
        # 处理 collision_dis 可能为 None 的情况
        collision_dis = self.collision_dis if self.collision_dis is not None else th.ones(self.num_envs, device=self.device) * 100.0
        
        priv_obs = th.hstack([
            self.position,              # 真实位置 (3)
            self.velocity,             # 真实速度 (3)
            self.quaternion,           # 真实姿态 (4)
            self.omega,                # 真实角速度 (3)
            self.target_direction,     # 目标方向 (3)
            collision_dis.unsqueeze(1), # 到障碍物的真实距离 (1)
            # ... 您可以添加任何您认为 Critic 应该知道的信息
        ]).to(self.device)

        return priv_obs
    @property
    def state(self):
        # generate velocity and quaternion noise
        size = (self.num_envs, 3)
        velocity_noise = self.velocity_noise_rng.generate(size).to(self.device)
        euler_zyx_noise = self.rotation_noise_rng.generate(size).to(self.device)
        delta_rot = Rotation3.from_euler_zyx(euler_zyx_noise, device=self.device)

        if self.inertial_frame == Frame.WORLD:
            q_noised = Rotation3(self.rot_wb.R @ delta_rot.R).to_quat()
            q_noised = th.where(q_noised[:, 0:1] < 0, -q_noised, q_noised)
            v_noised = self.velocity + velocity_noise

            return th.hstack(
                [
                    q_noised,
                    v_noised,
                    # self.omega,
                ]
            ).to(self.device)

        elif self.inertial_frame == Frame.START:
            R_sw = self.rot_ws.T
            R_wb = self.rot_wb.R
            R_sb = R_sw @ R_wb
            q_noised = Rotation3(R_sb @ delta_rot.R).to_quat()
            q_noised = th.where(q_noised[:, 0:1] < 0, -q_noised, q_noised)

            v_noised = self.velocity_sb + velocity_noise

            return th.hstack(
                [
                    q_noised,
                    v_noised,
                    # self.omega,
                ]
            ).to(self.device)

        elif self.inertial_frame == Frame.BODY:
            R_sw = self.rot_ws.T
            R_wb = self.rot_wb.R
            R_sb = R_sw @ R_wb
            q_noised = Rotation3(R_sb @ delta_rot.R).to_quat()
            q_noised = th.where(q_noised[:, 0:1] < 0, -q_noised, q_noised)

            v_noised = self.velocity_bf + velocity_noise
            return th.hstack(
                [
                    q_noised,
                    v_noised,
                    # self.omega,
                ]
            ).to(self.device)

        else:
            raise NotImplementedError

    def get_observation(self):
        observation = super().get_observation()
        # 确保所有观测值都在正确的设备上
        for key in observation:
            if isinstance(observation[key], th.Tensor):
                observation[key] = observation[key].to(self.device)
        return observation

    def reset_agents(self, indices: Optional[List] = None):
        timerlog.timer.tic("sample_targets")
        safe_spawn_radius = self.random_kwargs.get("safe_spawn_radius", 1.0)
        min_starting_distance_to_target = self.random_kwargs.get(
            "min_starting_distance_to_target", 5.0
        )
        indices = (
            th.arange(self.num_envs, device=self.device) if indices is None else indices
        )

        # generate position
        position = self.safe_generate(self.position_rng, indices, safe_spawn_radius).to(
            self.device
        )

        # generate target
        self._target[indices] = self.safe_generate(
            self.target_rng, indices, safe_spawn_radius
        ).to(self.device)
        self._target_speed[indices] = self.target_speed_rng.generate(
            (len(indices), 1)
        ).to(self.device)
        # make sure target is at least min_starting_distance_to_target from position
        too_close = th.zeros(self.num_envs, dtype=th.bool, device=self.device)
        max_iter = 1000
        for _ in range(max_iter):
            too_close[indices] = (position - self._target[indices]).norm(
                dim=1
            ) < min_starting_distance_to_target
            too_close_indices = too_close.nonzero(as_tuple=True)[0]
            if len(too_close_indices) == 0:
                break
            self._target[too_close_indices] = self.safe_generate(
                self.target_rng, too_close_indices, safe_spawn_radius
            ).to(self.device)
        timerlog.timer.toc("sample_targets")

        up = th.tensor([[0.0, 0.0, 1.0]]).expand(len(indices), 3)
        target_dir_wf = self._target[indices] - position
        start_rot = self.dynamics._calc_orientation(up, target_dir_wf, self.device)
        super().reset_agents(pos=position, start_rot=start_rot, indices=indices)
        # 在 reset_agents() 的结尾处
        self.vae_obs_buf[indices] = 0. # 重置 VAE 观测
        self.obs_hist_buf.buffer[indices] = 0. # 重置历史缓冲区

    def step(self, action: th.Tensor, is_test=False):
        device = self.device

        # --- 1. 这部分计算推力和目标方向的代码保持不变 ---
        if self.inertial_frame == Frame.START:
            thrust_sb = action[:, 0:3].to(self.device)
            thrust_wf = th.matmul(self.rot_ws.R, thrust_sb.unsqueeze(-1)).squeeze(-1)
        elif self.inertial_frame == Frame.WORLD:
            thrust_wf = action[:, 0:3].to(self.device)
        elif self.inertial_frame == Frame.BODY:
            thrust_b = action[:, 0:3].to(self.device)
            thrust_wf = th.matmul(self.rot_wb.R, thrust_b.unsqueeze(-1)).squeeze(-1)
        else:
            raise NotImplementedError

        if self.action_type == ActionType.THRUST_FIXED_YAW:
            x_vector_wf = self.rot_ws.R[:, 0]
            target_dir_wf = x_vector_wf
        elif self.action_type == ActionType.THRUST_TARGET_YAW:
            target_dir_wf = self.target_direction
        
        # ... [处理 THRUST_YAW 和 THRUST_YAW_RATE 的逻辑不变] ...
        elif self.action_type == ActionType.THRUST_YAW:
            yaw = action[:, 3].to(self.device)
            # ... [计算 Rz 和 x_vector 的逻辑] ...
            # ...
            if self.inertial_frame == Frame.START:
                x_vector_wf = (self.rot_ws.R @ Rz @ x_vector).squeeze(-1)
            # ... [其他 inertial_frame] ...
            target_dir_wf = x_vector_wf # 假设 yaw action 最终定义了 target_dir_wf
        
        elif self.action_type == ActionType.THRUST_YAW_RATE:
            # ... [计算 yaw 和 x_vector_wf 的逻辑] ...
            target_dir_wf = x_vector_wf
        else:
            raise NotImplementedError

        # --- 2. (!!!) 核心修改在这里 (!!!) ---
        
        # (A) 首先，调用 super().step() 来执行模拟，并获取基础返回值
        #     这将更新 self.position, self.velocity, self.state 等
        observations, reward, done, info = super().step(thrust_wf, target_dir_wf, is_test=is_test)

        # (B) 现在，self.state 和 self.privileged_observation 已经是最新状态
        
        # 1. 准备 VAE 观测数据
        #    (注意：我们使用 self.state，这是在 super().step() 中刚更新的)
        #    (您需要根据您的 VAE 设计来定义 vae_obs_buf 的内容)
        self.vae_obs_buf = self.state.clone().detach() 
        self.obs_hist_buf.update(self.vae_obs_buf)
        obs_hist = self.obs_hist_buf.get_concatenated()

        # 2. 准备特权观测
        #    (使用在 super().step() 后更新的最新状态)
        priv_obs = self.privileged_observation

        # 3. 准备 info，用于 buffer 收集
        obs_before_reset = observations # 这是 super().step() 返回的最新观测
        priv_obs_before_reset = priv_obs.clone() # 这是最新的特权观测

        # (C) 处理 is_test 和 done 的情况
        if not is_test:
            # (!!!) 将 VAE/Critic 需要的信息加入 info 
            for i, d in enumerate(done): # 使用 super().step() 返回的 done
                if d:
                    # 额外存储重置前的最后观测
                    info[i]["obs_before_reset"] = {k: v[i] for k, v in obs_before_reset.items()}
                    info[i]["privilege_obs_before_reset"] = priv_obs_before_reset[i]

            if done.any():
                # super().step() 内部已经调用了 self.reset_agents()
                # 我们需要获取 reset 后的新观测
                observations = self.get_observation()
                # (!!!) 注意：在 reset_agents 后，obs_hist 和 priv_obs 理论上也应该更新
                #     为确保一致性，我们在这里重新获取它们
                obs_hist = self.obs_hist_buf.get_concatenated()
                priv_obs = self.privileged_observation
                
            # 返回额外的信息给 SHAC 算法
            return observations, priv_obs, obs_hist, reward, done, info

        # (!!!) is_test=True 时的返回也需要修改
        return observations, priv_obs, obs_hist, reward, done, info

    def get_success(self):
        within_radius = (
            th.norm(self._target - self.position, dim=1) < self.success_radius
        )
        return within_radius

    def get_reward(self, action=None) -> th.Tensor:
        """
        BNL reward function
        """
        # BNL params
        beta_1 = self.reward_kwargs.get("beta_1", 2.5)
        beta_2 = self.reward_kwargs.get("beta_2", -32)
        lambda_v = self.reward_kwargs.get("lambda_v", 1)
        lambda_c = self.reward_kwargs.get("lambda_c", 2)
        lambda_a = self.reward_kwargs.get("lambda_a", 0.01)
        lambda_j = self.reward_kwargs.get("lambda_j", 0.001)
        lambda_om = self.reward_kwargs.get("lambda_om", 0.03)
        lambda_yaw = self.reward_kwargs.get("lambda_yaw", 0.5)
        lambda_vmax = self.reward_kwargs.get("lambda_vmax", 0.0)
        lambda_grad = self.reward_kwargs.get("lambda_grad", 0.0)
        falloff_dis = self.reward_kwargs.get("falloff_dis", 1.0)
        safe_view_degrees = self.reward_kwargs.get("safe_view_degrees", 10.0)
        vel_thresh_slerp_yaw = self.reward_kwargs.get("vel_thresh_slerp_yaw", 1.0)

        if self.scene_manager.load_geodesics:
            # only compute geodesic loss if geodesics are loaded
            with th.no_grad():
                geodesic_gradient = self.geodesic_gradient(self.position)
            # loss_grad = -th.sum(F.normalize(geodesic_gradient) * F.normalize(self.velocity), dim=1)
            desired_direction_vector = geodesic_gradient
        else:
            desired_direction_vector = self.target_direction
            # loss_grad = th.zeros(self.num_envs)

        # collision loss
        def positive_speed_towards_collision(
            velocity: th.Tensor, collision_vec: th.Tensor
        ):
            collision_dir = F.normalize(collision_vec, dim=1)
            speed = th.sum(velocity * collision_dir, dim=1)
            positive_speed = th.clamp(speed, min=0.0)
            return positive_speed

        # NOTE collision loss does not work well on policy that does not
        # know how to hover!!! Need to do curriculum training
        speed_towards_collision = positive_speed_towards_collision(
            self.velocity, self.collision_vector
        )
        distance_penalty = (
            falloff_dis - (self.collision_dis - self.robot_radius)
        ).relu() ** 2
        intersection_barrier = th.log(
            1 + th.exp(beta_2 * (self.collision_dis - self.robot_radius))
        )
        loss_c = (
            speed_towards_collision * distance_penalty + beta_1 * intersection_barrier
        )

        # penalize quadratically for exceeding speed limit
        loss_vmax = ((self.speed - self.target_speed).relu() ** 2).squeeze(1)

        # penalize deviation from target velocity
        desired_velocity = self.target_speed * desired_direction_vector
        velocity_difference = (desired_velocity - self.moving_average_velocity).norm(
            dim=1
        )
        loss_v = F.smooth_l1_loss(
            velocity_difference, th.zeros_like(velocity_difference), reduction="none"
        )

        # smoothness loss on acceleration, jerk, and body rate
        loss_a = self.acceleration.norm(dim=1) ** 2
        loss_j = self.jerk.norm(dim=1) ** 2
        loss_om = self.omega.norm(dim=1) ** 2

        def slerp(a: th.Tensor, b: th.Tensor, t: th.Tensor) -> th.Tensor:
            """Perform spherical linear interpolation (SLERP) between unit vectors (N, 3) a and b"""
            a = F.normalize(a, dim=1)
            b = F.normalize(b, dim=1)
            t = t.clamp(0.0, 1.0)

            dot = (a * b).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)  # cosine(theta)
            theta = th.acos(dot)  # (N, 1)

            sin_theta = th.sin(theta)
            near_zero = sin_theta < 1e-6

            # Linear interpolation fallback for very small angles
            slerp_result = th.where(
                near_zero,
                F.normalize((1 - t) * a + t * b, dim=1),
                (th.sin((1 - t) * theta) * a + th.sin(t * theta) * b) / sin_theta,
            )
            return F.normalize(slerp_result, dim=1)

        # at low speeds, yaw should track goal/geodesic direction
        # at high speeds, yaw should track velocity
        # avg_vel = self.moving_average_velocity.clone().detach()
        # desired_yaw_vector = avg_vel
        avg_vel = self.exp_moving_average_velocity.clone().detach()
        # t = (1. / vel_thresh_slerp_yaw) * avg_vel.norm(dim=1, keepdim=True) # t = 1 when vel.norm == vel_thresh_slerp_yaw
        t = (self.position - self.start_position).norm(dim=1, keepdim=True)
        desired_yaw_vector = (
            slerp(self.target_direction, avg_vel, t).clone().detach()
        )  # works, but should try without detach
        loss_yaw = -(desired_yaw_vector * self.yaw_vector).sum(dim=1)

        # WORKS
        # loss_yaw = F.smooth_l1_loss(self.yaw_vector, desired_yaw_vector, reduction="none")
        # loss_yaw = loss_yaw.sum(dim=1)

        def smooth_l1_cosine_loss(pred_vec, target_vec, angle_threshold_rad, delta=1.0):
            cos_sim = (pred_vec * target_vec).sum(dim=1).clamp(-1.0, 1.0)
            cos_thresh = np.cos(angle_threshold_rad)
            err = cos_thresh - cos_sim
            err = th.relu(err)
            loss = th.where(err < delta, 0.5 * (err**2) / delta, err - 0.5 * delta)
            return loss

        # safe_view_radians = np.deg2rad(safe_view_degrees)
        # loss_yaw = smooth_l1_cosine_loss(self.yaw_vector, desired_yaw_vector,
        # safe_view_radians, delta=0.05)

        # close to the start and goal, no yaw loss to prevent rapid yaw movements
        # loss_yaw = th.where(
        #     ((self.position - self.start_position).norm(dim=1) < 1.0),
        #     th.zeros_like(loss_yaw),
        #     loss_yaw
        # )
        loss_yaw = th.where(
            (self.target_distance < 1.0).squeeze(1), th.zeros_like(loss_yaw), loss_yaw
        )

        loss = (
            lambda_v * loss_v
            + lambda_vmax * loss_vmax
            + lambda_c * loss_c
            + lambda_a * loss_a
            + lambda_j * loss_j
            + lambda_om * loss_om
            + lambda_yaw * loss_yaw
            # + lambda_grad * loss_grad
        )
        reward = -loss
        metrics = {
            "loss_v": (lambda_v * loss_v).clone().detach().cpu(),
            "loss_vmax": (lambda_vmax * loss_vmax).clone().detach().cpu(),
            "loss_c": (lambda_c * loss_c).clone().detach().cpu(),
            "loss_a": (lambda_a * loss_a).clone().detach().cpu(),
            "loss_j": (lambda_j * loss_j).clone().detach().cpu(),
            "loss_om": (lambda_om * loss_om).clone().detach().cpu(),
            "loss_yaw": (lambda_yaw * loss_yaw).clone().detach().cpu(),
            # "loss_grad": (lambda_grad * loss_grad).clone().detach().cpu()
        }

        return reward, metrics

    @property
    def target(self):
        return self._target

    @property
    def target_speed(self):
        return self._target_speed

    @property
    def target_vector(self):
        return self._target.clone().detach() - self.position.clone().detach()

    @property
    def target_vector_bf(self):
        return self.dynamics._rotate_vector_world_to_body(self.target_vector)

    @property
    def target_direction(self):
        return F.normalize(self.target_vector, dim=1)

    @property
    def target_distance(self):
        return th.norm(self.target_vector, dim=1, keepdim=True)

    @property
    def target_velocity(self):
        # use PD control to get desired velocity command
        Kp = 1.5
        Kd = 0.0  # keep at zero, real world flight performance is poor with non-zero Kd
        des_velocity = Kp * self.target_vector - Kd * self.velocity
        # clamp so max norm is target_speed
        des_velocity_clamped = des_velocity.norm(dim=1, keepdim=True).clamp(
            max=self.target_speed.clone().detach()
        ) * F.normalize(des_velocity)
        return des_velocity_clamped

    @property
    def target_velocity_sb(self):
        return th.matmul(self.rot_ws.T, self.target_velocity.unsqueeze(-1)).squeeze(-1)

    @property
    def target_velocity_bf(self):
        return self.dynamics._rotate_vector_world_to_body(self.target_velocity)

    @property
    def yaw_vector(self):
        x_axis = self.rotation[:, :, 0]
        # zero out z component and normalize
        yaw_vec = th.stack(
            [x_axis[:, 0], x_axis[:, 1], th.zeros_like(x_axis[:, 2])], dim=1
        )
        yaw_vec = F.normalize(yaw_vec)
        return yaw_vec  # (N, 3)

    @property
    def x_axis(self):
        x_axis = self.rotation[:, :, 0]
        return x_axis
