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
        # allow policies to configure their own actions, inertial frames, and obs
        self.action_type = get_enum(ActionType, action_type)
        self.inertial_frame = get_enum(Frame, inertial_frame)
        self.target_type = get_enum(TargetType, target_type)

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

        # target generator
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

        # properties that we expose as read-only
        self._target = th.zeros((self.num_envs, 3), device=self.device)
        self._target_speed = th.zeros((self.num_envs, 1), device=self.device)

        # <<< START MODIFICATION 4.1: D-RAPF 初始化 >>>
        
        # 1. 獲取 D-RAPF 獎勵權重
        self.k_att_base = self.reward_kwargs.get("k_att_base", 1.0)
        self.k_rep_base = self.reward_kwargs.get("k_rep_base", 10.0)
        self.k_rot_base = self.reward_kwargs.get("k_rot_base", 2.0)
        # 總威脅度 $\rho$ 的歸一化分母
        self.rapf_max_threat_denominator = self.reward_kwargs.get("rapf_max_threat", 100.0) 
        
        # 2. 獲取深度傳感器規格 (用於構建坐標網格)
        try:
            depth_sensor_cfg = [s for s in sensor_kwargs if s["uuid"] == "depth"][0]
            H, W = depth_sensor_cfg["resolution"]
            # 獲取 HFOV (默認 90)，轉換為弧度
            hfov = np.deg2rad(depth_sensor_cfg.get("hfov", 90.0))
            # 獲取 VFOV (用於 Z 軸) - 估算
            vfov = hfov * (H / W) 
        except Exception:
            # (如果沒有深度傳感器，則使用默認值，D-RAPF 將不會被正確觸發)
            H, W = 64, 64
            hfov = np.deg2rad(90.0)
            vfov = np.deg2rad(60.0)

        # 3. 創建靜態坐標網格 (用於輕量化計算)
        # 機體坐標系: X-前, Y-左, Z-上
        # 深度圖 W (寬度) 對應 機體 Y (左右)
        # 深度圖 H (高度) 對應 機體 Z (上下)
        
        y_tan_max = np.tan(hfov / 2.0)
        z_tan_max = np.tan(vfov / 2.0)

        # 創建 (H, W) 網格, 每個像素代表其在機體 Z 軸上的方向分量
        z_coords = th.linspace(z_tan_max, -z_tan_max, H, device=self.device)
        self.rapf_z_coords = z_coords.view(1, H, 1).expand(self.num_envs, H, W)

        # 創建 (H, W) 網格, 每個像素代表其在機體 Y 軸上的方向分量
        y_coords = th.linspace(y_tan_max, -y_tan_max, W, device=self.device) # Y-左, 所以 W=0 時 Y 為正
        self.rapf_y_coords = y_coords.view(1, 1, W).expand(self.num_envs, H, W)
        
        # 我們假設所有排斥力 X 分量 (前/後) 都是 -1 (向後推)
        self.rapf_x_coords = -th.ones((self.num_envs, H, W), device=self.device)
        # <<< END MODIFICATION 4.1 >>>


        # add target to super's observation_space
        if self.target_type == TargetType.TARGET_VELOCITY:
            self.observation_space.spaces["target"] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
            )
        elif self.target_type == TargetType.TARGET_VELOCITY_TARGET_DISTANCE:
            self.observation_space.spaces["target"] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
            )

        # deprecated - action_space might not be correct as it depends on
        # the policy and we don't use this
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )

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
                    self.omega, # (創新點 2)
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
                    self.omega, # (創新點 2)
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
                    self.omega, # (創新點 2)
                ]
            ).to(self.device)

        else:
            raise NotImplementedError

    def reset_agents(self, indices: Optional[List] = None):
        # ... (此函數保持不變) ...
        timerlog.timer.tic("sample_targets")
        safe_spawn_radius = self.random_kwargs.get("safe_spawn_radius", 1.0)
        min_starting_distance_to_target = self.random_kwargs.get(
            "min_starting_distance_to_target", 5.0
        )
        indices = (
            th.arange(self.num_envs, device=self.device) if indices is None else indices
        )
        position = self.safe_generate(self.position_rng, indices, safe_spawn_radius).to(
            self.device
        )
        self._target[indices] = self.safe_generate(
            self.target_rng, indices, safe_spawn_radius
        ).to(self.device)
        self._target_speed[indices] = self.target_speed_rng.generate(
            (len(indices), 1)
        ).to(self.device)
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

    def step(self, action: th.Tensor, is_test=False):
        # ... (此函數保持不變) ...
        device = self.device
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
            return super().step(thrust_wf, x_vector_wf, is_test=is_test)
        elif self.action_type == ActionType.THRUST_TARGET_YAW:
            return super().step(thrust_wf, self.target_direction, is_test=is_test)
        elif self.action_type == ActionType.THRUST_YAW:
            yaw = action[:, 3].to(self.device)
        elif self.action_type == ActionType.THRUST_YAW_RATE:
            body_x = self.rot_wb.R[:, :, 0]
            body_x_proj = th.stack(
                [body_x[:, 0], body_x[:, 1], th.zeros_like(body_x[:, 2])], dim=1
            )
            body_x_proj = F.normalize(body_x_proj)
            cur_yaw = th.atan2(body_x_proj[:, 1], body_x_proj[:, 0])
            yaw_rate = action[:, 3].to(self.device)
            yaw = cur_yaw + yaw_rate * (1.0 / self.dynamics.ctrl_dt)
        else:
            raise NotImplementedError
        ones = th.ones(self.num_envs, device=device)
        zeros = th.zeros(self.num_envs, device=device)
        Rz = th.stack(
            [
                th.stack([th.cos(yaw), -th.sin(yaw), zeros], dim=1),
                th.stack([th.sin(yaw), th.cos(yaw), zeros], dim=1),
                th.stack([zeros, zeros, ones], dim=1),
            ],
            dim=1,
        )
        x_vector = (
            th.tensor([1.0, 0.0, 0.0], device=device)
            .view(1, 3, 1)
            .expand(self.num_envs, 3, 1)
        )
        if self.inertial_frame == Frame.START:
            x_vector_wf = (self.rot_ws.R @ Rz @ x_vector).squeeze(
                -1
            )
        elif self.inertial_frame == Frame.WORLD:
            x_vector_wf = (Rz @ x_vector).squeeze(-1)
        elif self.inertial_frame == Frame.BODY:
            x_vector_wf = (self.rot_wb.R @ Rz @ x_vector).squeeze(
                -1
            )
        else:
            raise NotImplementedError
        return super().step(thrust_wf, x_vector_wf, is_test=is_test)

    def get_success(self):
        # ... (此函數保持不變) ...
        within_radius = (
            th.norm(self._target - self.position, dim=1) < self.success_radius
        )
        return within_radius

    def get_observation(self):
        # ... (此函數保持不變，因為 D-RAPF 僅用於獎勵) ...
        assert self.state.shape == (self.num_envs, self.state_size)
        assert self.visual

        if self.inertial_frame == Frame.WORLD:
            target_velocity = self.target_velocity
        elif self.inertial_frame == Frame.START:
            target_velocity = self.target_velocity_sb
        elif self.inertial_frame == Frame.BODY:
            target_velocity = self.target_velocity_bf
        else:
            raise NotImplementedError

        obs = {
            "state": self.state.to(self.device),
        }
        if self.visual:
            obs["depth"] = self.sensor_obs["depth"]
        if self.target_type == TargetType.TARGET_VELOCITY:
            obs["target"] = target_velocity.to(self.device)
        elif self.target_type == TargetType.TARGET_VELOCITY_TARGET_DISTANCE:
            obs["target"] = th.cat(
                [
                    target_velocity,
                    1.0 / (self.target_distance.clone().detach().clamp(min=0.5)),
                ],
                dim=1,
            ).to(self.device)

        return obs

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
        lambda_grad = self.reward_kwargs.get("lambda_grad", 0.0) # (我們不再使用 loss_grad)
        falloff_dis = self.reward_kwargs.get("falloff_dis", 1.0)
        safe_view_degrees = self.reward_kwargs.get("safe_view_degrees", 10.0)
        vel_thresh_slerp_yaw = self.reward_kwargs.get("vel_thresh_slerp_yaw", 1.0)

        # <<< START MODIFICATION 4.3: 使用 D-RAPF 替換 desired_direction_vector >>>
        
        # (獲取機體坐標系下的目標向量)
        target_vector_bf = self.target_vector_bf

        # 1. 計算 D-RAPF 指導向量 (在機體坐標系)
        #    (這一步包含了吸引力、排斥力、旋轉力和動態權重)
        depth_on_env_device = self.sensor_obs["depth"].to(self.device)
        drapf_guidance_bf = self._compute_drapf(depth_on_env_device, target_vector_bf)
        # 2. 將 D-RAPF 向量設置為 "期望的" 導航方向
        #    (desired_direction_vector 是獎勵函數中斷言的坐標系，所以我們要轉換回去)
        if self.inertial_frame == Frame.START:
             # (我們需要將 D-RAPF 從機體系(B)轉回世界系(W)，再轉回起始系(S))
             # R_wb @ V_b = V_w
             # R_ws.T @ V_w = V_s
             drapf_guidance_wf = th.matmul(self.rot_wb.R, drapf_guidance_bf.unsqueeze(-1)).squeeze(-1)
             desired_direction_vector = th.matmul(self.rot_ws.T, drapf_guidance_wf.unsqueeze(-1)).squeeze(-1)
        elif self.inertial_frame == Frame.WORLD:
             desired_direction_vector = th.matmul(self.rot_wb.R, drapf_guidance_bf.unsqueeze(-1)).squeeze(-1)
        elif self.inertial_frame == Frame.BODY:
             desired_direction_vector = drapf_guidance_bf
        else:
            raise NotImplementedError

        # (不再需要 loss_grad，因為 D-RAPF 已經包含了避障邏輯)
        loss_grad = th.zeros(self.num_envs, device=self.device) 
        
        # <<< END MODIFICATION 4.3 >>>


        # collision loss
        def positive_speed_towards_collision(
            velocity: th.Tensor, collision_vec: th.Tensor
        ):
            collision_dir = F.normalize(collision_vec, dim=1)
            speed = th.sum(velocity * collision_dir, dim=1)
            positive_speed = th.clamp(speed, min=0.0)
            return positive_speed

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

        # --- 3. (關鍵) 速度損失 (loss_v) 現在也基於 D-RAPF ---
        # (desired_velocity 現在基於 D-RAPF 指導)
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
            # ... (slerp 函數保持不變) ...
            a = F.normalize(a, dim=1)
            b = F.normalize(b, dim=1)
            t = t.clamp(0.0, 1.0)
            dot = (a * b).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
            theta = th.acos(dot)
            sin_theta = th.sin(theta)
            near_zero = sin_theta < 1e-6
            slerp_result = th.where(
                near_zero,
                F.normalize((1 - t) * a + t * b, dim=1),
                (th.sin((1 - t) * theta) * a + th.sin(t * theta) * b) / sin_theta,
            )
            return F.normalize(slerp_result, dim=1)

        # --- 4. 偏航損失 (loss_yaw) ---
        avg_vel = self.exp_moving_average_velocity.clone().detach()
        t = (self.position - self.start_position).norm(dim=1, keepdim=True)
        # (desired_direction_vector 現在是 D-RAPF 指導)
        desired_yaw_vector = (
            slerp(desired_direction_vector, avg_vel, t).clone().detach()
        )
        loss_yaw = -(desired_yaw_vector * self.yaw_vector).sum(dim=1)

        # (創新點 1: 成功避障獎勵)
        avoidance_reward_value = self.reward_kwargs.get("avoidance_reward_value", 5.0)
        avoidance_check_dis = self.reward_kwargs.get("avoidance_check_dis", 1.0)
        avoidance_dis_gain = self.reward_kwargs.get("avoidance_dis_gain", 0.5)
        was_near_collision = self._last_collision_dis < avoidance_check_dis
        distance_increased = (self.collision_dis - self._last_collision_dis) > avoidance_dis_gain
        avoidance_success = was_near_collision & distance_increased
        reward_avoid = th.where(avoidance_success, th.ones_like(self._reward) * avoidance_reward_value, th.zeros_like(self._reward))

        loss = (
            lambda_v * loss_v
            + lambda_vmax * loss_vmax
            + lambda_c * loss_c
            + lambda_a * loss_a
            + lambda_j * loss_j
            + lambda_om * loss_om
            + lambda_yaw * loss_yaw
            + lambda_grad * loss_grad # (loss_grad 現在恆為 0)
        )
        reward = -loss + reward_avoid
        metrics = {
            "loss_v": (lambda_v * loss_v).clone().detach().cpu(),
            "loss_vmax": (lambda_vmax * loss_vmax).clone().detach().cpu(),
            "loss_c": (lambda_c * loss_c).clone().detach().cpu(),
            "loss_a": (lambda_a * loss_a).clone().detach().cpu(),
            "loss_j": (lambda_j * loss_j).clone().detach().cpu(),
            "loss_om": (lambda_om * loss_om).clone().detach().cpu(),
            "loss_yaw": (lambda_yaw * loss_yaw).clone().detach().cpu(),
            "reward_avoid": reward_avoid.clone().detach().cpu(),
            "loss_grad": (lambda_grad * loss_grad).clone().detach().cpu()
        }

        return reward, metrics

    @property
    def target(self):
        return self._target

    @property
    def target_speed(self):
        # 修复：避免in-place操作破坏计算图
        return self._target_speed.clone()

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

    # <<< START MODIFICATION 4.2: D-RAPF 核心計算邏輯 >>>
    @th.no_grad() # 關鍵：此計算不應引入梯度，它僅用於獎勵
    def _compute_drapf(self, depth_obs: th.Tensor, target_vector_bf: th.Tensor):
        """
        根據當前深度圖和目標向量，計算 D-RAPF 指導向量 (機體坐標系)。
        [cite_start]靈感來源於: Adaptive multi-UAV cooperative path planning based on novel rotation [cite: 931-932]
        機體坐標系: X-前, Y-左, Z-上
        """
        
        # --- 0. 準備輸入 ---
        # 僅使用最新一幀深度圖
        if depth_obs.dim() == 4: # (B, Stack, H, W)
            # (假設幀堆疊 > 1，如果 = 1, 則 depth_obs 可能是 (B, 1, H, W))
            current_depth = depth_obs[:, 0, :, :] # (B, H, W)
        elif depth_obs.dim() == 3: # (B, H, W)
            current_depth = depth_obs
        else: # (B, 1, H, W)
             current_depth = depth_obs.squeeze(1)

        
        # 獲取傳感器規格
        try:
            sensor_cfg = [s for s in self._sensor_kwargs if s["uuid"] == "depth"][0]
            FAR_CLIP = sensor_cfg.get("far", 20.0)
            NEAR_CLIP = sensor_cfg.get("near", 0.25)
        except Exception:
            FAR_CLIP = 20.0
            NEAR_CLIP = 0.25

        # [cite_start]--- 1. 計算 F_att (吸引力) [cite: 1018-1020] ---
        # F_att 就是歸一化的目標向量 (機體坐標系)
        F_att = F.normalize(target_vector_bf, dim=1, eps=1e-6)

        # --- 2. 計算 F_rep (排斥力) [適配前向視角] ---
        # 將深度圖轉換為「威脅度」 (0.0 到 1.0)
        threat = th.clamp(1.0 - (current_depth - NEAR_CLIP) / (FAR_CLIP - NEAR_CLIP), 0.0, 1.0)
        
        # (措施 2) 計算總威脅度 $\rho$ (0.0 到 1.0 之間)
        rho = threat.sum(dim=(1, 2)) / self.rapf_max_threat_denominator
        rho = rho.clamp(min=0.0, max=1.0) # (B,)
        
        # 總威脅度（用於歸一化力向量）
        total_threat_sum = threat.sum(dim=(1, 2)).clamp(min=1e-6) # (B,)
        
        # 計算排斥力 X (前/後), Y (左/右), Z (上/下) 分量
        # rapf_x_coords 是 (B, H, W) 且全為 -1
        F_rep_x = (threat * self.rapf_x_coords).sum(dim=(1, 2)) / total_threat_sum
        # rapf_y_coords 是 (B, H, W)，值域 [-tan, +tan]
        F_rep_y = (threat * self.rapf_y_coords).sum(dim=(1, 2)) / total_threat_sum
        # rapf_z_coords 是 (B, H, W)，值域 [-tan, +tan]
        F_rep_z = (threat * self.rapf_z_coords).sum(dim=(1, 2)) / total_threat_sum

        # [cite_start]合成 F_rep (機體 X, Y, Z) [cite: 1021-1023]
        F_rep = th.stack([F_rep_x, F_rep_y, F_rep_z], dim=1)
        F_rep = F.normalize(F_rep, dim=1, eps=1e-6)

        # [cite_start]--- 3. 計算 F_rot (旋轉力) [cite: 1165-1167] ---
        # 我們只關心 XY (俯視) 平面上的旋轉
        F_att_xy = F_att[:, 0:2] # (B, 2)
        F_rep_xy = F_rep[:, 0:2] # (B, 2)

        # [cite_start]角度差 [cite: 1202-1203]
        angle_att = th.atan2(F_att_xy[:, 1], F_att_xy[:, 0] + 1e-6)
        angle_rep = th.atan2(F_rep_xy[:, 1], F_rep_xy[:, 0] + 1e-6)
        delta_angle = angle_att - angle_rep
        
        # 旋轉力 T 是 F_rep_xy 的 90 度法線 (T = [-F_rep_y, F_rep_x])
        T_xy = th.stack([-F_rep_xy[:, 1], F_rep_xy[:, 0]], dim=1)
        
        # [cite_start]旋轉力 (B, 2)，方向由角度差的符號決定 [cite: 1203-1205]
        F_rot_xy = T_xy * th.sign(delta_angle).unsqueeze(-1)
        
        F_rot = th.zeros_like(target_vector_bf)
        F_rot[:, 0:2] = F_rot_xy

        # --- 4. 動態權重 (措施 2) ---
        # rho (B,) -> (B, 1)
        rho_expanded = rho.unsqueeze(-1)
        
        k_att = self.k_att_base * (1.0 - rho_expanded) # 密集時，降低吸引力
        k_rep = self.k_rep_base * (1.0 + rho_expanded) # 密集時，提高排斥力
        k_rot = self.k_rot_base * (1.0 + rho_expanded) # 密集時，提高旋轉力
        
        # [cite_start]--- 5. 合成總力 [cite: 1215] ---
        F_res = (F_att * k_att) + (F_rep * k_rep) + (F_rot * k_rot)
        
        # 返回歸一化的指導向量
        return F.normalize(F_res, dim=1, eps=1e-6)
    # <<< END MODIFICATION 4.2 >>>