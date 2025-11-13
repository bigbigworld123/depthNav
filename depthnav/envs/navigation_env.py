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
        
        # --- [修改点 1] APF/RAPF 参数 ---
        self.reward_kwargs = reward_kwargs or {}
        # 从reward_kwargs加载APF/RAPF参数，参考论文
        self.k_att = self.reward_kwargs.get("k_att", 1.5)
        self.k_rep = self.reward_kwargs.get("k_rep", 1.0)
        self.k_rot = self.reward_kwargs.get("k_rot", 0.5)
        # 对应论文中的 r_o (障碍物影响范围)
        self.d_rep_thresh = self.reward_kwargs.get("d_rep_thresh", 3.0) 
        # 对应论文中的 r_o (旋转力影响范围)
        self.d_rot_thresh = self.reward_kwargs.get("d_rot_thresh", 3.0) 
        # --- 结束 [修改点 1] ---

        # properties that we expose as read-only
        self._target = th.zeros((self.num_envs, 3), device=self.device)
        self._target_speed = th.zeros((self.num_envs, 1), device=self.device)

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

    def step(self, action: th.Tensor, is_test=False):
        device = self.device

        # compute thrust in world frame
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
            x_vector_wf = self.rot_ws.R[:, 0]  # use x axis of starting frame
            return super().step(thrust_wf, x_vector_wf, is_test=is_test)
        elif self.action_type == ActionType.THRUST_TARGET_YAW:
            return super().step(thrust_wf, self.target_direction, is_test=is_test)

        # else our action type requires us to calculate yaw
        elif self.action_type == ActionType.THRUST_YAW:
            yaw = action[:, 3].to(self.device)
        elif self.action_type == ActionType.THRUST_YAW_RATE:
            # obtain current yaw by computing angle between world x-axis and body x-axis
            body_x = self.rot_wb.R[:, :, 0]
            # project body x onto xy-plane
            body_x_proj = th.stack(
                [body_x[:, 0], body_x[:, 1], th.zeros_like(body_x[:, 2])], dim=1
            )
            body_x_proj = F.normalize(body_x_proj)
            # compute angle wrt world x-axis using atan2
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

        # get x-axis vector in world frame
        if self.inertial_frame == Frame.START:
            x_vector_wf = (self.rot_ws.R @ Rz @ x_vector).squeeze(
                -1
            )  # (N,3,1) -> (3,N)
        elif self.inertial_frame == Frame.WORLD:
            x_vector_wf = (Rz @ x_vector).squeeze(-1)  # (N,3,1) -> (3,N)
        elif self.inertial_frame == Frame.BODY:
            x_vector_wf = (self.rot_wb.R @ Rz @ x_vector).squeeze(
                -1
            )  # (N,3,1) -> (3,N)
        else:
            raise NotImplementedError

        return super().step(thrust_wf, x_vector_wf, is_test=is_test)

    def get_success(self):
        within_radius = (
            th.norm(self._target - self.position, dim=1) < self.success_radius
        )
        return within_radius

    def get_observation(self):
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

    # --- [修改点 2] 新增 APF/RAPF 辅助函数 ---
    
    def _calculate_attractive_force(self) -> th.Tensor:
        """
        计算指向目标的吸引力 (参考论文 Eq. 6)。
        使用一个简化的线性版本：F_att = k_att * direction_to_target
        """
        # self.target_direction 已经在 @property 中计算好了
        return self.k_att * self.target_direction

    def _calculate_repulsive_force(self) -> th.Tensor:
        """
        计算来自最近障碍物的排斥力 (参考论文 Eq. 7)。
        F_rep = k_rep * (1/d - 1/d_thresh) * (1/d^2) * grad(d)
        """
        d = self.collision_dis.unsqueeze(1)  # 距离 d, Shape: (N, 1)
        d_thresh = self.d_rep_thresh # 影响阈值 r_o

        # 创建一个mask，只在影响范围内才计算排斥力
        mask = (d < d_thresh).float()

        # grad(d) 是从障碍物指向UAV的方向
        # self.collision_vector 是 (collision_point - position), 方向指向障碍物
        grad_d = -F.normalize(self.collision_vector, dim=1) # Shape: (N, 3)

        # 计算力的幅度 (为避免d=0时除零，增加一个很小的epsilon)
        d_eps = d + 1e-6
        f_mag = self.k_rep * (1.0 / d_eps - 1.0 / d_thresh) * (1.0 / d_eps**2)
        
        # 仅在阈值内施加力
        F_rep = f_mag * grad_d * mask
        return F_rep

    def _calculate_rotational_force(self) -> th.Tensor:
        """
        计算旋转力以逃离局部最小值 (参考论文 Eq. 9 & 10)。
        """
        d = self.collision_dis.unsqueeze(1) # Shape (N, 1)
        d_thresh = self.d_rot_thresh # 影响阈值 r_o

        # 只在影响范围内才计算旋转力
        mask = (d < d_thresh).float()
        
        # --- 计算切线向量 T (参考 Eq. 10) ---
        
        # 1. 指向目标的向量
        vec_to_target = self.target_direction # (N, 3)
        
        # 2. 指向障碍物的向量 (归一化)
        vec_to_obstacle = F.normalize(self.collision_vector, dim=1) # (N, 3)

        # 3. 计算角度差 (Delta theta)，以决定顺时针或逆时针
        #    我们简化为在XY平面上计算
        angle_to_target = th.atan2(vec_to_target[:, 1], vec_to_target[:, 0])
        angle_to_obstacle = th.atan2(vec_to_obstacle[:, 1], vec_to_obstacle[:, 0])
        delta_theta = angle_to_target - angle_to_obstacle

        # 4. 定义旋转轴 n (论文中提到，我们简化为Z轴，适用于2.5D导航)
        n = th.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, -1)

        # 5. 计算切线方向 (即 T_clockwise)
        # T = (P_o - P_u)_normalized x n 
        T_clockwise = F.normalize(th.cross(vec_to_obstacle, n, dim=1), dim=1)
        T_counter_clockwise = -T_clockwise

        # 6. 根据角度差选择切线方向 (Eq. 10 逻辑)
        T = th.where(delta_theta.unsqueeze(1) > 0, T_clockwise, T_counter_clockwise)

        # --- 计算旋转力 (参考 Eq. 9) ---
        # 论文中的幅度为 k_rot * (1/d - 1/d_thresh) * (1/d)
        # 这里我们使用一个更平滑的线性衰减幅度: k_rot * (1 - d/d_thresh)
        d_eps = d + 1e-6
        f_mag = self.k_rot * (1.0 - d_eps / d_thresh).clamp(min=0.0)
        
        F_rot = f_mag * T * mask
        return F_rot

    # --- 结束 [修改点 2] ---


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

        # --- [!!] 关键修正: 在 no_grad() 中计算特权信息 [!!] ---
        # 这种方式模仿了原始代码处理 geodesic_gradient 的方式
        with th.no_grad():
            if self.scene_manager.load_geodesics:
                # 原始的 ToA 方法
                colorlog.log.warning("load_geodesics is True, using ToA gradient.")
                geodesic_gradient = self.geodesic_gradient(self.position)
                desired_direction_vector = geodesic_gradient
                
                # 初始化 metrics 的值
                F_att_norm_val = th.zeros(self.num_envs).cpu()
                F_rep_norm_val = th.zeros(self.num_envs).cpu()
                F_rot_norm_val = th.zeros(self.num_envs).cpu()
                F_res_norm_val = th.zeros(self.num_envs).cpu()
            else:
                # --- APF/RAPF 指导 ---
                # 计算力
                F_att = self._calculate_attractive_force()
                F_rep = self._calculate_repulsive_force()
                F_rot = self._calculate_rotational_force()

                # 合力
                F_res = F_att + F_rep + F_rot
                
                # [!!] 记录力的范数 (仍在 no_grad 块中) [!!]
                F_att_norm_val = F_att.norm(dim=1).cpu()
                F_rep_norm_val = F_rep.norm(dim=1).cpu()
                F_rot_norm_val = F_rot.norm(dim=1).cpu()
                F_res_norm_val = F_res.norm(dim=1).cpu()

                # 期望的运动方向是合力的方向
                desired_direction_vector = F.normalize(F_res, dim=1)
        
        # desired_direction_vector 现在是 detached 状态, 梯度不会通过它流向 self.position
        # --- [!!] 结束关键修正 [!!] ---


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
        # [!] 梯度从这里流向 self.velocity -> action -> policy
        desired_velocity = self.target_speed * desired_direction_vector
        velocity_difference = (desired_velocity - self.moving_average_velocity).norm(
            dim=1
        )
        loss_v = F.smooth_l1_loss(
            velocity_difference, th.zeros_like(velocity_difference), reduction="none"
        )

        # smoothness loss on acceleration, jerk, and body rate
        # [!] 梯度从这里流向 self.acceleration 等 -> self.velocity -> action -> policy
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
        # desired_yaw_vector = (
        #     slerp(self.target_direction, avg_vel, t).clone().detach()
        # )  # works, but should try without detach
        desired_yaw_vector = (
            slerp(desired_direction_vector, avg_vel, t).clone().detach()
        )
        # [!] 梯度从这里流向 self.yaw_vector -> self.rotation -> action -> policy
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
        
        # --- [修改点 4] 修正 metrics 字典 ---
        # 基础 metrics (这些都连接着计算图)
        metrics = {
            "loss_v": (lambda_v * loss_v).clone().detach().cpu(),
            "loss_vmax": (lambda_vmax * loss_vmax).clone().detach().cpu(),
            "loss_c": (lambda_c * loss_c).clone().detach().cpu(),
            "loss_a": (lambda_a * loss_a).clone().detach().cpu(),
            "loss_j": (lambda_j * loss_j).clone().detach().cpu(),
            "loss_om": (lambda_om * loss_om).clone().detach().cpu(),
            "loss_yaw": (lambda_yaw * loss_yaw).clone().detach().cpu(),
        }
        
        # 仅当不使用测地线时才记录APF/RAPF的metrics
        # (这些值来自 no_grad 块, 已经是 detached 的 cpu 张量)
        if not self.scene_manager.load_geodesics:
            metrics["F_att_norm"] = F_att_norm_val
            metrics["F_rep_norm"] = F_rep_norm_val
            metrics["F_rot_norm"] = F_rot_norm_val
            metrics["F_res_norm"] = F_res_norm_val
        # --- 结束 [修改点 4] ---

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