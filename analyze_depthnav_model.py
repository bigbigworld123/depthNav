import torch
import torch.nn as nn
import numpy as np
import sys
from gymnasium import spaces
from thop import profile, clever_format
from typing import List, Optional, Type, Union, Dict, Any
from abc import abstractmethod

# ====================================================================
# 辅助函数: 修复 NameError
# ====================================================================

def count_parameters(model):
    """
    计算模型中可训练参数的总量。
    """
    # 修复 NameError
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ====================================================================
# 1. 核心模型类和函数定义 (从 rislab/depthnav 源码提取)
#    (代码内容与您之前接收的修正版保持一致，这里仅为完整性展示)
# ====================================================================

# --- 辅助函数：创建 MLP 和 CNN ---

def create_mlp(
    input_dim: int,
    layer: List[int],
    output_dim: Optional[int] = None,
    activation_fn: Type[nn.Module] = nn.LeakyReLU,
) -> nn.Module:
    modules = []
    last_layer_dim = input_dim
    for curr_layer_dim in layer:
        modules.append(nn.Linear(last_layer_dim, curr_layer_dim))
        modules.append(activation_fn())
        last_layer_dim = curr_layer_dim

    if output_dim is not None:
        modules.append(nn.Linear(last_layer_dim, output_dim))
        
    if len(layer) > 0:
        modules.insert(1, nn.LayerNorm(layer[0]))
    
    return nn.Sequential(*modules)

def create_cnn(
    input_channels: int,
    kernel_size: List[int],
    channel: List[int],
    stride: List[int],
    padding: List[int],
) -> nn.Module:
    modules = []
    in_channels = input_channels
    for idx in range(len(channel)):
        modules.append(
            nn.Conv2d(
                in_channels,
                channel[idx],
                kernel_size=kernel_size[idx],
                stride=stride[idx],
                padding=padding[idx],
            )
        )
        modules.append(nn.LeakyReLU())
        in_channels = channel[idx]

    modules.append(nn.Flatten())
    return nn.Sequential(*modules)

def set_mlp_feature_extractor(cls, name, observation_space, net_arch, activation_fn):
    layer = net_arch.get("mlp_layer", [])
    features_dim = layer[-1] if len(layer) != 0 else observation_space.shape[0]
    input_dim = observation_space.shape[0]

    mlp = create_mlp(
        input_dim=input_dim,
        layer=net_arch.get("mlp_layer", []),
        activation_fn=activation_fn,
    )
    setattr(cls, name + "_extractor", mlp)
    return features_dim

# --- FeatureExtractor 抽象类和具体实现 ---

class FeatureExtractor(nn.Module):
    activation_fn_alias = {"leaky_relu": nn.LeakyReLU, "identity": nn.Identity}
    
    def __init__(self, observation_space: spaces.Dict, net_arch: Dict, activation_fn: Union[str, Type[nn.Module]]):
        super().__init__()
        self._features_dim = 1
        self.activation_fn = self.activation_fn_alias[activation_fn] if isinstance(activation_fn, str) else activation_fn
        self._build(observation_space, net_arch, self.activation_fn)
    
    @abstractmethod
    def _build(self, observation_space, net_arch, activation_fn): pass

    @abstractmethod
    def extract(self, observations) -> torch.Tensor: pass

    def forward(self, observations): return self.extract(observations)
    @property
    def features_dim(self): return self._features_dim

class ImageExtractor(FeatureExtractor):
    def _get_conv_output(self, net, shape):
        net.eval()
        with torch.no_grad():
            image = torch.rand(1, *shape)
            output = net(image)
        net.train()
        return output.numel()

    def set_cnn_feature_extractor(self, name, observation_space, net_arch, activation_fn):
        in_channels = observation_space.shape[0]
        H, W = net_arch.get("input_max_pool_H_W")
        self.input_max_pool_H_W = [H, W] 
        observation_shape = (in_channels, H, W)
        
        cnn = create_cnn(
            input_channels=in_channels,
            kernel_size=net_arch["kernel_size"],
            channel=net_arch["channels"],
            stride=net_arch["stride"],
            padding=net_arch["padding"],
        )
        _cnn_out_dim = self._get_conv_output(cnn, observation_shape)
        
        mlp_layer = net_arch.get("mlp_layer", [])
        full_cnn_mlp = nn.Sequential(
            cnn,
            create_mlp(input_dim=_cnn_out_dim, layer=mlp_layer, activation_fn=activation_fn)
        )
        _final_dim = mlp_layer[-1]

        setattr(self, name + "_extractor", full_cnn_mlp)
        self._image_extractor_names.append(name + "_extractor")
        return _final_dim

    def preprocess_depth(self, depth: torch.Tensor):
        inv_depth = 1.0 / (depth.float() + 1e-6)
        return inv_depth

    def _build(self, observation_space, net_arch, activation_fn):
        _image_features_dims = 0
        self._image_extractor_names = []
        for key in net_arch.keys():
            if "depth" in key:
                _image_features_dims += self.set_cnn_feature_extractor(
                    key, observation_space[key], net_arch.get(key, {}), activation_fn
                )
        self._features_dim = _image_features_dims

    def extract(self, observations) -> torch.Tensor:
        features = []
        for name in self._image_extractor_names:
            image = observations[name.split("_")[0]]
            image = self.preprocess_depth(image)
            x = getattr(self, name)(image)
            features.append(x)
        return torch.cat(features, dim=1)

class StateTargetImageExtractor(ImageExtractor):
    def _build(self, observation_space, net_arch, activation_fn):
        super()._build(observation_space, net_arch, activation_fn)
        
        _state_features_dim = set_mlp_feature_extractor(
            self, "state", observation_space["state"], net_arch.get("state", {}), activation_fn
        )
        _target_features_dim = set_mlp_feature_extractor(
            self, "target", observation_space["target"], net_arch.get("target", {}), activation_fn
        )

        self.concatenate = net_arch.get("concatenate", True)
        if self.concatenate:
            self._features_dim = _state_features_dim + _target_features_dim + self._features_dim
        else:
            self._features_dim = self._features_dim
            
    def extract(self, observation):
        state_features = self.state_extractor(observation["state"])
        target_features = self.target_extractor(observation["target"])
        image_features = super().extract(observation)
        if self.concatenate:
            return torch.cat([state_features, target_features, image_features], dim=1)
        else:
            return state_features + target_features + image_features

# --- 策略核心类 ---

class AccelerationBoundedYaw(nn.Module):
    def forward(self, z):
        acc = z[:, 0:3]
        yaw = z[:, 3]
        bounded_yaw = -torch.pi + (2 * torch.pi) * torch.sigmoid(yaw) 
        return torch.cat([acc, bounded_yaw.unsqueeze(1)], dim=1)

class MlpPolicy(nn.Module):
    output_activation_fn_alias = {"acceleration_bounded_yaw": AccelerationBoundedYaw, "identity": nn.Identity}

    def __init__(self, in_dim: int, net_arch: Dict, activation_fn: str, output_activation_fn: str, output_activation_kwargs: Optional[Dict] = None, device: str = "cpu"):
        super().__init__()
        pi_layers_dims = net_arch.get("mlp_layer", [])
        
        modules = []
        last_layer_dim_pi = in_dim
        modules.append(nn.Linear(last_layer_dim_pi, pi_layers_dims[0]))
        
        output_fn = self.output_activation_fn_alias[output_activation_fn]
        modules.append(output_fn(**(output_activation_kwargs or {})))
        
        self.policy_net = nn.Sequential(*modules)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor: return self.policy_net(features)

# --- MultiInputPolicy (将所有组件集成) ---

class MultiInputPolicy(MlpPolicy):
    feature_extractor_alias = {"StateTargetImageExtractor": StateTargetImageExtractor}
    
    def __init__(self, observation_space: spaces.Space, net_arch: Dict, activation_fn: str, output_activation_fn: str, feature_extractor_class: str, feature_extractor_kwargs: Dict, **kwargs):
        feature_extractor_class = self.feature_extractor_alias[feature_extractor_class]
        
        feature_extractor = feature_extractor_class(
            observation_space, **feature_extractor_kwargs
        )
        
        rnn_setting = net_arch.get("recurrent")
        rnn_class = nn.GRUCell 
        rnn_kwargs = rnn_setting.get("kwargs")
        in_dim = rnn_kwargs.get("hidden_size")
            
        super().__init__(in_dim, net_arch, activation_fn, output_activation_fn, **kwargs)
        
        self.feature_extractor = feature_extractor
        self.feature_norm = nn.LayerNorm(feature_extractor.features_dim)
        self.recurrent_extractor = rnn_class(input_size=feature_extractor.features_dim, **rnn_kwargs)
        self._latent_dim = in_dim
        self._is_recurrent = True

    def forward(self, obs, latent=None):
        features = self.feature_extractor(obs)
        features = self.feature_norm(features)
        latent = self.recurrent_extractor(features, latent) 
        actions = super().forward(latent)
        return actions, latent
    
    @property
    def latent_dim(self):
        return self._latent_dim

# ====================================================================
# 2. 配置与计算
# ====================================================================

# --- 配置常量 (基于 small_yaw.yaml 和您的 10D 状态修改) ---

STATE_DIM_CORRECT = 10     # 您的修改：Quat(4) + Vel(3) + Omega(3)
TARGET_DIM_CORRECT = 4     # 修正：Target_Vel(3) + Inv_Dist(1)
DEPTH_H_CORRECT = 72       
DEPTH_W_CORRECT = 128      

POLICY_NET_KWARGS = {
    "recurrent": {"class": "GRUCell", "kwargs": {"hidden_size": 192}},
    "mlp_layer": [4] 
}
POLICY_FEAT_KWARGS = {
    "activation_fn": "leaky_relu",
    "net_arch": {
        "state": {"mlp_layer": [192], "ln": True},
        "target": {"mlp_layer": [192], "ln": True},
        "depth": {
            "input_max_pool_H_W": [12, 16], 
            "kernel_size": [2, 3, 3],
            "channels": [32, 64, 128],
            "padding": [0, 0, 0],
            "stride": [1, 1, 1],
            "mlp_layer": [192],
            "ln": True
        },
        "concatenate": False
    }
}
H_POOLED, W_POOLED = POLICY_FEAT_KWARGS['net_arch']['depth']['input_max_pool_H_W']


# --- 实例化模型并计算 ---

def main():
    # 1. 定义修正后的观测空间
    OBSERVATION_SPACE = spaces.Dict({
        "state": spaces.Box(low=-np.inf, high=np.inf, shape=(STATE_DIM_CORRECT,), dtype=np.float32),
        "target": spaces.Box(low=-np.inf, high=np.inf, shape=(TARGET_DIM_CORRECT,), dtype=np.float32),
        "depth": spaces.Box(low=0, high=10, shape=(1, DEPTH_H_CORRECT, DEPTH_W_CORRECT), dtype=np.float32) 
    })

    # 2. 实例化模型
    model = MultiInputPolicy(
        observation_space=OBSERVATION_SPACE,
        net_arch=POLICY_NET_KWARGS,
        activation_fn="leaky_relu",
        output_activation_fn="acceleration_bounded_yaw",
        feature_extractor_class="StateTargetImageExtractor",
        feature_extractor_kwargs=POLICY_FEAT_KWARGS,
        device="cpu"
    )
    model.eval()

    # 3. 准备正确的模拟输入 (Batch Size = 1)
    obs_dummy = {
        "state": torch.randn(1, STATE_DIM_CORRECT),
        "target": torch.randn(1, TARGET_DIM_CORRECT),
        "depth": torch.randn(1, 1, H_POOLED, W_POOLED), # 使用 Max-Pooled 后的尺寸
    }
    latent_state_dummy = torch.randn(1, model.latent_dim) 

    # 4. 计算 FLOPs 和参数
    flops, params = profile(model, inputs=(obs_dummy, latent_state_dummy), verbose=False)
    flops_fmt, params_fmt = clever_format([flops, params], "%.2f")
    total_params = count_parameters(model)

    print("=" * 60)
    print(f"DepthNav 模型复杂度报告 (10D State - 最终修正)")
    print("=" * 60)
    print(f"✅ 总参数量 (Total Parameters): {total_params:,} ({params_fmt})")
    print(f"✅ 总计算量 (FLOPs/Step):      {flops_fmt}")
    print("-" * 60)
    print("修正后的输入配置:")
    print(f"  - State (D):     {STATE_DIM_CORRECT} (Quat + Vel + Omega)")
    print(f"  - Target (D):    {TARGET_DIM_CORRECT} (Vel + Inv. Dist.)")
    print(f"  - Depth (Pooled):{H_POOLED} x {W_POOLED}")
    print(f"  - Recurrent Dim: {model.latent_dim}")
    print("=" * 60)
    
if __name__ == "__main__":
    main()