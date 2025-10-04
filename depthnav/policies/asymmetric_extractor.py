# In file: depthnav/policies/asymmetric_extractor.py

import torch as th
import torch.nn as nn
from gymnasium import spaces
from typing import Type

# 导入基类和辅助函数
from .extractors import FeatureExtractor, create_mlp, create_cnn

class AsymmetricExtractor(FeatureExtractor):
    """
    非对称特征融合提取器。
    - state 和 target 使用较大维度的MLP进行编码。
    - depth 使用轻量化的CNN和一个小MLP编码成低维摘要。
    - 最后通过一个融合MLP将所有特征进行整合。
    """
    def __init__(
        self,
        observation_space: spaces.Dict,
        net_arch: dict = {},
        activation_fn: Type[nn.Module] = nn.ReLU,
    ):
        # 确保观测空间包含所需的所有键
        assert all(k in observation_space.spaces for k in ["state", "target", "depth"])
        super().__init__(observation_space, net_arch, activation_fn)

    def _build(self, observation_space, net_arch, activation_fn):
        """
        构建非对称的网络结构。
        """
        # --- 1. 为 state 和 target 创建编码器 ---
        state_arch = net_arch.get("state", {})
        target_arch = net_arch.get("target", {})
        
        self.state_extractor = create_mlp(
            input_dim=observation_space["state"].shape[0],
            layer=state_arch.get("mlp_layer", [128]), # 默认为128维输出
            activation_fn=activation_fn,
        )
        self.target_extractor = create_mlp(
            input_dim=observation_space["target"].shape[0],
            layer=target_arch.get("mlp_layer", [128]), # 默认为128维输出
            activation_fn=activation_fn,
        )
        state_features_dim = state_arch.get("mlp_layer", [128])[-1]
        target_features_dim = target_arch.get("mlp_layer", [128])[-1]

        # --- 2. 为 depth 创建轻量化编码器 ---
        depth_arch = net_arch.get("depth", {})
        
        # 定义一个轻量化的CNN
        self.depth_cnn = create_cnn(
            input_channels=observation_space["depth"].shape[0],
            kernel_size=depth_arch.get("kernel_size", [5, 3, 3]),
            channel=depth_arch.get("channels", [16, 32, 32]),
            stride=depth_arch.get("stride", [2, 2, 2]),
            padding=depth_arch.get("padding", [0, 0, 0]),
            activation_fn=activation_fn,
        )
        
        # 计算CNN输出维度
        with th.no_grad():
            dummy_input = th.zeros(1, *observation_space["depth"].shape)
            cnn_output_dim = self.depth_cnn(dummy_input).shape[1]
            
        # 在CNN后接一个小的MLP，用于降维
        self.depth_mlp = create_mlp(
            input_dim=cnn_output_dim,
            layer=depth_arch.get("mlp_layer", [32]), # 默认为32维输出
            activation_fn=activation_fn,
        )
        depth_features_dim = depth_arch.get("mlp_layer", [32])[-1]
        
        # --- 3. 创建融合网络 (Fusion Network) ---
        fusion_arch = net_arch.get("fusion", {})
        concatenated_dim = state_features_dim + target_features_dim + depth_features_dim
        
        self.fusion_mlp = create_mlp(
            input_dim=concatenated_dim,
            layer=fusion_arch.get("mlp_layer", [192]), # 默认融合后输出192维
            activation_fn=activation_fn,
        )
        
        # 最终特征维度由融合网络决定
        self._features_dim = fusion_arch.get("mlp_layer", [192])[-1]

    def extract(self, observations: th.Tensor) -> th.Tensor:
        """
        定义特征提取和融合的过程。
        """
        # 1. 分别提取 state, target, depth 的特征
        state_features = self.state_extractor(observations["state"])
        target_features = self.target_extractor(observations["target"])
        
        # 提取 depth 特征
        depth_cnn_features = self.depth_cnn(observations["depth"])
        depth_features = self.depth_mlp(depth_cnn_features)
        
        # 2. 拼接 (Concatenate) 所有特征
        concatenated_features = th.cat(
            [state_features, target_features, depth_features], dim=1
        )
        
        # 3. 通过融合网络进行最终的特征提炼
        final_features = self.fusion_mlp(concatenated_features)
        
        return final_features