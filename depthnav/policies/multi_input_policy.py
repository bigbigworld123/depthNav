import torch as th
from torch import nn
from typing import Type, Optional, Dict, Any, Union, List
from gymnasium import spaces

from .extractors import (
    FeatureExtractor,
    StateExtractor,
    StateTargetExtractor,
    ImageExtractor,
    StateImageExtractor,
    StateTargetImageExtractor,
)
from .mlp_policy import MlpPolicy


class LayerNormGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear_ih = nn.Linear(input_size, 3 * hidden_size)
        self.linear_hh = nn.Linear(hidden_size, 3 * hidden_size)
        self.ln_ih = nn.LayerNorm(3 * hidden_size)
        self.ln_hh = nn.LayerNorm(3 * hidden_size)

    def forward(self, x, h):
        gi = self.ln_ih(self.linear_ih(x))
        gh = self.ln_hh(self.linear_hh(h))
        i_r, i_z, i_n = gi.chunk(3, 1)
        h_r, h_z, h_n = gh.chunk(3, 1)

        r = th.sigmoid(i_r + h_r)
        z = th.sigmoid(i_z + h_z)
        n = th.tanh(i_n + r * h_n)
        h_next = (1 - z) * n + z * h
        return h_next


class MultiInputPolicy(MlpPolicy):
    """
    Builds an actor policy network with specifications from a dictionary.
    """

    feature_extractor_alias = {
        # "flatten": FlattenExtractor,
        "StateExtractor": StateExtractor,
        "ImageExtractor": ImageExtractor,
        "StateTargetExtractor": StateTargetExtractor,
        "StateImageExtractor": StateImageExtractor,
        "StateTargetImageExtractor": StateTargetImageExtractor,
    }
    recurrent_alias = {
        "GRUCell": th.nn.GRUCell,
        "LayerNormGRUCell": LayerNormGRUCell,
    }

    def __init__(
        self,
        observation_space: spaces.Space,
        net_arch: Dict[str, List[int]],
        activation_fn: Union[str, nn.Module],
        output_activation_fn: Union[str, nn.Module],
        feature_extractor_class: Type[FeatureExtractor],
        output_activation_kwargs: Optional[Dict[str, Any]] = None,
        feature_extractor_kwargs: Optional[Dict[str, Any]] = None,
        device: th.device = "cuda",
    ):
        if isinstance(feature_extractor_class, str):
            feature_extractor_class = self.feature_extractor_alias[
                feature_extractor_class
            ]
        feature_extractor_kwargs = feature_extractor_kwargs or {}
        
        self.use_motion_modulation = feature_extractor_kwargs.pop("use_motion_modulation", False)


        # get the size of features_dim before initializing MlpPolicy
        feature_extractor = feature_extractor_class(
            observation_space, **feature_extractor_kwargs
        )
        feature_norm = nn.LayerNorm(feature_extractor.features_dim)

        # add recurrent layer after feature_extractor
        _is_recurrent = False
        if net_arch.get("recurrent", None) is not None:
            _is_recurrent = True
            rnn_setting = net_arch.get("recurrent")
            rnn_class = rnn_setting.get("class")
            kwargs = rnn_setting.get("kwargs")

            if isinstance(rnn_class, str):
                rnn_class = self.recurrent_alias[rnn_class]

            recurrent_extractor = rnn_class(
                input_size=feature_extractor.features_dim, **kwargs
            )
            in_dim = kwargs.get("hidden_size")
        else:
            in_dim = feature_extractor.features_dim

        super().__init__(
            in_dim,
            net_arch,
            activation_fn,
            output_activation_fn,
            output_activation_kwargs,
            device,
        )

        self.feature_extractor = feature_extractor
        self.feature_norm = feature_norm
        if _is_recurrent:
            self._is_recurrent = True
            self._latent_dim = in_dim
            self.recurrent_extractor = recurrent_extractor

        if self.use_motion_modulation:
            # 运动信息维度为6 (3维线速度 + 3维角速度)
            ego_motion_dim = 6 
            # 调节器的输出维度必须和特征提取器的输出维度一致
            features_dim = feature_extractor.features_dim
            
            self.motion_modulator = nn.Sequential(
                nn.Linear(ego_motion_dim, features_dim),
                nn.Sigmoid() # 使用Sigmoid将输出缩放到0-1之间，作为门控信号
            ).to(device)

    def forward(self, obs, latent=None):
        features = self.feature_extractor(obs)
        features = self.feature_norm(features)
        if self.use_motion_modulation:
            # 从 state 观测中提取运动信息 (后6个维度)
            # state 格式: [quat(4), lin_vel(3), ang_vel(3)], 所以取最后6维
            ego_motion = obs["state"][:, -6:]
            
            # 通过调节器生成门控信号
            modulation_gate = self.motion_modulator(ego_motion)
            
            # 将门控信号与特征逐元素相乘，实现调节
            features = features * modulation_gate
            
        if self.is_recurrent:
            latent = self.recurrent_extractor(features, latent)
            actions = super().forward(latent)
            return actions, latent

        actions = super().forward(features)
        return actions