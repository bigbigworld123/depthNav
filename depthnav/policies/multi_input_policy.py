# depthnav/policies/multi_input_policy.py

import torch as th
from torch import nn
import torch.nn.functional as F
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

# ==========================================================================================
# 步驟 1.1: 創建我們全新的 MotionModulatedSRUCell (保持不變)
# ==========================================================================================
class MotionModulatedSRUCell(nn.Module):
    """
    一個融合了您原創的運動調節器和論文SRU思想的新型循環單元。
    """
    def __init__(self, input_size, hidden_size, motion_info_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.linear_ih = nn.Linear(input_size, 3 * hidden_size)
        self.linear_hh = nn.Linear(hidden_size, 3 * hidden_size)
        
        self.motion_modulator = nn.Sequential(
            nn.Linear(motion_info_size, hidden_size),
            nn.Tanh()
        )

    def forward(self, features, motion_info, h):
        gi = self.linear_ih(features)
        gh = self.linear_hh(h)
        i_r, i_z, i_n = gi.chunk(3, 1)
        h_r, h_z, h_n = gh.chunk(3, 1)

        resetgate = th.sigmoid(i_r + h_r)
        updategate = th.sigmoid(i_z + h_z)
        
        s_t = self.motion_modulator(motion_info)
        newgate_input = i_n + resetgate * h_n
        h_bar_t = th.tanh((1 + s_t) * newgate_input)

        h_next = (1 - updategate) * h + updategate * h_bar_t
        
        return h_next

# ==========================================================================================
# 步驟 1.2: 創建輕量化的拓撲記憶圖類 (保持不變)
# ==========================================================================================
class TopologicalMemory(nn.Module):
    def __init__(self, memory_size=100, feature_dim=192, tau_new=0.85, d_min=1.5, device="cpu"):
        super().__init__()
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        self.tau_new = tau_new
        self.d_min = d_min
        self.device = device

        self.register_buffer('node_features', th.zeros(memory_size, feature_dim))
        self.register_buffer('node_positions', th.zeros(memory_size, 3))
        self.register_buffer('adjacency_matrix', th.zeros(memory_size, memory_size))
        
        self.register_buffer('ptr', th.tensor([0], dtype=th.long))
        self.register_buffer('num_nodes', th.tensor([0], dtype=th.long))
        self.register_buffer('last_visited_node_idx', th.zeros(16, dtype=th.long)-1) # 假設最大batch size為16

    @th.no_grad()
    def find_most_similar(self, h_t):
        if self.num_nodes == 0:
            return -1.0, -1

        active_nodes = self.node_features[:self.num_nodes]
        similarities = F.cosine_similarity(h_t.unsqueeze(1), active_nodes.unsqueeze(0), dim=-1)
        max_similarity, most_similar_idx = th.max(similarities, dim=1)
        
        return max_similarity.item(), most_similar_idx.item()

    @th.no_grad()
    def update_and_get_info(self, h_t, current_position, dones):
        batch_size = h_t.size(0)
        topo_infos = th.zeros(batch_size, 1, device=self.device)

        for i in range(batch_size):
            if dones[i]:
                self.last_visited_node_idx[i] = -1
                continue

            h_i = h_t[i].unsqueeze(0)
            pos_i = current_position[i].unsqueeze(0)

            if self.num_nodes < 1:
                self.node_features[0] = h_i.squeeze(0)
                self.node_positions[0] = pos_i.squeeze(0)
                self.num_nodes += 1
                self.last_visited_node_idx[i] = 0
                current_node_idx = 0
            else:
                max_similarity, most_similar_idx = self.find_most_similar(h_i)
                
                should_add_new_node = False
                if max_similarity < self.tau_new:
                    should_add_new_node = True
                else:
                    dist_to_similar = th.norm(pos_i - self.node_positions[most_similar_idx])
                    if dist_to_similar > self.d_min:
                        should_add_new_node = True

                if should_add_new_node and self.num_nodes < self.memory_size:
                    ptr = self.num_nodes.item()
                    self.node_features[ptr] = h_i.squeeze(0)
                    self.node_positions[ptr] = pos_i.squeeze(0)
                    current_node_idx = ptr
                    self.num_nodes += 1
                elif should_add_new_node:
                    ptr = self.ptr.item()
                    self.node_features[ptr] = h_i.squeeze(0)
                    self.node_positions[ptr] = pos_i.squeeze(0)
                    current_node_idx = ptr
                    self.ptr[0] = (ptr + 1) % self.memory_size
                else:
                    current_node_idx = most_similar_idx
                    alpha = 0.5
                    self.node_features[current_node_idx] = alpha * self.node_features[current_node_idx] + (1 - alpha) * h_i.squeeze(0)
                    self.node_positions[current_node_idx] = alpha * self.node_positions[current_node_idx] + (1 - alpha) * pos_i.squeeze(0)

            last_idx = self.last_visited_node_idx[i].item()
            if last_idx != -1 and last_idx != current_node_idx:
                self.adjacency_matrix[last_idx, current_node_idx] = 1
                self.adjacency_matrix[current_node_idx, last_idx] = 1
            
            self.last_visited_node_idx[i] = current_node_idx
            
            num_neighbors = self.adjacency_matrix[current_node_idx].sum()
            topo_infos[i] = num_neighbors
        
        return topo_infos

    def reset(self):
        self.num_nodes.fill_(0)
        self.ptr.fill_(0)
        self.last_visited_node_idx.fill_(-1)
        self.adjacency_matrix.fill_(0)


# 沿用原有的 LayerNormGRUCell
class LayerNormGRUCell(nn.Module):
    # ... (保持原樣，無需修改)
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


# ==========================================================================================
# 步驟 1.3: 修改 MultiInputPolicy 以集成所有新模塊
# ==========================================================================================
class MultiInputPolicy(MlpPolicy):
    feature_extractor_alias = {
        "StateExtractor": StateExtractor,
        "ImageExtractor": ImageExtractor,
        "StateTargetExtractor": StateTargetExtractor,
        "StateImageExtractor": StateImageExtractor,
        "StateTargetImageExtractor": StateTargetImageExtractor,
    }
    recurrent_alias = {
        "GRUCell": th.nn.GRUCell,
        "LayerNormGRUCell": LayerNormGRUCell,
        "MotionModulatedSRUCell": MotionModulatedSRUCell,
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
        # 提前處理 kwargs
        feature_extractor_kwargs = feature_extractor_kwargs or {}

        # ===================== 核心修正 =====================
        # 1. 從 kwargs 中 "彈出" MultiInputPolicy 專用的參數
        self.use_topological_memory = feature_extractor_kwargs.pop("use_topological_memory", False)
        topological_memory_kwargs = feature_extractor_kwargs.pop("topological_memory_kwargs", {})
        # ====================================================
        
        # 創建特徵提取器 (現在傳入的是 "乾淨" 的 kwargs)
        if isinstance(feature_extractor_class, str):
            feature_extractor_class = self.feature_extractor_alias[feature_extractor_class]
        feature_extractor = feature_extractor_class(observation_space, **feature_extractor_kwargs)
        feature_norm = nn.LayerNorm(feature_extractor.features_dim)
        
        # 處理循環層
        _is_recurrent = False
        if net_arch.get("recurrent", None) is not None:
            _is_recurrent = True
            rnn_setting = net_arch.get("recurrent")
            rnn_class_str = rnn_setting.get("class")
            kwargs = rnn_setting.get("kwargs")
            rnn_class = self.recurrent_alias[rnn_class_str]

            if rnn_class_str == "MotionModulatedSRUCell":
                recurrent_extractor = rnn_class(
                    input_size=feature_extractor.features_dim, 
                    motion_info_size=6, 
                    **kwargs
                )
            else:
                recurrent_extractor = rnn_class(
                    input_size=feature_extractor.features_dim, **kwargs
                )
            
            in_dim = kwargs.get("hidden_size")
        else:
            in_dim = feature_extractor.features_dim

        # 計算決策層的輸入維度
        in_dim_mlp = in_dim
        if self.use_topological_memory:
            in_dim_mlp += 1

        super().__init__(
            in_dim_mlp,
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

        if self.use_topological_memory:
            self.memory = TopologicalMemory(feature_dim=in_dim, device=device, **topological_memory_kwargs)

    def forward(self, obs, latent=None, dones=None):
        features = self.feature_extractor(obs)
        features = self.feature_norm(features)

        if self.is_recurrent:
            if isinstance(self.recurrent_extractor, MotionModulatedSRUCell):
                motion_info = obs["state"][:, -6:]
                latent = self.recurrent_extractor(features, motion_info, latent)
            else:
                latent = self.recurrent_extractor(features, latent)
            
            final_features = latent
        else:
            final_features = features

        if self.use_topological_memory:
            current_position = obs["position"]
            if dones is None:
                dones = th.zeros(current_position.size(0), dtype=th.bool, device=self.device)
            topo_info = self.memory.update_and_get_info(final_features, current_position, dones)
            fused_features = th.cat([final_features, topo_info], dim=-1)
        else:
            fused_features = final_features
        
        actions = self.policy_net(fused_features)
        
        if self.is_recurrent:
            return actions, latent
        return actions