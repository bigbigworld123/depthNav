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
        feature_extractor_class: Type[FeatureExtractor], # <<< 修正：添加缺失的參數
        policy_kwargs: Dict[str, Any],
        output_activation_kwargs: Optional[Dict[str, Any]] = None,
        feature_extractor_kwargs: Optional[Dict[str, Any]] = None,
        device: th.device = "cuda",
    ):
        
        # ================================================================
        # <<< START REFACTORING: 錯誤修復 >>>
        #
        # 1. DO NOT CALL SUPER() YET.
        # 我們必須首先計算 mlp_in_dim (MlpPolicy 的輸入維度)
        # ================================================================
        
        if isinstance(feature_extractor_class, str):
            feature_extractor_class = self.feature_extractor_alias[
                feature_extractor_class
            ]
        feature_extractor_kwargs = feature_extractor_kwargs or {}
        
        # (獲取標記，但還不要分配 nn.Module)
        use_motion_modulation = feature_extractor_kwargs.pop("use_motion_modulation", False)

        # (在本地初始化 feature_extractor 以獲取其維度)
        _feature_extractor = feature_extractor_class(
            observation_space, **feature_extractor_kwargs
        )
        features_dim = _feature_extractor.features_dim

        # --- Recurrent & Attention Logic (用於計算 mlp_in_dim) ---
        _is_recurrent = False
        use_temporal_attention = policy_kwargs.get("use_temporal_attention", False)
        attention_k_steps = policy_kwargs.get("attention_k_steps", 10)
        
        recurrent_extractor_instance = None # 佔位符
        W_q_instance = None
        W_k_instance = None
        W_v_instance = None
        
        if net_arch.get("recurrent", None) is not None:
            _is_recurrent = True
            rnn_setting = net_arch.get("recurrent")
            rnn_class_str = rnn_setting.get("class")
            kwargs = rnn_setting.get("kwargs")
            
            if isinstance(rnn_class_str, str):
                rnn_class = self.recurrent_alias[rnn_class_str]

            # (在本地初始化，還不要分配給 self)
            recurrent_extractor_instance = rnn_class(
                input_size=features_dim, **kwargs
            ).to(device) # .to(device) 很重要
            
            hidden_size = kwargs.get("hidden_size")
            _latent_dim = hidden_size

            if use_temporal_attention:
                # (在本地初始化)
                W_q_instance = nn.Linear(hidden_size, hidden_size).to(device)
                W_k_instance = nn.Linear(hidden_size, hidden_size).to(device)
                W_v_instance = nn.Linear(hidden_size, hidden_size).to(device)
                
                mlp_in_dim = hidden_size * 2 # [h_t, context]
            else:
                mlp_in_dim = hidden_size # [h_t]
        else:
            mlp_in_dim = features_dim # 非循環
            _latent_dim = 0
        
        # ================================================================
        # 2. NOW CALL SUPER().__INIT__()
        # (現在調用父構造函數是安全的)
        # ================================================================
        super().__init__(
            mlp_in_dim,
            net_arch,
            activation_fn,
            output_activation_fn,
            output_activation_kwargs,
            device,
        )
        
        # ================================================================
        # 3. NOW ASSIGN ALL nn.Module ATTRIBUTES
        # (super() 已經運行, self._modules 已初始化)
        # ================================================================
        self.feature_extractor = _feature_extractor
        self.feature_norm = nn.LayerNorm(features_dim).to(device)
        
        self.use_motion_modulation = use_motion_modulation
        if self.use_motion_modulation:
            motion_input_dim = 6 
            modulation_layer_size = 64
            # (現在分配是安全的)
            self.motion_modulator = nn.Sequential(
                nn.Linear(motion_input_dim, modulation_layer_size),
                nn.LeakyReLU(),
                nn.Linear(modulation_layer_size, features_dim),
                nn.Sigmoid(),
            ).to(device)

        self._is_recurrent = _is_recurrent
        self._latent_dim = _latent_dim
        self.use_temporal_attention = use_temporal_attention
        self.attention_k_steps = attention_k_steps
        
        if self._is_recurrent:
            # (現在分配是安全的)
            self.recurrent_extractor = recurrent_extractor_instance
        
        if self.use_temporal_attention:
            # (現在分配是安全的)
            self.W_q = W_q_instance
            self.W_k = W_k_instance
            self.W_v = W_v_instance
        
        # <<< END REFACTORING >>>


    def forward(self, obs, latent=None):
        features = self.feature_extractor(obs)
        features = self.feature_norm(features)
        
        if self.use_motion_modulation:
            # 從 state 觀測中提取運動信息 (後6個維度)
            ego_motion = obs["state"][:, -6:]
            
            # 通過調節器生成門控信號
            modulation_gate = self.motion_modulator(ego_motion)
            
            # 將門控信號與特徵逐元素相乘，實現調節
            features = features * modulation_gate
            
        if self.is_temporal_attention:
            # latent 此時是一個元組 (h_prev, history_buffer_prev)
            h_prev, history_buffer_prev = latent
            
            # 1. GRU Step: 得到當前隱藏狀態 h_t
            h_t = self.recurrent_extractor(features, h_prev)
            
            # 2. Attention Step
            query = self.W_q(h_t.unsqueeze(1)) # (B, 1, H)
            keys = self.W_k(history_buffer_prev)     # (B, K, H)
            values = self.W_v(history_buffer_prev)   # (B, K, H)
            
            attn_scores = th.bmm(query, keys.transpose(-1, -2)) # (B, 1, K)
            attn_dist = F.softmax(attn_scores, dim=-1)         # (B, 1, K)
            
            context = th.bmm(attn_dist, values).squeeze(1)     # (B, H)
            
            # 3. Concatenate (TA_state)
            ta_state = th.cat([h_t, context], dim=-1)
            
            # 4. Final MLP
            actions = super().forward(ta_state)
            
            # 5. 更新歷史緩衝區
            new_history_buffer = th.cat([h_t.unsqueeze(1), history_buffer_prev[:, :-1, :]], dim=1)
            
            # 6. 返回新的狀態元組
            return actions, (h_t, new_history_buffer)

        elif self.is_recurrent:
            # 原始 GRU 邏輯 (latent 是 h_prev)
            latent = self.recurrent_extractor(features, latent)
            actions = super().forward(latent)
            return actions, latent

        # 非循環邏輯
        actions = super().forward(features)
        return actions
        
    @property
    def is_temporal_attention(self):
        """標記是否啟用時間注意力"""
        return self._is_recurrent and self.use_temporal_attention

    @property
    def attention_history_shape(self):
        """返回 (K, H) 以便 BPTT 初始化緩衝區"""
        return (self.attention_k_steps, self._latent_dim)