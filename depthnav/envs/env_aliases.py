from typing import Dict
from .base_env import BaseEnv
from .hover_env import HoverEnv_Flightning, HoverEnv_BNL
from .navigation_env import NavigationEnv
from .navigation2_env import Navigation2Env

env_aliases: Dict[str, BaseEnv] = {
    "hover_env_flightning": HoverEnv_Flightning,
    "hover_env_bnl": HoverEnv_BNL,
    "navigation_env": NavigationEnv,
    "navigation2_env": Navigation2Env,
}