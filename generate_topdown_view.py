#!/usr/bin/env python3
import yaml
import argparse
import cv2
import os
from copy import deepcopy
import torch as th

# 导入项目所需的相关模块
from depthnav.envs.env_aliases import env_aliases

def generate_image(args):
    """加载环境并生成一张俯视图图片"""

    # 加载环境配置文件
    with open(args.cfg_file, "r") as file:
        config = yaml.safe_load(file)

    env_config = deepcopy(config["env"])
    env_config["num_envs"] = 1
    env_config["single_env"] = True
    
    # 设置渲染参数
    env_config["scene_kwargs"]["render_settings"] = {
        "mode": "fix",
        "view": "top",
        "sensor_type": "color",
        "resolution": [1024, 1024],
        "object_path": "./datasets/depthnav_dataset/configs/agents/DJI_Mavic_Mini_2.object_config.json",
    }

    print("正在初始化环境...")
    env_class = env_aliases[config["env_class"]]
    env = env_class(requires_grad=False, **env_config)

    print("正在加载场景...")
    env.reset()

    # --- 关键修正：将无人机移动到视野之外 ---
    print("正在将无人机移出视野...")
    # 定义一个远离场景的位置（例如，地板下方）
    hidden_position = th.tensor([[0.0, 0.0, -20.0]], device=env.device)
    # 使用 scene_manager 的 set_pose 方法来移动无人机模型
    env.scene_manager.set_pose(position=hidden_position, rotation=env.rot_wb.to_quat())

    print("环境加载成功，正在渲染俯视图...")
    top_down_image = env.scene_manager.render()[0]

    if top_down_image.shape[2] == 4:
        top_down_image = cv2.cvtColor(top_down_image, cv2.COLOR_RGBA2BGR)
    
    output_path = args.output_path
    cv2.imwrite(output_path, top_down_image)
    print(f"俯视图已成功保存到: {os.path.abspath(output_path)}")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为depthnav环境生成一张静态的俯视图图片。")
    parser.add_argument(
        "--cfg_file", 
        type=str, 
        default="examples/navigation/eval_cfg/nav_level1.yaml",
        help="用于加载场景的环境配置文件路径。"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="top_down_view.png",
        help="生成的俯视图图片的保存路径。"
    )
    args = parser.parse_args()
    
    generate_image(args)