# In file: examples/navigation/create_true_maze.py

import os
import json
from copy import deepcopy
from depthnav.common import std_to_habitat

# 场景JSON模板，定义了基础环境
empty_scene = {
    "stage_instance": {
        "template_name": "stages/box_2", # 使用一个空旷的大房间作为背景
    },
    "object_instances": [],
    "articulated_object_instances": [],
    "default_lighting": "lighting/garage_v1_0",
    "user_defined": {},
}

def add_wall(scene_objects, center_pos, size):
    """一个辅助函数，用于在场景中添加一堵墙（一个被缩放的立方体）"""
    wall = {
        "template_name": "primitives/unit/unit_cube", # 墙的基础模型是一个单位立方体
        "translation": std_to_habitat(center_pos, None)[0].tolist(),
        "rotation": [1, 0, 0, 0], # 不旋转
        "non_uniform_scale": size.tolist(), # 非均匀缩放，把它拉伸成墙的形状
        "motion_type": "STATIC",
        "translation_origin": "COM",
    }
    scene_objects.append(wall)

def main():
    # 为我们的新迷宫场景命名
    scene_name = "custom_scenes/true_maze_eval"
    dataset_path = "./datasets/depthnav_dataset"
    save_dir = os.path.join(dataset_path, "configs", scene_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 准备场景JSON
    scene_json = deepcopy(empty_scene)

    # --- 使用 add_wall 函数来精确地建造迷宫 ---
    # 尺寸单位：[x方向长度, y方向长度, z方向高度]
    # 位置单位：[x, y, z] 中心点坐标

    # 1. 中间的长竖墙，它制造了左右两个通道
    add_wall(scene_json["object_instances"], center_pos=th.tensor([0.0, 0.0, 2.0]), size=th.tensor([1.0, 20.0, 4.0]))

    # 2. 上方的横墙，迫使机器人必须从左侧进入
    add_wall(scene_json["object_instances"], center_pos=th.tensor([5.0, 5.0, 2.0]), size=th.tensor([10.0, 1.0, 4.0]))

    # 3. 下方的横墙，迫使机器人必须从右侧离开
    add_wall(scene_json["object_instances"], center_pos=th.tensor([-5.0, -5.0, 2.0]), size=th.tensor([10.0, 1.0, 4.0]))

    # 保存最终的场景文件
    scene_filename = f"{os.path.basename(scene_name)}_0.scene_instance.json"
    save_path = os.path.join(save_dir, scene_filename)
    with open(save_path, "w") as f:
        json.dump(scene_json, f, indent=4)

    print(f"成功生成真正的迷宫场景，保存在: {os.path.abspath(save_path)}")

    # --- 更新数据集配置文件，让系统能找到新场景 ---
    summary_path = os.path.join(dataset_path, "depthnav_dataset.scene_dataset_config.json")
    with open(summary_path, "r") as f:
        summary = json.load(f)

    scene_config_path = f"configs/{scene_name}"
    if scene_config_path not in summary["scene_instances"]["paths"][".json"]:
        summary["scene_instances"]["paths"][".json"].append(scene_config_path)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"已更新数据集配置文件: {summary_path}")

    # (可选) 生成后直接用查看器预览
    os.system(f"python depthnav/scripts/scene_viewer.py --scene {scene_config_path}")

if __name__ == "__main__":
    # 需要导入 torch，因为 add_wall 使用了 th.tensor
    import torch as th
    main()