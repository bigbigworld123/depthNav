import os
import json
from copy import deepcopy

# 确保我们从一个完全空白的场景开始
empty_scene = {
    "stage_instance": {"template_name": ""},
    "object_instances": [],
    "articulated_object_instances": [],
    "default_lighting": "lighting/garage_v1_0",
    "user_defined": {},
}

def add_wall(scene_objects, center_pos, size):
    """一个通用的辅助函数，用于在Habitat坐标系 (Y-up) 中添加墙壁。"""
    wall = {
        "template_name": "primitives/unit/unit_cube",
        "translation": center_pos,
        "rotation": [1, 0, 0, 0],
        "non_uniform_scale": size,
        "motion_type": "STATIC",
        "translation_origin": "COM",
    }
    scene_objects.append(wall)

def main():
    # 场景名字可以自定义
    scene_name = "custom_scenes/target_maze"
    dataset_path = "./datasets/depthnav_dataset"
    save_dir = os.path.join(dataset_path, "configs", scene_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    scene_json = deepcopy(empty_scene)

    # --- 定义迷宫的尺寸参数 ---
    WALL_HEIGHT = 4.0
    WALL_THICKNESS = 1.0
    FLOOR_THICKNESS = 0.1
    WALL_CENTER_Y = (FLOOR_THICKNESS / 2.0) + (WALL_HEIGHT / 2.0)

    # 1. 添加地板 (顶面在 Y=0)
    add_wall(scene_json["object_instances"],
             center_pos=[0.0, -FLOOR_THICKNESS / 2.0, 0.0],
             size=[22.0, FLOOR_THICKNESS, 22.0])

    # --- 严格按照目标图片布局，构建“完美拼接”的墙壁 ---
    
    # 外围墙壁 (四条独立的、不重叠的墙)
    add_wall(scene_json["object_instances"], [0.0, WALL_CENTER_Y, 10.0], [20.0, WALL_HEIGHT, WALL_THICKNESS])  # 上
    add_wall(scene_json["object_instances"], [0.0, WALL_CENTER_Y, -10.0], [20.0, WALL_HEIGHT, WALL_THICKNESS]) # 下
    add_wall(scene_json["object_instances"], [-10.0, WALL_CENTER_Y, 0.0], [WALL_THICKNESS, WALL_HEIGHT, 20.0]) # 左
    add_wall(scene_json["object_instances"], [10.0, WALL_CENTER_Y, 0.0], [WALL_THICKNESS, WALL_HEIGHT, 20.0])  # 右

    # 内部墙壁 (根据图片布局)
    # 左上部分
    add_wall(scene_json["object_instances"], [-4.5, WALL_CENTER_Y, 8.5], [9.0, WALL_HEIGHT, WALL_THICKNESS])
    add_wall(scene_json["object_instances"], [-8.5, WALL_CENTER_Y, 6.5], [WALL_THICKNESS, WALL_HEIGHT, 5.0])
    add_wall(scene_json["object_instances"], [-6.0, WALL_CENTER_Y, 4.5], [4.0, WALL_HEIGHT, WALL_THICKNESS])
    
    # 中间部分
    add_wall(scene_json["object_instances"], [1.5, WALL_CENTER_Y, 6.5], [WALL_THICKNESS, WALL_HEIGHT, 5.0])
    add_wall(scene_json["object_instances"], [4.0, WALL_CENTER_Y, 4.5], [4.0, WALL_HEIGHT, WALL_THICKNESS])
    add_wall(scene_json["object_instances"], [7.5, WALL_CENTER_Y, 6.0], [5.0, WALL_HEIGHT, WALL_THICKNESS])
    add_wall(scene_json["object_instances"], [-1.5, WALL_CENTER_Y, 2.5], [13.0, WALL_HEIGHT, WALL_THICKNESS])
    add_wall(scene_json["object_instances"], [0.0, WALL_CENTER_Y, 0.0], [4.0, WALL_HEIGHT, WALL_THICKNESS])
    add_wall(scene_json["object_instances"], [6.5, WALL_CENTER_Y, 0.5], [WALL_THICKNESS, WALL_HEIGHT, 3.0])
    
    # 左下部分
    add_wall(scene_json["object_instances"], [-7.5, WALL_CENTER_Y, 0.0], [5.0, WALL_HEIGHT, WALL_THICKNESS])
    add_wall(scene_json["object_instances"], [-4.5, WALL_CENTER_Y, -2.0], [WALL_THICKNESS, WALL_HEIGHT, 4.0])
    add_wall(scene_json["object_instances"], [-7.5, WALL_CENTER_Y, -4.0], [5.0, WALL_HEIGHT, WALL_THICKNESS])
    add_wall(scene_json["object_instances"], [-1.5, WALL_CENTER_Y, -6.0], [9.0, WALL_HEIGHT, WALL_THICKNESS])
    
    # 右下部分
    add_wall(scene_json["object_instances"], [4.5, WALL_CENTER_Y, -4.5], [WALL_THICKNESS, WALL_HEIGHT, 7.0])
    add_wall(scene_json["object_instances"], [7.5, WALL_CENTER_Y, -8.5], [5.0, WALL_HEIGHT, WALL_THICKNESS])
    add_wall(scene_json["object_instances"], [7.5, WALL_CENTER_Y, -2.5], [WALL_THICKNESS, WALL_HEIGHT, 3.0])


    # --- 保存和更新配置 ---
    scene_filename = f"{os.path.basename(scene_name)}_0.scene_instance.json"
    save_path = os.path.join(save_dir, scene_filename)
    with open(save_path, "w") as f:
        json.dump(scene_json, f, indent=4)
    print(f"成功生成目标迷宫场景: {os.path.abspath(save_path)}")

    summary_path = os.path.join(dataset_path, "depthnav_dataset.scene_dataset_config.json")
    with open(summary_path, "r") as f:
        summary = json.load(f)

    scene_config_path = f"configs/{scene_name}"
    if scene_config_path not in summary["scene_instances"]["paths"][".json"]:
        summary["scene_instances"]["paths"][".json"].append(scene_config_path)
        summary["scene_instances"]["paths"][".json"].sort()

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"已更新数据集配置文件: {summary_path}")

    # 启动查看器预览
    print("\n正在启动场景查看器...")
    os.system(f"python depthnav/scripts/scene_viewer.py --scene {scene_config_path}")

if __name__ == "__main__":
    main()