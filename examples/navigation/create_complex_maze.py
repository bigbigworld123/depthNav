import os
import json
from copy import deepcopy

# 确保我们从一个完全空白的场景开始
empty_scene = {
    "stage_instance": {"template_name": ""},
    "object_instances": [],
    "articulated_object_instances": [],
    "default_lighting": "default",
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
    scene_name = "custom_scenes/perfectly_joined_maze"
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

    # 1. 添加地板 (尺寸 22x22, 顶面在 Y=0)
    add_wall(scene_json["object_instances"],
             center_pos=[0.0, -FLOOR_THICKNESS / 2.0, 0.0],
             size=[22.0, FLOOR_THICKNESS, 22.0])

    # 2. 外围墙壁 (边界在 +/- 10.5, 确保完美贴合)
    BOUNDARY_EDGE = 10.5 # 定义迷宫的内边界
    # 上下墙壁从左边界延伸到右边界 (长度为22)
    add_wall(scene_json["object_instances"], [0.0, WALL_CENTER_Y, BOUNDARY_EDGE + WALL_THICKNESS/2], [BOUNDARY_EDGE*2, WALL_HEIGHT, WALL_THICKNESS])
    add_wall(scene_json["object_instances"], [0.0, WALL_CENTER_Y, -BOUNDARY_EDGE - WALL_THICKNESS/2], [BOUNDARY_EDGE*2, WALL_HEIGHT, WALL_THICKNESS])
    # 左右墙壁夹在上下墙壁之间 (长度为21)
    add_wall(scene_json["object_instances"], [-BOUNDARY_EDGE - WALL_THICKNESS/2, WALL_CENTER_Y, 0.0], [WALL_THICKNESS, WALL_HEIGHT, BOUNDARY_EDGE*2])
    add_wall(scene_json["object_instances"], [BOUNDARY_EDGE + WALL_THICKNESS/2, WALL_CENTER_Y, 0.0], [WALL_THICKNESS, WALL_HEIGHT, BOUNDARY_EDGE*2])

    # 3. 内部墙壁 (精确计算以贴合外墙)
    # 第一堵横墙，从左墙内侧延伸
    # 左墙内侧X坐标 = -10.5。墙体长度 16.5。
    # 它的右端点在 X = -10.5 + 16.5 = 6.0
    # 它的中心点X坐标 = -10.5 + 16.5 / 2 = -2.25
    add_wall(scene_json["object_instances"], [-2.25, WALL_CENTER_Y, 4.0], [16.5, WALL_HEIGHT, WALL_THICKNESS])
    
    # 第二堵横墙，延伸至右墙内侧
    # 右墙内侧X坐标 = 10.5。墙体长度 16.5。
    # 它的左端点在 X = 10.5 - 16.5 = -6.0
    # 它的中心点X坐标 = 10.5 - 16.5 / 2 = 2.25
    add_wall(scene_json["object_instances"], [2.25, WALL_CENTER_Y, -4.0], [16.5, WALL_HEIGHT, WALL_THICKNESS])

    # --- 保存和更新配置 ---
    scene_filename = f"{os.path.basename(scene_name)}_0.scene_instance.json"
    save_path = os.path.join(save_dir, scene_filename)
    with open(save_path, "w") as f:
        json.dump(scene_json, f, indent=4)
    print(f"成功生成完美贴合的迷宫: {os.path.abspath(save_path)}")

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

    print("\n正在启动场景查看器...")
    os.system(f"python depthnav/scripts/scene_viewer.py --scene {scene_config_path}")

if __name__ == "__main__":
    main()