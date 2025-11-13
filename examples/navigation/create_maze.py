# generator_intuitive_maze.py (使用直观的辅助函数，易于修改)
import os
import json
from copy import deepcopy

# 关键：保持与S型迷宫完全相同的空舞台设置
empty_scene = {
    "stage_instance": {"template_name": ""},
    "object_instances": [],
    "articulated_object_instances": [],
    "default_lighting": "lighting/garage_v1_0",
    "user_defined": {},
}

# --------------------------------------------------------------------------
# !! 新增的、易于理解的辅助函数 !!
# --------------------------------------------------------------------------

WALL_HEIGHT = 5.0      # 您可以在这里统一修改墙壁的高度
WALL_THICKNESS = 0.2   # 您可以在这里统一修改墙壁的厚度
FLOOR_THICKNESS = 0.1
WALL_CENTER_Y = (FLOOR_THICKNESS / 2.0) + (WALL_HEIGHT / 2.0)

def _add_primitive_wall(scene_objects, center_pos, size):
    """底层函数，用于实际创建方块 (您无需修改这里)"""
    wall = {
        "template_name": "primitives/unit/unit_cube",
        "translation": [float(c) for c in center_pos],
        "rotation": [1.0, 0.0, 0.0, 0.0],
        "non_uniform_scale": [float(s) for s in size],
        "motion_type": "STATIC",
        "translation_origin": "COM",
    }
    scene_objects.append(wall)

def add_horizontal_wall(scene_objects, z_pos, x_start, x_end):
    """
    画一条横墙 (沿X轴)。
    z_pos: 墙壁的前后位置。
    x_start: 墙壁的左端点X坐标。
    x_end: 墙壁的右端点X坐标。
    """
    width = abs(x_end - x_start)
    center_x = (x_start + x_end) / 2.0
    _add_primitive_wall(scene_objects, 
                        center_pos=[center_x, WALL_CENTER_Y, z_pos], 
                        size=[width, WALL_HEIGHT, WALL_THICKNESS])

def add_vertical_wall(scene_objects, x_pos, z_start, z_end):
    """
    画一条竖墙 (沿Z轴)。
    x_pos: 墙壁的左右位置。
    z_start: 墙壁的后端点Z坐标。
    z_end: 墙壁的前端点Z坐标。
    """
    depth = abs(z_end - z_start)
    center_z = (z_start + z_end) / 2.0
    _add_primitive_wall(scene_objects,
                        center_pos=[x_pos, WALL_CENTER_Y, center_z],
                        size=[WALL_THICKNESS, WALL_HEIGHT, depth])

# --------------------------------------------------------------------------

def main():
    scene_name = "custom_scenes/intuitive_image_maze"
    dataset_path = "./datasets/depthnav_dataset"
    save_dir = os.path.join(dataset_path, "configs", scene_name)

    if os.path.exists(save_dir):
        import shutil
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    scene_json = deepcopy(empty_scene)
    
    BOUNDARY_EDGE = 10.5

    # 1. 添加地板 (尺寸 22x22)
    _add_primitive_wall(scene_json["object_instances"],
                      center_pos=[0.0, -FLOOR_THICKNESS / 2.0, 0.0],
                      size=[22.0, FLOOR_THICKNESS, 22.0])

    # 2. 外围墙壁 (边界在 +/- 10.5)
    add_horizontal_wall(scene_json["object_instances"], z_pos=BOUNDARY_EDGE, x_start=-BOUNDARY_EDGE, x_end=BOUNDARY_EDGE) # 上
    add_horizontal_wall(scene_json["object_instances"], z_pos=-BOUNDARY_EDGE, x_start=-BOUNDARY_EDGE, x_end=BOUNDARY_EDGE)# 下
    add_vertical_wall(scene_json["object_instances"], x_pos=BOUNDARY_EDGE, z_start=-BOUNDARY_EDGE, z_end=BOUNDARY_EDGE)  # 右
    
    # 左侧墙壁留出入口 (对应图片)

    # 添加完整的左侧墙壁
    add_vertical_wall(scene_json["object_instances"], x_pos=-BOUNDARY_EDGE, z_start=-BOUNDARY_EDGE, z_end=BOUNDARY_EDGE)
    
    # ==========================================================
    # !! 3. 内部墙壁布局 (现在您可以轻松修改这里了) !!
    # !! 坐标范围: X和Z都在 -10.5 到 +10.5 之间 !!
    # ==========================================================
    print("正在搭建图片迷宫内部结构...")
    # 从上到下，从左到右添加墙壁
    add_horizontal_wall(scene_objects=scene_json["object_instances"], z_pos=5.0, x_start=-10.5, x_end=5.0)
    # add_vertical_wall(scene_objects=scene_json["object_instances"], x_pos=0.0, z_start=-5.0, z_end=10.5)
    add_horizontal_wall(scene_objects=scene_json["object_instances"], z_pos=0.0, x_start=-10.5, x_end=-2.0)
    add_vertical_wall(scene_objects=scene_json["object_instances"], x_pos=-5.0, z_start=-5.0, z_end=0.0)
    add_horizontal_wall(scene_objects=scene_json["object_instances"], z_pos=-5.0, x_start=-2.0, x_end=5.0)
    # add_vertical_wall(scene_objects=scene_json["object_instances"], x_pos=5.0, z_start=-10.5, z_end=0.0)

    # 添加缺失的通道
    add_horizontal_wall(scene_objects=scene_json["object_instances"], z_pos=0.0, x_start=5.0, x_end=10.5)

    # --- 保存和更新配置 (无需修改) ---
    scene_filename = f"{os.path.basename(scene_name)}_0.scene_instance.json"
    save_path = os.path.join(save_dir, scene_filename)
    with open(save_path, "w") as f:
        json.dump(scene_json, f, indent=4)
    print(f"成功生成直观版图片迷宫: {os.path.abspath(save_path)}")
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