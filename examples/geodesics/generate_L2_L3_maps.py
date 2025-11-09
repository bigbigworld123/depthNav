# examples/navigation/generate_L2_L3_maps.py

import os
from tqdm import trange
from depthnav.envs.scene_generator import (
    SceneGenerator,
    CylinderGenerator,
)
from depthnav.utils.type import Uniform
from depthnav.envs.navigation_env import NavigationEnv

# --- 配置参数 ---
DATASET_PATH = "./datasets/depthnav_dataset"
NUM_TRAIN_SCENES = 10 
GRID_RESOLUTION = 0.2 # 避免内存不足，使用 0.2m 分辨率

def generate_geodesics_for_scene(scene_name_prefix, target_pos=[0.0, 0.0, 1.5]):
    """
    为指定场景路径下的所有地图生成测地线。
    """
    print(f"--- 正在为场景 {scene_name_prefix} 生成测地线 ---")
    
    env = NavigationEnv(
        num_envs=1,
        single_env=True,
        visual=True,
        random_kwargs={
            "target": {"class": "uniform", "mean": target_pos, "half": [0.0, 0.0, 0.0]},
        },
        scene_kwargs={
            "path": f"configs/{scene_name_prefix}", 
            "reload_scenes": True,
            "load_geodesics": False,
            "spawn_obstacles": False,
        },
        sensor_kwargs=[
            {"sensor_type": "depth", "uuid": "depth", "resolution": [72, 128]}
        ],
    )

    num_scenes_to_process = len(env.scene_manager._data_loader)
    print(f"找到 {num_scenes_to_process} 张地图需要处理...")

    for i in trange(num_scenes_to_process):
        env.reset() 
        env.scene_manager.generate_geodesic(
            scene_id=0,
            target=env.target[0], 
            grid_resolution=GRID_RESOLUTION,
        )

    env.close()
    print(f"--- 测地线生成完毕 ---")

# --- L1 修复：为现有地图生成测地线 ---
def generate_level_1_fix_geodesics():
    """
    课程2 (Level 1): 为现有的 configs/level_1 文件夹中的所有地图生成测地线
    """
    SCENE_NAME = "level_1" 
    print(f"--- 正在修复 [Level 1] 现有地图的测地线 ---")
    # 注意：我们这里不需要调用 SceneGenerator，因为地图文件已存在
    generate_geodesics_for_scene(SCENE_NAME)


# --- L2 重新生成：双环 (20m-7m-2.5m) ---
def generate_level_2_dual_ring():
    """
    课程3 (Level 2): 生成双环迷宫 (中环+内环)
    """
    SCENE_NAME = "curriculum/level_2_dual_ring"
    print(f"--- 正在重新生成 [Level 2] 双环场景 ---")
    g = SceneGenerator(
        dataset_path=DATASET_PATH,
        num=NUM_TRAIN_SCENES,
        name=SCENE_NAME,
        stage="stages/box_2",
        obstacle_groups=[
            {
                # 组1：外环墙壁 (20m-7m) - 对应于您新定义的范围
                'keep_in': CylinderGenerator(
                    base_center=[0.0, 0.0, 1.5],
                    radius=20.0, 
                    height=2.0,
                    dataset_path=DATASET_PATH,
                    object_set="primitives/unit/unit_cube",
                    scale_rng=Uniform([0.2, 5.0, 6.0], [0.0, 0.0, 0.0]),
                    rotation_rng=Uniform([0.0, 0.0, 0.0], [0.2, 0.2, 3.14]),
                    density=0.02, 
                    seed=42,
                ),
                'keep_out': [
                    CylinderGenerator(base_center=[0.0, 0.0, 0.0], radius=7.0, height=5.0)
                ]
            },
            {
                # 组2：内环方块 (7m-2.5m)
                'keep_in': CylinderGenerator(
                    base_center=[0.0, 0.0, 0.0],
                    radius=7.0,
                    height=5.0,
                    dataset_path=DATASET_PATH,
                    object_set="primitives/unit/unit_cube",
                    scale_rng=Uniform([1.875], [0.375]),
                    rotation_rng=Uniform([0.0, 0.0, 0.0], [0.5, 0.5, 0.5]),
                    density=0.02,
                    seed=43,
                ),
                'keep_out': [
                    CylinderGenerator(base_center=[0.0, 0.0, 0.0], radius=2.5, height=5.0)
                ]
            }
        ],
    )
    g.generate()
    generate_geodesics_for_scene(SCENE_NAME)


# --- L3 重新生成：三环 (20m-12m-7m-2.5m) ---
def generate_level_3_tri_ring():
    """
    课程4 (Level 3): 生成三环迷宫
    """
    SCENE_NAME = "curriculum/level_3_tri_ring"
    print(f"--- 正在重新生成 [Level 3] 三环场景 ---")
    
    # 注意：为了体现难度递进，L3 我们使用更多的障碍物类型
    g = SceneGenerator(
        dataset_path=DATASET_PATH,
        num=NUM_TRAIN_SCENES,
        name=SCENE_NAME,
        stage="stages/box_2",
        obstacle_groups=[
            {
                # 组1：最外环墙壁 (20m-12m)
                'keep_in': CylinderGenerator(
                    base_center=[0.0, 0.0, 1.5],
                    radius=20.0,
                    height=2.0,
                    dataset_path=DATASET_PATH,
                    object_set="primitives/unit/unit_cube",
                    scale_rng=Uniform([0.2, 5.0, 6.0], [0.0, 0.0, 0.0]),
                    rotation_rng=Uniform([0.0, 0.0, 0.0], [0.2, 0.2, 3.14]),
                    density=0.02,
                    seed=42,
                ),
                'keep_out': [
                    CylinderGenerator(base_center=[0.0, 0.0, 0.0], radius=12.0, height=5.0)
                ]
            },
            {
                # 组2：中环圆柱 (12m-7m)
                'keep_in': CylinderGenerator(
                    base_center=[0.0, 0.0, 0.0],
                    radius=12.0,
                    height=5.0,
                    dataset_path=DATASET_PATH,
                    object_set="primitives/unit/unit_cylinder_vertical",
                    scale_rng=Uniform([0.75, 8.0, 0.75], [0.5, 0.0, 0.5]),
                    rotation_rng=Uniform([0.0, 0.0, 0.0], [0.5, 0.5, 0.5]),
                    density=0.02,
                    seed=44,
                ),
                'keep_out': [
                    CylinderGenerator(base_center=[0.0, 0.0, 0.0], radius=7.0, height=5.0)
                ]
            },
            {
                # 组3：内环方块 (7m-2.5m)
                'keep_in': CylinderGenerator(
                    base_center=[0.0, 0.0, 0.0],
                    radius=7.0,
                    height=5.0,
                    dataset_path=DATASET_PATH,
                    object_set="primitives/unit/unit_cube",
                    scale_rng=Uniform([1.875], [0.375]),
                    rotation_rng=Uniform([0.0, 0.0, 0.0], [0.5, 0.5, 0.5]),
                    density=0.02,
                    seed=43,
                ),
                'keep_out': [
                    CylinderGenerator(base_center=[0.0, 0.0, 0.0], radius=2.5, height=5.0)
                ]
            }
        ],
    )
    g.generate()
    generate_geodesics_for_scene(SCENE_NAME)


if __name__ == "__main__":
    generate_level_1_fix_geodesics() # 修复 L1
    generate_level_2_dual_ring()     # 重新生成 L2 (双环)
    generate_level_3_tri_ring()      # 重新生成 L3 (三环)
    print("\n--- 所有课程地图已准备就绪 ---")