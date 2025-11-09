import os
from pathlib import Path

from depthnav.envs.scene_generator import (
    SceneGenerator,
    CylinderGenerator,
)
from depthnav.utils.type import Uniform

# --- 配置参数 ---
DATASET_PATH = "./datasets/depthnav_dataset"
SCENE_NAME = "level_1/ring_layered_large"
NUM_SCENES = 1

def generate_layered_ring_maze():
    """
    生成一个分层环形障碍物场景：
    - 外环 (5m-10m): 墙壁
    - 内环 (2.5m-5m): 方块
    - 中心区域 (0m-2.5m): 空旷
    """

    print(f"--- 开始生成分层环形障碍物场景 ---")
    print(f"场景路径: {DATASET_PATH}/{SCENE_NAME}")

    # 使用修改后的 SceneGenerator 和新的 obstacle_groups 结构
    g = SceneGenerator(
        dataset_path=DATASET_PATH,
        num=NUM_SCENES,
        name=SCENE_NAME,
        stage="stages/box_2",
        # 使用 obstacle_groups 代替 keep_in_bounds 和 keep_out_bounds
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
    
    # --- 生成场景文件 ---
    g.generate()
    print(f"\n--- 成功生成场景文件 ---")

    # --- 预览生成的场景 ---
    # 确保使用正确的场景名称格式
    base_scene_name = os.path.basename(SCENE_NAME)
    scene_path = f"{SCENE_NAME}/{base_scene_name}_0" 
    print(f"\n--- 正在启动预览器以显示场景: {scene_path} ---")
    
    # 获取数据集配置文件的完整路径
    summary_path = str(list(Path(DATASET_PATH).glob("*.scene_dataset_config.json"))[0])

    # 调用 scene_viewer.py 脚本进行预览
    os.system(
        f"python depthnav/scripts/scene_viewer.py --dataset \"{summary_path}\" --scene \"{scene_path}\""
    )


if __name__ == "__main__":
    generate_layered_ring_maze()