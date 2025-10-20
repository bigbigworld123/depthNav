# debug_maze.py (v3 - Updated for new map scale)
import yaml
import torch as th
import numpy as np
import os
from depthnav.envs.env_aliases import env_aliases

def debug():
    # --- 1. 加载你的评估配置文件 ---
    cfg_file = "examples/navigation/eval_cfg/eval_array_maze.yaml" 
    if not os.path.exists(cfg_file):
        print(f"错误：找不到配置文件 '{cfg_file}'。")
        print(f"当前工作目录是: {os.getcwd()}")
        return
        
    print(f"--- 正在加载配置文件: {cfg_file} ---")
    with open(cfg_file, "r") as file:
        config = yaml.safe_load(file)
    env_config = config["env"]
    env_class = env_aliases[config["env_class"]]

    env = None
    try:
        # --- 2. 初始化环境实例 ---
        print("\n--- 步骤 1: 初始化 NavigationEnv 实例 ---")
        env = env_class(requires_grad=False, **env_config)
        print("--- 成功: 环境实例已创建。 ---")

        # --- 3. 完整重置环境 (加载场景) ---
        print("\n--- 步骤 2: 调用 env.reset() 来加载场景和初始化智能体 ---")
        env.reset()
        print("--- 成功: env.reset() 执行完毕，场景已加载，没有崩溃。 ---")

        # --- 4. 测试碰撞检测 (使用缩小后的新坐标) ---
        print("\n--- 步骤 3: 在已加载的场景中测试碰撞检测 ---")
        scene_manager = env.scene_manager
        
        # !! 使用与缩小后地图匹配的新坐标 !!
        start_pos_std = th.tensor([[-12.0, 0.5, -6.0]])
        target_pos_std = th.tensor([[12.0, 0.5, 6.0]])
        wall_pos_std = th.tensor([[-14.0, 0.5, -8.0]]) # 对应迷宫 (0,0) 的墙体

        print(f"正在检查起始点 (路径上): {start_pos_std.numpy()}")
        is_collision_start = scene_manager.get_point_is_collision(
            std_positions=start_pos_std, scene_id=0, uav_radius=0.1
        )
        print(f"--> 起始点是否碰撞? {is_collision_start.item()}")

        print(f"正在检查目标点 (路径上): {target_pos_std.numpy()}")
        is_collision_target = scene_manager.get_point_is_collision(
            std_positions=target_pos_std, scene_id=0, uav_radius=0.1
        )
        print(f"--> 目标点是否碰撞? {is_collision_target.item()}")
        
        print(f"正在检查墙内点: {wall_pos_std.numpy()}")
        is_collision_wall = scene_manager.get_point_is_collision(
            std_positions=wall_pos_std, scene_id=0, uav_radius=0.1
        )
        print(f"--> 墙内点是否碰撞? {is_collision_wall.item()}")

        if not is_collision_start.item() and not is_collision_target.item() and is_collision_wall.item():
             print("--- 成功: 碰撞检查行为符合预期！问题已解决。 ---")
        else:
             print("--- 失败: 碰撞检查仍然返回了意外的结果。 ---")
             print("预期结果: 起始点=False, 目标点=False, 墙内点=True")

    except Exception as e:
        print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("--- 失败: 脚本在执行过程中捕获到异常。 ---")
        import traceback
        traceback.print_exc()
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        
    finally:
        if env:
            env.close()
            print("\n环境已关闭。")
            
    print("\n--- 调试脚本执行完毕。 ---")

if __name__ == "__main__":
    debug()