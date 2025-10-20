#!/usr/bin/env python3
"""
Minimal habitat_sim scene viewer
(MODIFIED to support global top-down view with 'G', print coordinates with 'P', and cycle scenes with 'TAB')
"""

import argparse
import ctypes
import math
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from depthnav.common import habitat_to_std
import quaternion  # This imports the numpy-quaternion library

flags = sys.getdlopenflags()
sys.setdlopenflags(flags | ctypes.RTLD_GLOBAL)

import magnum as mn
import numpy as np
from magnum.platform.glfw import Application

import habitat_sim
from habitat_sim.agent import AgentState
from habitat_sim.logging import LoggingContext, logger
from habitat_sim.utils.settings import default_sim_settings, make_cfg


class HabitatSimInteractiveViewer(Application):
    def __init__(self, sim_settings: Dict[str, Any]) -> None:
        self.sim_settings: Dict[str:Any] = sim_settings
        window_size: mn.Vector2 = (
            self.sim_settings["window_width"],
            self.sim_settings["window_height"],
        )
        configuration = self.Configuration()
        configuration.title = "Habitat Sim Interactive Viewer"
        configuration.size = window_size
        Application.__init__(self, configuration)
        self.fps: float = 60.0
        self.sim_settings["width"] = self.sim_settings["window_width"]
        self.sim_settings["height"] = self.sim_settings["window_height"]

        key = Application.KeyEvent.Key
        self.pressed = {
            key.UP: False, key.DOWN: False, key.LEFT: False, key.RIGHT: False,
            key.A: False, key.D: False, key.S: False, key.W: False,
            key.X: False, key.Z: False,
        }
        self.key_to_action = {
            key.UP: "look_up", key.DOWN: "look_down", key.LEFT: "turn_left",
            key.RIGHT: "turn_right", key.A: "move_left", key.D: "move_right",
            key.S: "move_backward", key.W: "move_forward", key.X: "move_down",
            key.Z: "move_up",
        }
        self.cfg: Optional[habitat_sim.simulator.Configuration] = None
        self.sim: Optional[habitat_sim.simulator.Simulator] = None
        self.reconfigure_sim()
        self.time_since_last_simulation = 0.0
        LoggingContext.reinitialize_from_env()
        logger.setLevel("INFO")
        self.print_help_text()

    def draw_event(
        self,
        active_agent_id_and_sensor_name: Tuple[int, str] = (0, "color_sensor"),
    ) -> None:
        agent_acts_per_sec = self.fps
        mn.gl.default_framebuffer.clear(
            mn.gl.FramebufferClear.COLOR | mn.gl.FramebufferClear.DEPTH
        )
        self.time_since_last_simulation += Timer.prev_frame_duration
        num_agent_actions: int = self.time_since_last_simulation * agent_acts_per_sec
        self.move_and_look(int(num_agent_actions))
        if self.time_since_last_simulation >= 1.0 / self.fps:
            self.time_since_last_simulation = math.fmod(
                self.time_since_last_simulation, 1.0 / self.fps
            )
        keys = active_agent_id_and_sensor_name
        self.sim._Simulator__sensors[keys[0]][keys[1]].draw_observation()
        agent = self.sim.get_agent(keys[0])
        self.render_camera = agent.scene_node.node_sensor_suite.get(keys[1])
        self.render_camera.render_target.blit_rgba_to_default()
        mn.gl.default_framebuffer.bind()
        self.swap_buffers()
        Timer.next_frame()
        self.redraw()

    def default_agent_config(self) -> habitat_sim.agent.AgentConfiguration:
        make_action_spec = habitat_sim.agent.ActionSpec
        make_actuation_spec = habitat_sim.agent.ActuationSpec
        MOVE, LOOK = 0.07, 1.5
        action_list = [
            "move_left", "turn_left", "move_right", "turn_right", "move_backward",
            "look_up", "move_forward", "look_down", "move_down", "move_up",
        ]
        action_space: Dict[str, habitat_sim.agent.ActionSpec] = {}
        for action in action_list:
            actuation_spec_amt = MOVE if "move" in action else LOOK
            action_spec = make_action_spec(action, make_actuation_spec(actuation_spec_amt))
            action_space[action] = action_spec
        sensor_spec: List[habitat_sim.sensor.SensorSpec] = self.cfg.agents[self.agent_id].sensor_specifications
        agent_config = habitat_sim.agent.AgentConfiguration(
            height=0.1, radius=0.1, sensor_specifications=sensor_spec,
            action_space=action_space, body_type="cylinder",
        )
        return agent_config

    def reconfigure_sim(self) -> None:
        self.cfg = make_cfg(self.sim_settings)
        self.agent_id: int = self.sim_settings["default_agent"]
        self.cfg.agents[self.agent_id] = self.default_agent_config()
        if self.sim is None:
            self.sim = habitat_sim.Simulator(self.cfg)
        else:
            if self.sim.config.sim_cfg.scene_id == self.cfg.sim_cfg.scene_id:
                self.sim.config.sim_cfg.scene_id = "NONE"
            self.sim.reconfigure(self.cfg)
        self.default_agent = self.sim.get_agent(self.agent_id)
        self.render_camera = self.default_agent.scene_node.node_sensor_suite.get("color_sensor")
        self.sim_settings["scene"] = self.sim.curr_scene_name
        Timer.start()

    def move_and_look(self, repetitions: int) -> None:
        if repetitions == 0: return
        agent = self.sim.agents[self.agent_id]
        action_queue: List[str] = [self.key_to_action[k] for k, v in self.pressed.items() if v]
        for _ in range(int(repetitions)):
            [agent.act(x) for x in action_queue]

    def key_press_event(self, event: Application.KeyEvent) -> None:
        key = event.key
        pressed = Application.KeyEvent.Key
        mod = Application.InputEvent.Modifier
        shift_pressed = bool(event.modifiers & mod.SHIFT)

        if key == pressed.ESC:
            self.exit_event(Application.ExitEvent)
            return
        elif key == pressed.H:
            self.print_help_text()
            
        # --- 功能 1: 切换全局视角 ---
        elif key == pressed.G:
            print("Switching to Global Top-Down View...")
            agent = self.sim.get_agent(self.agent_id)
            scene_bb = self.sim.get_active_scene_graph().get_root_node().cumulative_bb
            center = scene_bb.center()
            top_down_position = mn.Vector3(center.x, center.y + 20.0, center.z)
            angle = -math.pi / 2.0
            w = math.cos(angle / 2.0)
            x = math.sin(angle / 2.0)
            look_down_rotation = quaternion.quaternion(w, x, 0, 0)
            agent_state = AgentState(position=top_down_position, rotation=look_down_rotation)
            agent.set_state(agent_state)
            
        # --- 功能 2: 打印坐标 ---
        elif key == pressed.P:
            agent_state = self.sim.get_agent(self.agent_id).get_state()
            std_pos, _ = habitat_to_std(habitat_pos=agent_state.position)
            print(f"\n{'='*50}\nCurrent Standard Coordinates (X, Y, Z):\n[{std_pos[0][0]:.2f}, {std_pos[0][1]:.2f}, {std_pos[0][2]:.2f}]\n{'='*50}\n")
            
        # ==========================================================
        # !! 核心修复：将 TAB 键的功能重新添加回来 !!
        # ==========================================================
        elif key == pressed.TAB:
            inc = -1 if shift_pressed else 1
            scene_ids = self.sim.metadata_mediator.get_scene_handles()
            cur_scene_index = 0
            if self.sim_settings["scene"] not in scene_ids:
                matching_scenes = [
                    (ix, x)
                    for ix, x in enumerate(scene_ids)
                    if self.sim_settings["scene"] in x
                ]
                if not matching_scenes:
                    logger.warning(
                        f"The current scene, '{self.sim_settings['scene']}', is not in the list, starting cycle at index 0."
                    )
                else:
                    cur_scene_index = matching_scenes[0][0]
            else:
                cur_scene_index = scene_ids.index(self.sim_settings["scene"])
            next_scene_index = min(max(cur_scene_index + inc, 0), len(scene_ids) - 1)
            self.sim_settings["scene"] = scene_ids[next_scene_index]
            self.reconfigure_sim()
            logger.info(f"Reconfigured simulator for scene: {self.sim_settings['scene']}")
            
        elif key == pressed.R:
            self.reconfigure_sim()

        if key in self.pressed:
            self.pressed[key] = True
        event.accepted = True
        self.redraw()

    def key_release_event(self, event: Application.KeyEvent) -> None:
        if event.key in self.pressed: self.pressed[event.key] = False
        event.accepted = True; self.redraw()
        
    def mouse_move_event(self, event: Application.MouseMoveEvent) -> None:
        if event.buttons & Application.MouseMoveEvent.Buttons.LEFT:
            delta = self.get_mouse_position(event.relative_position) / 2
            self.default_agent.act("turn_right", delta.x)
            self.default_agent.act("look_up", delta.y)
        event.accepted = True; self.redraw()
        
    def get_mouse_position(self, p: mn.Vector2i) -> mn.Vector2i:
        return p * (mn.Vector2i(self.framebuffer_size) / mn.Vector2i(self.window_size))
        
    def exit_event(self, event: Application.ExitEvent):
        self.sim.close(destroy=True); event.accepted = True; exit(0)
        
    def print_help_text(self) -> None:
        logger.info(
"""
=====================================================
Welcome to the Habitat-sim Interactive Viewer!
=====================================================
Mouse:
-------
    LEFT CLICK + DRAG: Turn agent and look up/down.

Key Commands:
-------------
    ESC:        Exit the application.
    'h':        Display this help message.
    'g':        Switch to Global Top-Down View.
    'p':        Print current agent coordinates to the terminal.
    'r':        Reset the simulator.
    TAB:        Cycle through scenes.

Agent Controls:
---------------
    'wasd':     Move forward/backward and left/right.
    'zx':       Move up/down.
    ARROW KEYS: Turn left/right and look up/down.
=====================================================
"""
        )

class Timer:
    start_time = 0.0; prev_frame_time = 0.0; prev_frame_duration = 0.0; running = False
    @staticmethod
    def start(): Timer.running = True; Timer.start_time = time.time(); Timer.prev_frame_time = Timer.start_time
    @staticmethod
    def next_frame():
        if not Timer.running: return
        Timer.prev_frame_duration = time.time() - Timer.prev_frame_time
        Timer.prev_frame_time = time.time()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene",
        default="garage_empty_0",
        type=str,
        help='scene/stage file to load (default: "garage_empty_0")',
    )
    parser.add_argument(
        "--dataset",
        default="./datasets/depthnav_dataset/depthnav_dataset.scene_dataset_config.json",
        type=str,
        metavar="DATASET",
        help='dataset configuration file to use (default: "./datasets/depthnav_dataset/depthnav_dataset.scene_dataset_config.json")',
    )
    parser.add_argument(
        "--width",
        default=800,
        type=int,
        help="Horizontal resolution of the window.",
    )
    parser.add_argument(
        "--height",
        default=600,
        type=int,
        help="Vertical resolution of the window.",
    )
    args = parser.parse_args()
    if args.width < 1:
        parser.error("width must be a positive non-zero integer.")
    if args.height < 1:
        parser.error("height must be a positive non-zero integer.")
    return args

if __name__ == "__main__":
    args = parse_args()

    sim_settings: Dict[str, Any] = default_sim_settings
    sim_settings["scene"] = args.scene
    sim_settings["scene_dataset_config_file"] = args.dataset
    sim_settings["window_width"] = args.width
    sim_settings["window_height"] = args.height
    sim_settings["default_agent_navmesh"] = False

    HabitatSimInteractiveViewer(sim_settings).exec()