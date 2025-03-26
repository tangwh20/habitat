import os
from typing import TYPE_CHECKING, Union, cast

# transformers needs to be imported before habitat
from transformers import AutoModelForVision2Seq, AutoProcessor

import git
import matplotlib.pyplot as plt
import numpy as np
import torch
import json
from pathlib import Path
from PIL import Image
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R
import quaternion

import habitat
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
    TeleportActionConfig,
)
from habitat.core.agent import Agent
from habitat.utils.visualizations.utils import (
    images_to_video,
    observations_to_image,
    overlay_frame,
)

from utils.data_utils import EvalConfig

# Quiet the Habitat simulator logging
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"


if TYPE_CHECKING:
    from habitat.core.simulator import Observations
    from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim


class OpenVLAContinuousAgent(Agent):
    def __init__(self, processor, vla, device):
        self.processor = processor
        self.vla = vla
        self.device = torch.device(device)
        self.vla.to(self.device)

        self._world_pos = np.array([0., 0., 0.])
        self._world_quat = np.array([0., 0., 0., 1.])

    def act(self, observations):
        # Get the image from the observations
        image = Image.fromarray(observations["rgb"])
        
        # Get the instruction from the observations
        instruction = observations["instruction"]["text"]
        instruction = instruction.lower()
        
        # Get the action from the model        
        inputs = self.processor(instruction, image).to(self.device, dtype=torch.bfloat16)
        waypoints_pred = self.vla.predict_action(**inputs, unnorm_key="sacson", do_sample=False).reshape(8,2)
        print(f"waypoints_pred:{waypoints_pred}")
        teleport_positions = self._get_world_pos(waypoints_pred)
        teleport_rotations = self._get_world_rot(waypoints_pred)

        actions_pred = [            
            {
                "action": "teleport",
                "action_args": {
                    "position": position,
                    "rotation": rotation,
                },
            } for position, rotation in zip(teleport_positions, teleport_rotations)
        ]

        return actions_pred

    def reset(self, agent_state: None):
        if agent_state is None:
            self._world_pos = np.array([0., 0., 0.])
            self._world_quat = np.array([0., 0., 0., 1.])
        else:
            self._world_pos = np.array(agent_state.position)
            self._world_quat = quaternion.as_float_array(agent_state.rotation)

    def _get_world_pos(self, waypoints: np.ndarray):
        assert waypoints.shape == (8, 2)
        positions = np.stack([self._world_pos] * 8)

        yaw = R.from_quat(self._world_quat).as_euler("yxz", degrees=True)[0]
        mat = np.array([
            [np.sin(np.radians(yaw)), np.cos(np.radians(yaw))],
            [np.cos(np.radians(yaw)), -np.sin(np.radians(yaw))]
        ])
        dist = np.matmul(mat, waypoints.T).T
        positions[:, 0] += dist[:, 0]
        positions[:, 2] += dist[:, 1]

        return positions
    
    def _get_world_rot(self, waypoints: np.ndarray):
        assert waypoints.shape == (8, 2)
        euler = R.from_quat(self._world_quat).as_euler("yxz", degrees=True)
        euler = np.stack([euler] * 8)

        heading = np.concatenate([
            waypoints[0, None],
            waypoints[1:] - waypoints[:-1]
        ])
        heading_yaw = np.degrees(np.arctan2(heading[:, 1], heading[:, 0]))

        euler[:, 0] += heading_yaw
        quaternions = R.from_euler("yxz", euler, degrees=True).as_quat()
        quaternions = np.roll(quaternions, -1, axis=1)
        return quaternions

    @property
    def world_state(self):
        return self._world_pos, self._world_quat


def load_dataset(config_path: str, use_habitat_config: bool = True):
    # Create habitat config
    if use_habitat_config:
        config = habitat.get_config(config_path)
    else:
        config = OmegaConf.load(config_path)
    # breakpoint()
    # Add habitat.tasks.nav.nav.TopDownMap and habitat.tasks.nav.nav.Collisions measures
    with habitat.config.read_write(config):
        config.habitat.task.measurements.update(
            {
                "top_down_map": TopDownMapMeasurementConfig(
                    map_padding=3,
                    map_resolution=1024,
                    draw_source=True,
                    draw_border=True,
                    draw_shortest_path=True,
                    draw_view_points=True,
                    draw_goal_positions=True,
                    draw_goal_aabbs=True,
                    fog_of_war=FogOfWarConfig(
                        draw=True,
                        visibility_dist=5.0,
                        fov=90,
                    ),
                ),
                "collisions": CollisionsMeasurementConfig(),
            }
        )
        config.habitat.task.actions.update(
            {
                "teleport": TeleportActionConfig()
            }
        )
        # Use self-generated dataset
        config.habitat.dataset.update(
            {
                "data_path": "/home/tangwenhao/Workspace/habitat/outputs/gpt-4o_vlnce/train.json.gz"
            }
        )
    # Create dataset
    dataset = habitat.make_dataset(
        id_dataset=config.habitat.dataset.type, config=config.habitat.dataset
    )

    return config, dataset


def load_model(cfg: EvalConfig):
    # configs
    device_id = cfg.device
    run_dir = cfg.run_dir

    # load from checkpoints directly
    processor = AutoProcessor.from_pretrained(run_dir, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        run_dir,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device_id)

    # load dataset statistics
    with open(Path(run_dir) / "dataset_statistics.json", "r") as f:
        vla.norm_stats = json.load(f)

    # create openvla agent
    agent = OpenVLAContinuousAgent(processor, vla, device_id)

    return agent


def eval(config_path: str, output_path: str, num_episodes: int = 1):
    # Load the dataset
    config, dataset = load_dataset(config_path, use_habitat_config=True)

    # Load the model
    vla_config = EvalConfig()
    vla_config.run_root_dir = Path("/data/jiyufeng/openvla/lora-instruct-scratch/run")
    vla_config.exp_id = "openvla-7b+sacson+b16+lr-0.0005+lora-r32+dropout-0.0"
    vla_config.output_root_dir = Path(output_path)
    vla_config.device = "cuda:0"
    agent = load_model(vla_config)

    # Create dataset
    dataset = habitat.make_dataset(
        id_dataset=config.habitat.dataset.type, config=config.habitat.dataset
    )
    # Create simulation environment
    with habitat.Env(config=config, dataset=dataset) as env:
        for i in range(num_episodes):            
            # Load the first episode and reset agent
            observations = env.reset()
            agent_state = env.sim.get_agent_state() # get_agent_state returns rotations scalar-first
            agent.reset(agent_state)

            # Get metrics
            info = env.get_metrics()
            # Concatenate RGB-D observation and topdowm map into one image
            instruction = observations.pop("instruction")
            frame = observations_to_image(observations, info)

            # Remove top_down_map from metrics
            info.pop("top_down_map")
            # Overlay numeric metrics onto frame
            frame = overlay_frame(frame, info)
            # Add fame to vis_frames
            vis_frames = [frame]

            # Predict actions and step in the environment
            observations.update({"instruction": instruction})
            actions_pred = agent.act(observations)
            # Step in the environment
            for action in actions_pred:
                observations = env.step(action)
                print(f"Action: {action}")
                print(f"Agent state: {env.sim.get_agent_state()}")
                info = env.get_metrics()
                observations.pop("instruction")
                frame = observations_to_image(observations, info)

                info.pop("top_down_map")
                frame = overlay_frame(frame, info)
                vis_frames.append(frame)

            # Create video from images and save to disk
            import time
            episode_id = env.current_episode.episode_id
            instruction_text = instruction["text"]
            instruction_text = instruction_text.lower().replace(" ", "_")
            video_name = f"{episode_id}_{instruction_text}_{time.strftime('%Y-%m-%d-%H-%M-%S')}"
            # scene_id = os.path.basename(env.current_episode.scene_id).split('.')[0]
            # video_name = f"{episode_id}_{scene_id}_{time.strftime('%Y-%m-%d-%H-%M-%S')}"
            images_to_video(
                vis_frames, output_path, video_name, fps=6, quality=9
            )
            vis_frames.clear()


if __name__ == "__main__":
    repo = git.Repo(".", search_parent_directories=True)
    dir_path = repo.working_tree_dir
    output_path = os.path.join(
        dir_path, "outputs/eval_openvla/continuous"
    )
    os.makedirs(output_path, exist_ok=True)
    os.chdir(dir_path)

    # config_path = "scripts/example/example.yaml"
    config_path = "config/benchmark/nav/vln_r2r.yaml"
    eval(config_path, output_path, num_episodes=10)
    