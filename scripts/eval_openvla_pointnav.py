import os
from typing import TYPE_CHECKING, Union, cast

# transformers needs to be imported before habitat
from transformers import AutoModelForVision2Seq, AutoProcessor

import git
import matplotlib.pyplot as plt
import numpy as np
import torch
import shutil
import json
from pathlib import Path
from PIL import Image

import habitat
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.core.agent import Agent
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import (
    images_to_video,
    observations_to_image,
    overlay_frame,
)
from habitat_sim.utils import viz_utils as vut

from utils.data_utils import EvalConfig
from utils.action_transform import waypoints_to_actions

# Quiet the Habitat simulator logging
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"


if TYPE_CHECKING:
    from habitat.core.simulator import Observations
    from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim

STEP_LENGTH = 0.25
TURN_ANGLE = np.pi / 12.


class OpenVLAHabitatAgent(Agent):
    def __init__(self, processor, vla, device):
        self.processor = processor
        self.vla = vla
        self.device = torch.device(device)
        self.vla.to(self.device)

        self.position = np.array([0., 0.])
        self.yaw = 0.

    def act(self, observations):
        # Get the image from the observations
        image = Image.fromarray(observations["rgb"])

        # Get the instruction from the observations
        point_goal = observations["pointgoal_with_gps_compass"]
        relative_goal = point_goal - self.position
        distance = np.linalg.norm(relative_goal)
        yaw_goal = np.arctan2(relative_goal[1], relative_goal[0])
        instruction = f"Go to relative position {distance} meters, \
                        relative yaw {yaw_goal - self.yaw} radians."
        instruction = instruction.lower()

        # Get the action from the model        
        inputs = self.processor(instruction, image).to(self.device, dtype=torch.bfloat16)
        waypoints_pred = self.vla.predict_action(**inputs, unnorm_key="sacson", do_sample=False).reshape(8,2)
        self.position, self.yaw, actions_pred = waypoints_to_actions(waypoints_pred)
        print(f"position:{self.position}, goal:{point_goal}, distance:{distance}")
        print(f"yaw:{self.yaw}, yaw_goal:{yaw_goal}, yaw_diff:{yaw_goal - self.yaw}")
        print("actions_pred:", actions_pred)

        return actions_pred
    
    def reset(self) -> None:
        pass

        

repo = git.Repo(".", search_parent_directories=True)
dir_path = repo.working_tree_dir
data_path = os.path.join(dir_path, "data")
output_path = os.path.join(
    dir_path, "outputs/eval_openvla/habitat"
)
os.makedirs(output_path, exist_ok=True)
os.chdir(dir_path)


def load_dataset(config_path: str):
    # Create habitat config
    config = habitat.get_config(
        config_path=os.path.join(dir_path, config_path)
    )
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
    agent = OpenVLAHabitatAgent(processor, vla, device_id)

    return agent


def eval(max_steps: int = 100):
    # Load the dataset
    config, dataset = load_dataset(
        "config/benchmark/nav/pointnav/pointnav_habitat_test.yaml"
    )

    # Load the model
    eval_config = EvalConfig()
    eval_config.run_root_dir = Path("/data/jiyufeng/openvla/lora-instruct-scratch/run")
    eval_config.exp_id = "openvla-7b+sacson+b16+lr-0.0005+lora-r32+dropout-0.0"
    eval_config.data_root_dir = Path(data_path)
    eval_config.output_root_dir = Path(output_path)
    agent = load_model(eval_config)

    # Create dataset
    dataset = habitat.make_dataset(
        id_dataset=config.habitat.dataset.type, config=config.habitat.dataset
    )
    # Create simulation environment
    with habitat.Env(config=config, dataset=dataset) as env:
        # Load the first episode and reset agent
        observations = env.reset()
        agent.reset()

        # Get metrics
        info = env.get_metrics()
        # Concatenate RGB-D observation and topdowm map into one image
        frame = observations_to_image(observations, info)

        # Remove top_down_map from metrics
        info.pop("top_down_map")
        # Overlay numeric metrics onto frame
        frame = overlay_frame(frame, info)
        # Add fame to vis_frames
        vis_frames = [frame]

        # Repeat the steps above while agent doesn't reach the goal
        step_count = 0
        while not env.episode_over and step_count < max_steps:
            # Get the next best actions
            actions_pred = agent.act(observations)

            for action in actions_pred:
                # Step in the environment
                if env.episode_over:
                    break
                step_count += 1
                observations = env.step(action)
                info = env.get_metrics()
                frame = observations_to_image(observations, info)

                info.pop("top_down_map")
                frame = overlay_frame(frame, info)
                vis_frames.append(frame)

        current_episode = env.current_episode
        # video_name = f"{os.path.basename(current_episode.scene_id)}_{current_episode.episode_id}"
        import time
        video_name = f"test-{time.strftime('%Y-%m-%d-%H-%M-%S')}"
        # Create video from images and save to disk
        images_to_video(
            vis_frames, output_path, video_name, fps=6, quality=9
        )
        vis_frames.clear()


if __name__ == "__main__":
    eval(max_steps=500)
    