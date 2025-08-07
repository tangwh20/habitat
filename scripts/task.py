import os
import git
import json
import gzip
import numpy as np
from PIL import Image
from tqdm import tqdm, trange

import habitat
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.core.agent import Agent
from habitat.utils.visualizations.utils import (
    images_to_video,
    observations_to_image,
    overlay_frame,
)

from agents import FixedAgent

# Quiet the Habitat simulator logging
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

CONFIG_MAP = {
    "vlnce": "config/benchmark/nav/vln_r2r.yaml",
    "objectnav": "config/benchmark/nav/objectnav/objectnav_mp3d.yaml",
}

class Task:
    """
    A class to handle task-related operations, such as loading data and rolling out actions.
    """
    def __init__(
            self, 
            task_name: str, 
            agent: FixedAgent = None, 
            scene_name: str = None,
            output_path: str = None,
            save_video: bool = False,
            remove_existing_output: bool = False
        ):
        assert task_name in ["vlnce", "objectnav"], "Invalid task name"
        self.task_name = task_name
        self.agent = agent
        self.scene_name = scene_name
        self.output_path = output_path
        self.save_video = save_video
        self.remove_existing_output = remove_existing_output

        self.repo = git.Repo(".", search_parent_directories=True)
        self.dir_path = self.repo.working_tree_dir
        if self.output_path is None:
            self.output_path = os.path.join(self.dir_path, "outputs", task_name)
        if self.remove_existing_output and os.path.exists(self.output_path):
            import shutil
            shutil.rmtree(self.output_path)
        os.makedirs(self.output_path, exist_ok=True)
        os.chdir(self.dir_path)

        self.config, self.dataset = self._load_dataset()

        if self.agent is None:
            if self.task_name == "vlnce":
                dataset_path = self.config.habitat.dataset.data_path
                dataset_path = dataset_path.replace("{split}", self.config.habitat.dataset.split)
                gt_path = dataset_path.replace(".json.gz", "_gt.json.gz")
            elif self.task_name == "objectnav":
                gt_path = None
            self.agent = FixedAgent(self.task_name, data_path=gt_path)

        # breakpoint()

        self.env = habitat.Env(config=self.config, dataset=self.dataset)


    def _load_dataset(self):
        # Create habitat config
        config = habitat.get_config(os.path.join(self.dir_path, CONFIG_MAP[self.task_name]))

        # Update config with custom settings
        with habitat.config.read_write(config):
            # breakpoint()
            config.habitat.update({"seed": 200})
            config.habitat.environment.iterator_options.update({"group_by_scene": False})
            config.habitat.simulator.habitat_sim_v0.update({"allow_sliding": False})
            config.habitat.simulator.agents.main_agent.update({"radius": 0.1}) # Agent collision radius (default is 0.1)
            config.habitat.task.measurements.update({"collisions": CollisionsMeasurementConfig()})

            if self.save_video:
                config.habitat.task.measurements.update({
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
                    )
                })
            if self.task_name == "objectnav" and self.scene_name is not None:
                config.habitat.dataset.update({
                    "data_path": "data/datasets/objectnav/mp3d/v1/{split}/content/" + self.scene_name + ".json.gz",
                }) # Change this if needed

        # Create dataset
        dataset = habitat.make_dataset(
            id_dataset=config.habitat.dataset.type, config=config.habitat.dataset
        )

        return config, dataset
    
    def rollout(self):
        # Load the first episode and reset agent
        observations = self.env.reset()
        episode_id = self.env.current_episode.episode_id
        scene_id = os.path.basename(self.env.current_episode.scene_id).split('.')[0]
        self.agent.reset(self.env.current_episode)

        if os.path.exists(os.path.join(self.output_path, f"{scene_id}_{episode_id}")):
            print(f"Episode {episode_id} already exists")
            return 
        # print(f"Episode {episode_id} started")

        # Get metrics
        info = self.env.get_metrics()
        # Get text information from observations
        if self.task_name == "vlnce":
            instruction = observations.pop("instruction")
            instruction_text = instruction["text"]
        elif self.task_name == "objectnav":
            object_goal_id = observations.pop("objectgoal").item()
            object_goal_text = self.env.current_episode.object_category

        if self.save_video:
            # Concatenate RGB-D observation and topdown map into one image
            frame = observations_to_image(observations, info)
            # Remove top_down_map from metrics
            info.pop("top_down_map")
            # Overlay numeric metrics onto frame
            frame = overlay_frame(frame, info)
            # Add fame to vis_frames
            vis_frames = [frame]

        current_step = 0
        collisions = []
        distances = []
        # Repeat the steps above while agent doesn't reach the goal
        while not self.env.episode_over:
            # Get the next best action
            # breakpoint()
            self.agent.update_state(self.env.sim.get_agent_state())
            action = self.agent.act(observations, current_step)
            if action is None:
                break

            # Step in the environment
            observations = self.env.step(action)
            if self.task_name == "vlnce":
                observations.pop("instruction")
            elif self.task_name == "objectnav":
                observations.pop("objectgoal")

            info = self.env.get_metrics()
            collisions.append(info["collisions"]["is_collision"])
            distances.append(info["distance_to_goal"])

            if self.save_video:
                frame = observations_to_image(observations, info)
                info.pop("top_down_map")
                frame = overlay_frame(frame, info)
                vis_frames.append(frame)

            current_step += 1

        # Save image observations on each reference step
        episode_output_path = os.path.join(self.output_path, scene_id, episode_id)
        image_output_path = os.path.join(episode_output_path, "images")
        os.makedirs(image_output_path, exist_ok=True)
        for step, view in enumerate(self.agent.views):
            image_path = os.path.join(image_output_path, f"{step}.png")
            Image.fromarray(view).save(image_path)
        self.agent.views.clear()

        # Save actions and positions as json
        with open(os.path.join(episode_output_path, "data.json"), "w") as f:
            if self.task_name == "vlnce":
                json.dump({
                    "instruction": instruction_text,
                    "actions": self.agent.actions.tolist(),
                    "positions": self.agent.positions,
                    "rotations": self.agent.rotations,
                    "distances": distances,
                    "collisions": collisions,
                }, f)
            elif self.task_name == "objectnav":
                json.dump({
                    "object_goal": object_goal_text,
                    "object_goal_id": object_goal_id,
                    "actions": self.agent.actions.tolist(),
                    "positions": self.agent.positions,
                    "rotations": self.agent.rotations,
                    "distances": distances,
                    "collisions": collisions,
                }, f)

        # Save video
        if self.save_video:
            images_to_video(vis_frames, episode_output_path, "video", fps=6, quality=9)
            vis_frames.clear()
    


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Habitat task rollout")
    parser.add_argument("--task_name", type=str, choices=["vlnce", "objectnav"], required=True, help="Task name")
    parser.add_argument("--scene_name", type=str, default=None, help="Scene name for objectnav task. If not provided, will process all scenes in the dataset.")
    parser.add_argument("--output_path", type=str, default=None, help="Output path for results")
    parser.add_argument("--save_video", action="store_true", help="Save video of the rollout")
    parser.add_argument("--remove_existing_output", action="store_true", help="Remove existing output directory")
    args = parser.parse_args()

    # Create error log file
    import time
    log_filepath = f"logs/objectnav/parallel_0806/{args.scene_name}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log"
    os.makedirs(os.path.dirname(log_filepath), exist_ok=True)
    log_f = open(log_filepath, "a")
    log_f.write(f"Start processing scene {args.scene_name} at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_f.flush()

    task = Task(**vars(args))

    num_episodes = task.dataset.num_episodes
    print(f"Number of episodes in the dataset: {num_episodes}") # 10,819 for vlnce, 2,632,422 for objectnav
    for i in trange(num_episodes, desc=f"Processing Scene {args.scene_name}:"):
        try:
            task.rollout()
        except Exception as e:
            print(f"Error in episode {i}: {e}")
            log_f.write(f"Error in episode {i}: {e}\n")
            log_f.flush()
            continue
        log_f.write(f"Finished episode {i}/{num_episodes}\n")
        log_f.flush()
    
    task.env.close()
    log_f.write(f"Finished processing scene {args.scene_name} at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_f.close()

    