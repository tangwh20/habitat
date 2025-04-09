import os
import git
import json
import gzip
import numpy as np
from PIL import Image

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


class FixedAgent(Agent):
    """
    An agent that follows a fixed sequence of actions.
    """
    def __init__(self, data_path: str):
        with gzip.open(data_path, "rt") as f:
            self.data = json.load(f)
        self.actions = None
        self.forward_steps = None
        self.reference_waypoint_steps = None
        self.reference_action_steps = None
        self.episode_id = None

        self.views = []

    def act(self, observations, current_step):
        if current_step < len(self.actions):
            action = self.actions[current_step]
            # if current_step in self.reference_steps:
            #     self.reference_views.append(observations["rgb"])
            self.views.append(observations["rgb"])
        else:
            action = 0
        return action

    def reset(self, episode):
        self.episode_id = str(episode.episode_id)
        self.actions = np.array(self.data[self.episode_id]["actions"])
        self.forward_steps = np.where(self.actions == 1)[0]

        reference_path = np.array(episode.reference_path)
        gt_path = np.array(self.data[self.episode_id]["locations"])

        self._get_reference_steps(reference_path, gt_path)

    def _get_reference_steps(self, reference_path, gt_path):
        distance_to_reference = np.linalg.norm(
            gt_path[:, None, :] - reference_path[None, :, :], axis=2
        )
        self.reference_waypoint_steps = np.argmin(distance_to_reference, axis=0)
        movement_steps = self.forward_steps.tolist()
        movement_steps.append(movement_steps[-1] + 1)
        movement_steps = np.array(movement_steps)
        self.reference_action_steps = movement_steps[self.reference_waypoint_steps]

    @property
    def waypoints(self):
        return np.array(self.data[self.episode_id]["locations"])
    


repo = git.Repo(".", search_parent_directories=True)
dir_path = repo.working_tree_dir
config_path = "config/benchmark/nav/vln_r2r.yaml"
output_path = os.path.join(
    dir_path, "/data1/tangwenhao/datasets/vlnce_rxr_split"
)
os.chdir(dir_path)


def load_dataset(dir_path, config_path):
    # Create habitat config
    config = habitat.get_config(
        config_path=os.path.join(dir_path, config_path)
    )
    # Add habitat.tasks.nav.nav.TopDownMap and habitat.tasks.nav.nav.Collisions measures
    with habitat.config.read_write(config):
        # breakpoint()
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
        config.habitat.dataset.update(
            {
                "data_path": "data/datasets/vln/mp3d/rxr/v1/{split}/{split}.json.gz",
            }
        )
        config.habitat.environment.iterator_options.update(
            {
                "group_by_scene": False,
            }
        )
    # Create dataset
    dataset = habitat.make_dataset(
        id_dataset=config.habitat.dataset.type, config=config.habitat.dataset
    )

    return config, dataset


if __name__ == "__main__":
    config, dataset = load_dataset(dir_path, config_path)
    print(config)
    
    # Create simulation environment
    with habitat.Env(config=config, dataset=dataset) as env:
        # breakpoint()
        # gt_data_path = "/home/tangwenhao/Workspace/habitat/data/versioned_data/R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz"
        gt_data_path = "/home/tangwenhao/Workspace/habitat/data/datasets/vln/mp3d/rxr/v1/train/train_gt.json.gz"
        agent = FixedAgent(gt_data_path)

        num_episodes = 119052
        for i in range(num_episodes):
            # Load the first episode and reset agent
            observations = env.reset()
            episode_id = env.current_episode.episode_id
            scene_id = os.path.basename(env.current_episode.scene_id).split('.')[0]
            agent.reset(env.current_episode)

            if os.path.exists(os.path.join(output_path, f"{scene_id}_{episode_id}")):
                print(f"Episode {episode_id} already exists")
                continue
            print(f"Episode {episode_id} started")

            # Get metrics
            info = env.get_metrics()
            # Concatenate RGB-D observation and topdowm map into one image
            instruction = observations.pop("instruction")
            instruction_text = instruction["text"]
            frame = observations_to_image(observations, info)

            # Remove top_down_map from metrics
            info.pop("top_down_map")
            # Overlay numeric metrics onto frame
            frame = overlay_frame(frame, info)
            # Add fame to vis_frames
            vis_frames = [frame]

            current_step = 0
            # Repeat the steps above while agent doesn't reach the goal
            while not env.episode_over:
                # Get the next best action
                action = agent.act(observations, current_step)
                if action is None:
                    break

                # Step in the environment
                observations = env.step(action)
                info = env.get_metrics()
                observations.pop("instruction")
                frame = observations_to_image(observations, info)

                info.pop("top_down_map")
                frame = overlay_frame(frame, info)
                vis_frames.append(frame)
                current_step += 1

            # Save image observations on each reference step
            episode_output_path = os.path.join(output_path, f"{scene_id}_{episode_id}")
            image_output_path = os.path.join(episode_output_path, "images")
            os.makedirs(image_output_path, exist_ok=True)
            for step, view in enumerate(agent.views):
                if step in agent.reference_action_steps:
                    image_path = os.path.join(image_output_path, f"step_{step}_ref.png")
                else:
                    image_path = os.path.join(image_output_path, f"step_{step}.png")
                Image.fromarray(view).save(image_path)
            agent.views.clear()

            # Save actions and positions as json
            with open(os.path.join(episode_output_path, "actions.json"), "w") as f:
                json.dump(
                    {
                        "instruction": instruction_text,
                        "actions": agent.actions.tolist(),
                        "reference_action_steps": agent.reference_action_steps.tolist(),
                        "waypoints": agent.waypoints.tolist(),
                        "reference_waypoint_steps": agent.reference_waypoint_steps.tolist()
                    }, f
                )

            # Save video
            # video_output_path = os.path.join(output_path, "videos")
            # os.makedirs(video_output_path, exist_ok=True)
            # video_name = f"{i}_{scene_id}_{episode_id}"
            # images_to_video(
            #     vis_frames, video_output_path, video_name, fps=6, quality=9
            # )
            # vis_frames.clear()

            print("===============================================================")
            print(f"Finished {i + 1}/{num_episodes} episodes")
            print("===============================================================")


        
        
