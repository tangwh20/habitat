import os
import git
import json
import gzip

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
        self.actions = []

    def act(self, observations, current_step):
        if current_step < len(self.actions):
            action = self.actions[current_step]
        else:
            action = 0
        return action

    def reset(self, episode_id: str):
        self.actions = self.data[episode_id]["actions"]


repo = git.Repo(".", search_parent_directories=True)
dir_path = repo.working_tree_dir
config_path = "config/benchmark/nav/vln_r2r.yaml"
output_path = os.path.join(
    dir_path, "outputs/tutorials/vlnce"
)
os.makedirs(output_path, exist_ok=True)
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
                "data_path": "data/versioned_data/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz",
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
        gt_data_path = "/home/tangwenhao/Workspace/habitat/data/versioned_data/R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz"
        agent = FixedAgent(gt_data_path)

        num_episodes = 10
        for i in range(num_episodes):
            # Load the first episode and reset agent
            observations = env.reset()
            episode_id = str(env.current_episode.episode_id)
            agent.reset(episode_id)
            print(f"Episode {episode_id}")

            # Get metrics
            info = env.get_metrics()
            # Concatenate RGB-D observation and topdowm map into one image
            observations.pop("instruction")
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

            current_episode = env.current_episode
            video_name = f"{i}_{os.path.basename(current_episode.scene_id)}_{current_episode.episode_id}"
            images_to_video(
                vis_frames, output_path, video_name, fps=6, quality=9
            )
            vis_frames.clear()


        
        
