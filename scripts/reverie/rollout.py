import os
import json
import git
import quaternion
import numpy as np
from PIL import Image
from typing import TYPE_CHECKING, cast, Union

import habitat
from habitat.core.agent import Agent
from habitat.tasks.nav.nav import NavigationEpisode
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.utils.visualizations.utils import (
    images_to_video,
    overlay_frame,
    observations_to_image,
)

if TYPE_CHECKING:
    from habitat.core.simulator import Observations
    from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim

class ShortestPathFollowerAgent(Agent):
    r"""Implementation of the :ref:`habitat.core.agent.Agent` interface that
    uses :ref`habitat.tasks.nav.shortest_path_follower.ShortestPathFollower` utility class
    for extracting the action on the shortest path to the goal.
    """

    def __init__(self, env: habitat.Env, goal_radius: float):
        self.env = env
        self.shortest_path_follower = ShortestPathFollower(
            sim=cast("HabitatSim", env.sim),
            goal_radius=goal_radius,
            return_one_hot=False,
        )
        self.current_goal_index = 0

    def act(self, observations: "Observations") -> Union[int, np.ndarray]:
        goals = cast(NavigationEpisode, self.env.current_episode).goals
        goal_position = goals[self.current_goal_index].position
        # breakpoint()
        goal_position[1] = float(self.state.position[1])
        action = self.shortest_path_follower.get_next_action(goal_position)
        while action == 0:
            self.current_goal_index += 1
            if self.current_goal_index >= len(goals):
                return action
            else:
                goal_position = goals[self.current_goal_index].position
                goal_position[1] = float(self.state.position[1])
                action = self.shortest_path_follower.get_next_action(goal_position)
        return action

    def reset(self) -> None:
        self.current_goal_index = 0

    @property
    def state(self):
        return self.env.sim.get_agent_state()


def get_config(dir_path: str, config_path: str):    
    config = habitat.get_config(os.path.join(dir_path, config_path))

    # Add habitat.tasks.nav.nav.TopDownMap and habitat.tasks.nav.nav.Collisions measures
    with habitat.config.read_write(config):
        config.habitat.seed = 0
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
                # "data_path": "/home/tangwenhao/Workspace/habitat/scripts/example/example_episode.json.gz"
                "data_path": "/home/tangwenhao/Workspace/habitat/data/versioned_data/REVERIE_v1/train_new.json.gz"
            }
        )
        config.habitat.environment.iterator_options.update(
            {
                "group_by_scene": False,
            }
        )
    # breakpoint()
    return config


if __name__ == "__main__":
    # data_path = "/home/tangwenhao/Workspace/habitat/data/datasets/vln/mp3d/reverie/v1/train_new.json"
    # with open(data_path, "r") as f:
    #     traj_data = json.load(f)

    repo = git.Repo(".", search_parent_directories=True)
    dir_path = repo.working_tree_dir
    data_path = os.path.join(dir_path, "data")
    # output_path = os.path.join(
    #     dir_path, "outputs/reverie_split"
    # )
    output_path = "/data1/tangwenhao/datasets/reverie_split"
    os.makedirs(output_path, exist_ok=True)
    os.chdir(dir_path)


    # Read instructions from original json file
    with open("/home/tangwenhao/Workspace/habitat/data/datasets/vln/mp3d/reverie/v1/train.json", "r") as f:
        raw_data = json.load(f)

    # Create habitat config
    config_path = "config/benchmark/nav/pointnav/pointnav_mp3d.yaml"
    config = get_config(dir_path, config_path)
    # Create dataset
    dataset = habitat.make_dataset(
        id_dataset=config.habitat.dataset.type, config=config.habitat.dataset
    )

    # Create simulation environment
    with habitat.Env(config=config, dataset=dataset) as env:
        # breakpoint()
        # Create ShortestPathFollowerAgent agent
        agent = ShortestPathFollowerAgent(
            env=env,
            goal_radius=config.habitat.task.measurements.success.success_distance,
        )
        # Create video of agent navigating in the first episode
        num_episodes = len(raw_data)
        for i in range(num_episodes):
            # Load the first episode and reset agent
            observations = env.reset()
            agent.reset()

            current_episode = env.current_episode
            episode_dir = f"{os.path.basename(current_episode.scene_id).split('.')[0]}_{current_episode.episode_id}"
            if os.path.exists(f"{output_path}/{episode_dir}"):
                print(f"Episode {episode_dir} already exists, skipping...")
                continue

            # Get metrics
            info = env.get_metrics()
            # Concatenate RGB-D observation and topdowm map into one image
            frame = observations_to_image(observations, info)

            # Remove top_down_map from metrics
            info.pop("top_down_map")
            # Overlay numeric metrics onto frame
            frame = overlay_frame(frame, info)
            # Add fame to vis_frames
            # vis_frames = [frame]

            # Repeat the steps above while agent doesn't reach the goal
            rgbs = [observations["rgb"]]
            actions = []
            positions = []
            rotations = []
            while not env.episode_over:
                # Get the next best action
                action = agent.act(observations)
                if action is None:
                    break
                actions.append(action)

                # Get agent state
                positions.append(agent.state.position.tolist())
                rotations.append(quaternion.as_float_array(agent.state.rotation).tolist())

                # Step in the environment
                observations = env.step(action)
                rgbs.append(observations["rgb"])

                # info = env.get_metrics()
                # frame = observations_to_image(observations, info)

                # info.pop("top_down_map")
                # frame = overlay_frame(frame, info)
                # vis_frames.append(frame)
            rgbs.pop()
            assert len(rgbs) == len(actions)

            os.makedirs(f"{output_path}/{episode_dir}", exist_ok=True)
            for ii, rgb in enumerate(rgbs):
                Image.fromarray(rgb).save(f"{output_path}/{episode_dir}/{ii}.png")                

            # save actions and positions as json
            assert len(actions) == len(positions) == len(rotations)
            with open(f"{output_path}/{episode_dir}/actions.json", "w") as f:
                json.dump(
                    {
                        "actions": actions,
                        "positions": positions,
                        "rotations": rotations,
                        "instructions_l": raw_data[i]["instructions_l"],
                        "instructions": raw_data[i]["instructions"],
                    }, f
                )
            

            # video_name = f"{os.path.basename(current_episode.scene_id).split('.')[0]}_{current_episode.episode_id}"
            # # Create video from images and save to disk
            # images_to_video(
            #     vis_frames, output_path, video_name, fps=6, quality=9
            # )
            # vis_frames.clear()

            # breakpoint()
            # num_goals = len(cast(NavigationEpisode, current_episode).goals)
            print(f"Episode {i} done ({num_episodes} total)")
            print("=====================================")
