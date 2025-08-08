import os
import json
import git
import quaternion
import numpy as np
from PIL import Image
from typing import TYPE_CHECKING, cast, Union, Dict, List

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
        config.habitat.dataset.update(
            {
                "data_path": "/home/tangwenhao/Workspace/habitat/data/datasets/vln/mp3d/cvdn/v1/train_new.json.gz"
            }
        )
        config.habitat.environment.iterator_options.update(
            {
                "group_by_scene": False,
            }
        )
    # breakpoint()
    return config


def get_extra_data(single_raw_data, viewpoints_dict):
    nav_camera = single_raw_data["nav_camera"]
    nav_steps = single_raw_data["nav_steps"]
    dialog_history = single_raw_data["dialog_history"]

    def transform_camera_message(camera_message: Dict):
        message = {
            "heading": camera_message["heading"],
            "elevation": camera_message["elevation"],
            "position": viewpoints_dict[camera_message["pano"]]
        }

        return message
    
    def transform_dialog_message(dialog_message: Dict):
        viewpoint = nav_steps[dialog_message["nav_idx"]]
        message = {
            "role": dialog_message["role"],
            "message": dialog_message["message"],
            "position": viewpoints_dict[viewpoint]
        }

        return message
        
    
    camera_info = [
        {
            "dia_idx": obs["dia_idx"],
            "message": [
                transform_camera_message(message) for message in obs["message"] if message["pano"] in viewpoints_dict
            ]
        } for obs in nav_camera
    ]

    dialog_info = [
        transform_dialog_message(obs) for obs in dialog_history if nav_steps[obs["nav_idx"]] in viewpoints_dict
    ]

    return {
        "camera_info": camera_info,
        "dialog_info": dialog_info,
        "target": single_raw_data["target"],
    }


def process_dialog(dialog: List[Dict], goal_idxs: List[int]):
    dialog_data = []
    for dialog_item in dialog:
        nav_idx = dialog_item["nav_idx"]
        nav_idx_new = np.where(np.array(goal_idxs) == nav_idx)[0]
        dialog_data.append(
            {
                "nav_idx": int(nav_idx_new[0]) if len(nav_idx_new) > 0 else -1,
                "role": dialog_item["role"],
                "message": dialog_item["message"],
            }
        )
    return dialog_data


def rollout(env: habitat.Env, agent: ShortestPathFollowerAgent):
    # Load the first episode and reset agent
    observations = env.reset()
    agent.reset()

    current_episode = env.current_episode
    episode_dir = f"{os.path.basename(current_episode.scene_id).split('.')[0]}_{current_episode.episode_id}"
    if os.path.exists(f"{output_path}/{episode_dir}"):
        print(f"Episode {episode_dir} already exists, skipping...")
        return

    # Get metrics
    info = env.get_metrics()
    # Concatenate RGB-D observation and topdowm map into one image
    frame = observations_to_image(observations, info)

    # Overlay numeric metrics onto frame
    frame = overlay_frame(frame, info)
    # Add fame to vis_frames
    vis_frames = [frame]

    # Repeat the steps above while agent doesn't reach the goal
    rgbs = [observations["rgb"]]
    goal_idxs = []
    actions = []
    positions = []
    rotations = []
    while not env.episode_over:
        # Get the next best action
        goal_idxs.append(agent.current_goal_index)
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

        info = env.get_metrics()
        frame = observations_to_image(observations, info)

        frame = overlay_frame(frame, info)
        vis_frames.append(frame)
    rgbs.pop()
    assert len(rgbs) == len(actions)

    os.makedirs(f"{output_path}/{episode_dir}", exist_ok=True)
    for ii, rgb in enumerate(rgbs):
        Image.fromarray(rgb).save(f"{output_path}/{episode_dir}/{ii}.png")

    target = current_episode.target
    dialog = current_episode.dialog

    # save actions and positions as json
    assert len(actions) == len(positions) == len(rotations)
    with open(f"{output_path}/{episode_dir}/actions.json", "w") as f:
        json.dump(
            {
                "target": target,
                "dialog": process_dialog(dialog, goal_idxs),
                "actions": actions,
                "positions": positions,
                "rotations": rotations,
                "goal_idxs": goal_idxs,
            }, f
        )
    

    # video_name = f"{os.path.basename(current_episode.scene_id).split('.')[0]}_{current_episode.episode_id}"
    # # Create video from images and save to disk
    # images_to_video(
    #     vis_frames, output_path, video_name, fps=6, quality=9
    # )
    # vis_frames.clear()


if __name__ == "__main__":

    repo = git.Repo(".", search_parent_directories=True)
    dir_path = repo.working_tree_dir
    data_path = os.path.join(dir_path, "data")
    # output_path = os.path.join(
    #     dir_path, "outputs/cvdn_split"
    # )
    output_path = "/data1/tangwenhao/datasets/cvdn_with_dialog"
    os.makedirs(output_path, exist_ok=True)
    os.chdir(dir_path)

    import time
    err_f = open(f"err_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log", "a")
    
    # Read viewpoints map
    with open("/data1/tangwenhao/datasets/matterport3d/viewpoints.json", "r") as f:
        viewpoints_dict = json.load(f)

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
        num_episodes = 1299 # len(raw_data)
        for i in range(num_episodes):
            try:
                rollout(env, agent)
            except Exception as e:
                print(f"Error in episode {i}: {e}")
                err_f.write(f"Error in episode {i}: {e}\n")
                err_f.flush()
                continue

            # breakpoint()
            print(f"Episode {i} done ({num_episodes} total)")
            print("=====================================")

    err_f.close()