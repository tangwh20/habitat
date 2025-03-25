import os
import numpy as np
import quaternion
import git
import json
from PIL import Image
from typing import TYPE_CHECKING, cast, Union

import habitat
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.core.agent import Agent
from habitat.tasks.nav.nav import NavigationEpisode
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations.utils import images_to_video, overlay_frame
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import observations_to_image

# Quiet the Habitat simulator logging
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

if TYPE_CHECKING:
    from habitat.core.simulator import Observations
    from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim


repo = git.Repo(".", search_parent_directories=True)
dir_path = repo.working_tree_dir
data_path = os.path.join(dir_path, "data")
output_path = os.path.join(
    dir_path, "/data1/tangwenhao/datasets/split"
)
os.makedirs(output_path, exist_ok=True)
os.chdir(dir_path)


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

    def act(self, observations: "Observations") -> Union[int, np.ndarray]:
        return self.shortest_path_follower.get_next_action(
            cast(NavigationEpisode, self.env.current_episode).goals[0].position
        )

    def reset(self) -> None:
        pass

    @property
    def state(self):
        return self.env.sim.get_agent_state()


def example_top_down_map_measure():
    # Create habitat config
    config = habitat.get_config(
        config_path=os.path.join(
            dir_path,
            "config/benchmark/nav/pointnav/pointnav_mp3d.yaml",
        )
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
        config.habitat.environment.iterator_options.update(
            {
                "group_by_scene": False,
            }
        )
    # breakpoint()
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
        num_episodes = 100000
        for i in range(num_episodes):
            # Load the first episode and reset agent
            observations = env.reset()
            agent.reset()
            # if i < num_episodes - 1:
            #     continue

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

            current_episode = env.current_episode
            episode_dir = f"{os.path.basename(current_episode.scene_id).split('.')[0]}_{current_episode.episode_id}"
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
                        "rotations": rotations
                    }, f
                )
            

            print(f"Episode {i} done")
            print("=====================================")

            # video_name = f"{os.path.basename(current_episode.scene_id).split('.')[0]}_{current_episode.episode_id}"
            # # Create video from images and save to disk
            # images_to_video(
            #     vis_frames, output_path, video_name, fps=6, quality=9
            # )
            # vis_frames.clear()
            # # Display video
            # vut.display_video(f"{output_path}/{video_name}.mp4")


if __name__ == "__main__":
    example_top_down_map_measure()