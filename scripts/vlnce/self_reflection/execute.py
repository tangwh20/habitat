import os
import git
import json
import gzip
import quaternion
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
        with open(data_path, "r") as f:
            self.data = json.load(f)
        self.actions = None
        self.forward_steps = None
        self.episode_id = None

        self.views = []
        self.positions = []
        self.rotations = []

    def act(self, observations, current_step):
        if current_step < len(self.actions):
            action = self.actions[current_step]
            self.views.append(observations["rgb"])
        else:
            action = 0
        return action
    
    def update_state(self, agent_state):
        position = agent_state.position.tolist()
        self.positions.append(position)

        rotation = quaternion.as_float_array(agent_state.rotation)
        rotation = np.roll(rotation, -1).tolist()  # Convert from quaternion to habitat sim format
        self.rotations.append(rotation)

    def reset(self, episode):
        self.episode_id = str(episode.trajectory_id)
        actions = self.data[self.episode_id]["actions"]
        if actions[-1] != 0:  # Ensure the last action is STOP
            actions.append(0)
        self.actions = np.array(actions)
        self.forward_steps = np.where(self.actions == 1)[0]

        self.views.clear()
        self.positions.clear()
        self.rotations.clear()

    @property
    def waypoints(self):
        return np.array(self.data[self.episode_id]["locations"])
    


repo = git.Repo(".", search_parent_directories=True)
dir_path = repo.working_tree_dir
config_path = "config/benchmark/nav/vln_r2r.yaml"
output_path = os.path.join(
    dir_path, "outputs", "self_reflection_test"
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
        config.habitat.update(
            {
                "seed": 100,
            }
        )
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
                "data_path": "outputs/self_reflection_test/1LXtFkjw3qL_129/stepwise_episode.json.gz"
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


def rollout(env: habitat.Env, agent: FixedAgent):
    # Load the first episode and reset agent
    observations = env.reset()
    episode_id = env.current_episode.episode_id
    trajectory_id = env.current_episode.trajectory_id
    scene_id = os.path.basename(env.current_episode.scene_id).split('.')[0]
    agent.reset(env.current_episode)

    # if os.path.exists(os.path.join(output_path, f"{scene_id}_{episode_id}", "states.json")):
    #     print(f"Episode {episode_id} already exists")
    #     return 
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
    collisions = []
    # Repeat the steps above while agent doesn't reach the goal
    while not env.episode_over:
        # Get the next best action
        agent.update_state(env.sim.get_agent_state())
        action = agent.act(observations, current_step)
        if action is None:
            break

        # Step in the environment
        observations = env.step(action)
        info = env.get_metrics()
        collisions.append(info["collisions"]["is_collision"])
        observations.pop("instruction")
        frame = observations_to_image(observations, info)

        info.pop("top_down_map")
        frame = overlay_frame(frame, info)
        vis_frames.append(frame)
        current_step += 1

    # Save image observations on each step
    episode_output_path = os.path.join(output_path, f"{scene_id}_{episode_id}", "stepwise", f"step_{trajectory_id}")
    image_output_path = os.path.join(episode_output_path, "images")
    os.makedirs(image_output_path, exist_ok=True)
    for step, view in enumerate(agent.views):
        image_path = os.path.join(image_output_path, f"{step}.png")
        Image.fromarray(view).save(image_path)
    agent.views.clear()

    # Save actions and positions as json
    # with open(os.path.join(episode_output_path, "actions.json"), "w") as f:
    #     json.dump(
    #         {
    #             "instruction": instruction_text,
    #             "actions": agent.actions.tolist(),
    #             "waypoints": agent.waypoints.tolist(),
    #         }, f
    #     )

    # Save positions and rotations as json
    with open(os.path.join(episode_output_path, "states.json"), "w") as f:
        # assert len(agent.positions) == len(agent.rotations) == len(agent.actions), \
        #     "Positions, rotations and actions must have the same length"
        try:
            positions = np.array(agent.positions)[agent.actions == 1].tolist()
            rotations = np.array(agent.rotations)[agent.actions == 1].tolist()
            filtered_collisions = np.array(collisions)[agent.actions == 1].tolist()
        except Exception as e:
            print(f"Error processing positions and rotations: {e}")
            breakpoint()
        json.dump(
            {
                "instruction": instruction_text,
                "num_steps": len(positions),
                "positions": positions,
                "rotations": rotations,
                "actions": agent.actions.tolist(),
                "collisions": filtered_collisions,
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


if __name__ == "__main__":
    # Create error log file
    import time
    err_f = open(f"logs/rollout_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log", "a")

    config, dataset = load_dataset(dir_path, config_path)
    print(config)
    
    # Create simulation environment
    with habitat.Env(config=config, dataset=dataset) as env:
        # breakpoint()
        gt_data_path = "outputs/self_reflection_test/1LXtFkjw3qL_129/qwen_actions.json"
        agent = FixedAgent(gt_data_path)

        num_episodes = len(agent.data)
        for i in range(num_episodes):
            # Run the agent in the environment
            try:
                rollout(env, agent)
            except Exception as e:
                err_f.write(f"Error in episode {env.current_episode.episode_id}: {e}\n")
                err_f.flush()
                print(f"Error in episode {env.current_episode.episode_id}: {e}")
                # continue

            print("===============================================================")
            print(f"Finished {i + 1}/{num_episodes} episodes")
            print("===============================================================")

    err_f.close()
    print("Finished all episodes")
        
