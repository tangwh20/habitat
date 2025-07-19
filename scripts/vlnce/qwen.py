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




# QwenActionAgent: Load actions from a Qwen-style JSON file and map to HabitatSimActions
class QwenActionAgent(Agent):
    def __init__(self, qwen_json_path: str):
        with open(qwen_json_path, "r") as f:
            data = json.load(f)

        # Qwen JSON: actions is a list of lists, one per step
        # Flatten and concatenate all actions
        if "actions" in data:
            step_actions = data["actions"]
        elif "ACTIONS" in data:
            # fallback for other format
            step_actions = data["ACTIONS"]
        else:
            raise ValueError("Qwen JSON must contain an 'actions' key")

        all_actions = []
        for step in step_actions:
            all_actions.extend(step)

        self.actions = []
        for act in all_actions:
            if "FORWARD" in act:
                self.actions.append(HabitatSimActions.MOVE_FORWARD)
            elif "LEFT" in act:
                self.actions.append(HabitatSimActions.TURN_LEFT)
            elif "RIGHT" in act:
                self.actions.append(HabitatSimActions.TURN_RIGHT)
            elif "STOP" in act:
                self.actions.append(HabitatSimActions.STOP)
            else:
                self.actions.append(HabitatSimActions.STOP)  # fallback

        self.pointer = 0
        self.views = []

    def act(self, observations, current_step):
        if self.pointer < len(self.actions):
            action = self.actions[self.pointer]
            self.pointer += 1
        else:
            action = HabitatSimActions.STOP
        self.views.append(observations["rgb"])
        return action

    def reset(self, episode):
        self.pointer = 0

    @property
    def waypoints(self):
        return None
    

repo = git.Repo(".", search_parent_directories=True)
dir_path = repo.working_tree_dir
config_path = "config/benchmark/nav/vln_r2r.yaml"
output_path = os.path.join(
    dir_path, "/data1/jiyufeng/datasets/qwen_vlnce_rollout/"
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
                "seed": 200,
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
                "data_path": "data/datasets/vln/mp3d/r2r/v1/{split}/{split}.json.gz",
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


def rollout(env: habitat.Env, agent: Agent, save_dir: str):
    observations = env.reset()
    agent.reset(env.current_episode)

    os.makedirs(save_dir, exist_ok=True)

    step = 0
    while not env.episode_over:
        image = observations["rgb"]
        Image.fromarray(image).save(os.path.join(save_dir, f"step_{step}.png"))

        action = agent.act(observations, step)
        observations = env.step(action)
        step += 1


if __name__ == "__main__":
    # Create error log file
    import time
    err_f = open(f"logs/rollout_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log", "a")

    config, dataset = load_dataset(dir_path, config_path)
    print(config)

    # --- Specify your single rollout config here ---
    import math
    import numpy as np
    scene_id = "1LXtFkjw3qL"
    episode_id = "10181"
    start_position = [1.5, 0.0, 2.3]  # x, y, z in scene
    start_yaw = math.radians(90)     # heading in radians
    qwen_json_path = "/home/jiyufeng/ReasonNav/scripts/hallucination/test/1LXtFkjw3qL_10181/qwen.json"
    output_name = f"{scene_id}_{episode_id}"

    with habitat.Env(config=config, dataset=dataset) as env:
        # --- Locate the specific episode ---
        matched_episode = None
        for ep in dataset.episodes:
            print(f"Checking episode {ep.episode_id} in scene {ep.scene_id}")
            if ep.episode_id == episode_id and scene_id in ep.scene_id:
                matched_episode = ep
                break
        assert matched_episode is not None, "Episode not found"

        env.episode_iterator = iter([matched_episode])
        observations = env.reset()

        # Override agent pose (this is critical)
        from habitat.utils.geometry import quat_from_angle_axis
        env.sim.set_agent_state(position=start_position,
                                rotation=quat_from_angle_axis(start_yaw, np.array([0, 1, 0])))

        agent = QwenActionAgent(qwen_json_path)

        # Only a single rollout for this episode
        try:
            rollout(env, agent, os.path.join(output_path, output_name))
        except Exception as e:
            err_f.write(f"Error in episode {episode_id}: {e}\n")
            err_f.flush()
            print(f"Error in episode {episode_id}: {e}")

    err_f.close()
    print("Finished single rollout")
        
