import os
import json
import git
import quaternion
import numpy as np
from PIL import Image
from typing import TYPE_CHECKING, cast, Union, Dict

import habitat

def get_config(dir_path: str, config_path: str):    
    config = habitat.get_config(os.path.join(dir_path, config_path))

    # Add habitat.tasks.nav.nav.TopDownMap and habitat.tasks.nav.nav.Collisions measures
    with habitat.config.read_write(config):
        config.habitat.seed = 0
        config.habitat.dataset.update(
            {
                "data_path": "/home/tangwenhao/Workspace/habitat/data/datasets/vln/mp3d/cvdn/v1/train_dialog.json.gz"
            }
        )
        config.habitat.environment.iterator_options.update(
            {
                "group_by_scene": False,
            }
        )
    # breakpoint()
    return config


def rollout(env: habitat.Env):
    # Load the first episode and reset agent
    observations = env.reset()

    current_episode = env.current_episode
    episode_dir = f"{os.path.basename(current_episode.scene_id).split('.')[0]}_{current_episode.episode_id}"
    if os.path.exists(f"{output_path}/{episode_dir}"):
        print(f"Episode {episode_dir} already exists, skipping...")
        return

    rgb = observations["rgb"]

    dialog = current_episode.dialog
    message_navigator = dialog[0]["message"]
    message_oracle = dialog[1]["message"]

    os.makedirs(f"{output_path}/{episode_dir}", exist_ok=True)
    Image.fromarray(rgb).save(f"{output_path}/{episode_dir}/image.png")

    # save actions and positions as json
    with open(f"{output_path}/{episode_dir}/dialog.json", "w") as f:
        json.dump(
            {
                "target": current_episode.target,
                "navigator": message_navigator,
                "oracle": message_oracle,
            }, f, indent=4
        )
    

if __name__ == "__main__":

    repo = git.Repo(".", search_parent_directories=True)
    dir_path = repo.working_tree_dir
    data_path = os.path.join(dir_path, "data")
    # output_path = os.path.join(
    #     dir_path, "outputs/cvdn_dialog"
    # )
    output_path = "/data1/tangwenhao/datasets/cvdn_dialog"
    os.makedirs(output_path, exist_ok=True)
    os.chdir(dir_path)

    import time
    err_f = open(f"err_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log", "a")

    # Read instructions from original json file
    with open("/home/tangwenhao/Workspace/habitat/data/datasets/vln/mp3d/cvdn/v1/train_dialog.json", "r") as f:
        raw_data = json.load(f)
    
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
        # Create video of agent navigating in the first episode
        num_episodes = len(raw_data["episodes"])
        print(f"Number of episodes: {num_episodes}")
        print("=====================================")
        for i in range(num_episodes):
            try:
                rollout(env)
            except Exception as e:
                print(f"Error in episode {i}: {e}")
                err_f.write(f"Error in episode {i}: {e}\n")
                err_f.flush()
                continue

            # breakpoint()
            print(f"Episode {i+1} done ({num_episodes} total)")
            print("=====================================")

    err_f.close()