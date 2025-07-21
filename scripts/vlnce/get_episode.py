import os
import json
import gzip
import copy


base_path = "/home/tangwenhao/Workspace/habitat/outputs/self_reflection_test"

train_data_path = "/home/tangwenhao/Workspace/habitat/data/datasets/vln/mp3d/r2r/v1/train/train.json.gz"
gt_data_path = "/home/tangwenhao/Workspace/habitat/data/datasets/vln/mp3d/r2r/v1/train/train_gt.json.gz"
with gzip.open(train_data_path, "rt") as f:
    train_data = json.load(f)
    instruction_vocab = train_data["instruction_vocab"]
with gzip.open(gt_data_path, "rt") as f:
    train_data_gt = json.load(f)


def get_episode(episode_id: str):
    scene_name, episode_num = episode_id.split("_")
    episode_num = int(episode_num)
    # episode_data = train_data["episodes"][episode_num - 1]
    for i in range(len(train_data["episodes"])):
        if train_data["episodes"][i]["episode_id"] == episode_num:
            episode_data = train_data["episodes"][i]
            break
    
    # gt_data = train_data_gt[str(episode_num)]
    gt_data = {episode_num: train_data_gt[str(episode_num)]}

    with open(os.path.join(base_path, episode_id, "episode.json"), "w") as f:
        json.dump(episode_data, f)
    with open(os.path.join(base_path, episode_id, "gt.json"), "w") as f:
        json.dump(gt_data, f)

    return episode_data, gt_data


def get_stepwise_episode(episode_id: str):
    scene_name, episode_num = episode_id.split("_")

    # Generate stepwise episode data for rollout
    with open(os.path.join(base_path, episode_id, "episode.json"), "r") as f:
        episode_data = json.load(f)
    with open(os.path.join(base_path, episode_id, "states.json"), "r") as f:
        states_data = json.load(f)

    episodes = []
    for i in range(states_data["num_steps"]):
        step_data = copy.deepcopy(episode_data)
        step_data["trajectory_id"] = i
        step_data["start_position"] = states_data["positions"][i]
        step_data["start_rotation"] = states_data["rotations"][i]
        episodes.append(step_data)

    stepwise_episode_data = {
        "episodes": episodes,
        "instruction_vocab": instruction_vocab,
    }

    with gzip.open(os.path.join(base_path, episode_id, "stepwise_episode.json.gz"), "wt") as f:
        json.dump(stepwise_episode_data, f)

    # Generate Qwen action data for reference
    action_map = {
        "STOP": 0,
        "FORWARD 0.25M": 1,
        "TURN LEFT 15 DEGREES": 2,
        "TURN RIGHT 15 DEGREES": 3,
    }

    with open(os.path.join(base_path, episode_id, "qwen.json"), "r") as f:
        qwen_data = json.load(f)
        qwen_actions = qwen_data["actions"]
    
    habitat_actions = []
    for i in range(len(qwen_actions) - 5):
        actions = qwen_actions[i]
        habitat_actions.append(
            [action_map.get(act, 0) for act in actions if act in action_map]
        )
    
    with open(os.path.join(base_path, episode_id, "qwen_actions.json"), "w") as f:
        json.dump({
            f"{i}": {"actions": habitat_actions[i]} for i in range(len(habitat_actions))
        }, f)



if __name__ == "__main__":
    # Example usage
    # episode_id = "1LXtFkjw3qL_129"

    # for episode_id in os.listdir(base_path):
    #     get_stepwise_episode(episode_id)

    # preview_data = {}
    # preview_data_gt = {}

    # episodes = []
    for episode_id in os.listdir(base_path):
        episode_data, gt_data = get_episode(episode_id)
    #     episodes.append(episode_data)

    #     episode_num = episode_id.split("_")[1]
    #     preview_data_gt[episode_num] = gt_data

    #     print(f"Episode {episode_id} data saved.")
    
    # for key in train_data:
    #     if key == "episodes":
    #         preview_data[key] = episodes
    #     else:
    #         preview_data[key] = train_data[key]

    # with gzip.open("scripts/vlnce/preview_data.json.gz", "wt") as f:
    #     json.dump(preview_data, f)
    # with gzip.open("scripts/vlnce/preview_data_gt.json.gz", "wt") as f:
    #     json.dump(preview_data_gt, f)