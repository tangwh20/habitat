import os
import json
import gzip
import numpy as np

episodes = []

data_base_path = "/data1/tangwenhao/datasets/split"
instruction_base_path = "/data1/tangwenhao/datasets/gpt-4o"

output_base_path = "/home/tangwenhao/Workspace/habitat/outputs/gpt-4o_vlnce"
os.makedirs(output_base_path, exist_ok=True)

with gzip.open("/home/tangwenhao/Workspace/habitat/data/versioned_data/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz", "rt") as f:
    data = json.load(f)
    instruction_vocab = data["instruction_vocab"]

count = 0
for traj_id in os.listdir(instruction_base_path):
    scene_id = traj_id.split("_")[0]
    data_path = f"{data_base_path}/{traj_id}"
    instruction_path = f"{instruction_base_path}/{traj_id}"

    full_data = json.load(open(f"{data_path}/actions.json", "r"))
    actions = full_data["actions"]
    positions = full_data["positions"]
    rotations = full_data["rotations"]

    for name in os.listdir(instruction_path):
        if not name.endswith(".json"):
            continue
        start_step = int(name.split("_")[1].split(".")[0])
        instruction_data = json.load(open(f"{instruction_path}/{name}", "r"))
        instruction = instruction_data["instruction"]
        instruction_words = instruction.lower().split(' ')
        instruction_tokens = [0] * 200
        current_token_idx = 0
        for word in instruction_words:
            if word.endswith(".") or word.endswith(","):
                raw_word = word[:-1]
                punctuation = word[-1]
                instruction_tokens[current_token_idx] = instruction_vocab["word2idx_dict"].get(raw_word, 0)
                current_token_idx += 1
                instruction_tokens[current_token_idx] = instruction_vocab["word2idx_dict"].get(punctuation, 0)
                current_token_idx += 1
            else:
                instruction_tokens[current_token_idx] = instruction_vocab["word2idx_dict"].get(word, 0)
                current_token_idx += 1
                

        origin_step = start_step
        while actions[origin_step] != 1:
            origin_step += 1
        goal_step = origin_step + len(instruction_data["actions"]) - 1

        start_position = positions[origin_step]
        start_rotation = rotations[origin_step]
        start_rotation = np.roll(np.array(start_rotation), -1).tolist()
        goal_position = positions[goal_step]

        geodesic_distance = np.linalg.norm(
            np.array(start_position) - np.array(goal_position)
        ).item()

        count += 1
        episode = {
            "episode_id": count,
            "trajectory_id": count,
            "scene_id": f"mp3d/{scene_id}/{scene_id}.glb",
            "start_position": start_position,
            "start_rotation": start_rotation,
            "info": {
                "geodesic_distance": geodesic_distance
            },
            "goals": [{
                "position": goal_position,
                "radius": 3.0,
            }],
            "instruction": {
                "instruction_text": instruction,
                "instruction_tokens": instruction_tokens
            },
            "reference_path": positions[origin_step:goal_step + 1]
        }
        episodes.append(episode)
        # if count > 100:
        #     break

    print(f"Trajectory {traj_id} processed! Total episodes: {count}")
    # if count > 100:
    #     break

data = {
    "episodes": episodes,
    "instruction_vocab": instruction_vocab
}

with open(f"{output_base_path}/train.json", "w") as f:
    json.dump(data, f)

with gzip.open(f"{output_base_path}/train.json.gz", "wt") as f:
    json.dump(data, f)
