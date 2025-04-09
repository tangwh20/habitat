import os
import json
import gzip

r2r_path = "/home/tangwenhao/Workspace/habitat/data/datasets/vln/mp3d/r2r/v1"
rxr_path = "/home/tangwenhao/Workspace/habitat/data/datasets/vln/mp3d/rxr/v1"

split = "train"

# Concatenate guide and follower data
# with gzip.open(f"{rxr_path}/{split}/{split}_guide.json.gz", "rt") as f:
#     train_guide = json.load(f)
# print(f"train_guide: {len(train_guide['episodes'])}")
# with gzip.open(f"{rxr_path}/{split}/{split}_follower.json.gz", "rt") as f:
#     train_follower = json.load(f)
# print(f"train_follower: {len(train_follower['episodes'])}")
# train_data = {"episodes": train_guide["episodes"] + train_follower["episodes"]}
# print(f"train_data: {len(train_data['episodes'])}")
# with gzip.open(f"{rxr_path}/{split}/{split}.json.gz", "wt") as f:
#     json.dump(train_data, f)


def generate_instruction_tokens(instruction: str):    
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
    return instruction_tokens


# Load R2R data
with gzip.open(f"{r2r_path}/{split}/{split}.json.gz", "rt") as f:
    r2r_data = json.load(f)
    instruction_vocab = r2r_data["instruction_vocab"]

num_traj = 119052

with gzip.open(f"{rxr_path}/{split}/{split}_raw.json.gz", "rt") as f:
    data = json.load(f)
    episodes = data["episodes"]

episodes_new = []
for i, episode in enumerate(episodes):
    episode_id = int(episode["episode_id"])
    trajectory_id = int(episode["trajectory_id"]) if "trajectory_id" in episode else episode_id
    scene_id = episode["scene_id"]
    start_position = episode["start_position"]
    start_rotation = episode["start_rotation"]
    info = {}
    goals = episode["goals"]
    instruction = {
        "instruction_text": episode["instruction"]["instruction_text"],
        "instruction_tokens": [0] * 200,
    }
    reference_path = episode["reference_path"]

    episode_new = {
        "episode_id": episode_id,
        "trajectory_id": trajectory_id,
        "scene_id": scene_id,
        "start_position": start_position,
        "start_rotation": start_rotation,
        "info": info,
        "goals": goals,
        "instruction": instruction,
        "reference_path": reference_path,
    }
    episodes_new.append(episode_new)

    print(f"Processed {i + 1}/{len(episodes)} episodes!")

print(f"Processed {len(episodes)} episodes!")
data_new = {
    "episodes": episodes_new,
    "instruction_vocab": instruction_vocab,
}

with open(f"{rxr_path}/{split}/{split}.json", "wt") as f:
    json.dump(data_new, f)
print(f"Saved {rxr_path}/{split}/{split}.json")
with gzip.open(f"{rxr_path}/{split}/{split}.json.gz", "wt") as f:
    json.dump(data_new, f)
print(f"Saved {rxr_path}/{split}/{split}.json.gz")
    
    