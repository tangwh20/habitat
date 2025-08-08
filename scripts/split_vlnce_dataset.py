import os
import json
import gzip

data_path = "/home/tangwenhao/Workspace/habitat/data/datasets/vln/mp3d/r2r/v1/train"
output_path = "/home/tangwenhao/Workspace/habitat/outputs/data/vlnce/data_raw"

os.chdir(data_path)
with gzip.open("train.json.gz", "rt") as f:
    data = json.load(f)
    instruction_vocab = data["instruction_vocab"]

episodes = []
for episode in data["episodes"]:
    episode_id = str(episode["episode_id"])
    scene_id = os.path.basename(episode["scene_id"]).split('.')[0]
    if not os.path.exists(os.path.join(output_path, scene_id, episode_id)):
        episodes.append(episode)

total_num = len(episodes)
num_splits = 10
split_size = total_num // num_splits
for i in range(num_splits):
    split_episodes = episodes[i * split_size: (i + 1) * split_size]
    split_data = {
        "episodes": split_episodes,
        "instruction_vocab": instruction_vocab,
    }
    with gzip.open(f"train_{i}.json.gz", "wt") as f:
        json.dump(split_data, f)


print("Number of entries:", len(data["episodes"]))