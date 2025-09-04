import os
import json
import gzip

split = "train"
data_path = f"/home/tangwenhao/Workspace/habitat/data/datasets/vln/mp3d/r2r/v1/{split}"
output_path = "/data1/tangwenhao/datasets/vlnce/data_raw"

os.chdir(data_path)
with gzip.open(f"{split}.json.gz", "rt") as f:
    data = json.load(f)
    instruction_vocab = data["instruction_vocab"]

episodes = []
for episode in data["episodes"]:
    episode_id = str(episode["episode_id"])
    scene_id = os.path.basename(episode["scene_id"]).split('.')[0]
    if scene_id == "D7N2EKCX4Sj" and episode_id == "5429":
        episodes.append(episode)
        break
#     if not os.path.exists(os.path.join(output_path, scene_id, episode_id)):
#         episodes.append(episode)

# episodes = data["episodes"]

# total_num = len(episodes)
# num_splits = 20
# split_size = total_num // num_splits
# for i in range(num_splits):
#     split_episodes = episodes[i * split_size: (i + 1) * split_size]
#     split_data = {
#         "episodes": split_episodes,
#         "instruction_vocab": instruction_vocab,
#     }
#     with gzip.open(f"{split}_{i}.json.gz", "wt") as f:
#         json.dump(split_data, f)

with gzip.open(f"{split}_0.json.gz", "wt") as f:
    json.dump({"episodes": episodes, "instruction_vocab": instruction_vocab}, f)

print("Number of entries:", len(episodes))