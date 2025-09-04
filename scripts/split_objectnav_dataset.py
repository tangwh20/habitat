import os
import shutil
import json
import gzip
from tqdm import tqdm

traj_dir = "/data1/tangwenhao/datasets/objectnav/data_raw"
split = "train"
data_path = "/home/tangwenhao/Workspace/habitat/data/datasets/objectnav/mp3d/v1"
os.chdir(data_path)
os.makedirs(f"{split}_0/content", exist_ok=True)
shutil.copy(f"{split}/{split}.json.gz", f"{split}_0/{split}.json.gz")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Split ObjectNav dataset by scene")
    parser.add_argument("--scene_filename", type=str, required=True, help="Scene name to filter by")
    args = parser.parse_args()

    # for filename in tqdm(os.listdir(f"{split}/content")):
    with gzip.open(f"{split}/content/{args.scene_filename}", "rt") as f:
        data = json.load(f)

    # episode_list = []
    # for episode in tqdm(data["episodes"]):
    #     episode_list.append(episode["episode_id"])
    # breakpoint()

    episodes = []
    for episode in tqdm(data["episodes"]):
        episode_id = episode["episode_id"]
        # scene_id = episode["scene_id"].split("/")[-1].split(".")[0]
        if episode_id in os.listdir(os.path.join(traj_dir, args.scene_filename.split(".")[0])):
            episodes.append(episode)
        # breakpoint()
    print(f"{args.scene_filename}: {len(data['episodes'])} -> {len(episodes)}")

    data["episodes"] = episodes
    with gzip.open(f"{split}_0/content/{args.scene_filename}", "wt") as f:
        json.dump(data, f)
        
