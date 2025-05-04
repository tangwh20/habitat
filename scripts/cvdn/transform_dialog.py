import os
import json
import gzip
import numpy as np
from scipy.spatial.transform import Rotation as R

data_path = "/home/tangwenhao/Workspace/habitat/data/datasets/vln/mp3d/cvdn"
version = "v1"
split = "train"

viewpoints_dict_path = "/data1/tangwenhao/datasets/matterport3d/viewpoints.json"

with open(f"{data_path}/{version}/{split}.json", "r") as f:
    data = json.load(f) 

with open(viewpoints_dict_path, "r") as f:
    viewpoints_dict = json.load(f)


def heading2rotation(rpos: np.ndarray) -> np.ndarray:
    assert rpos.shape == (3,)

    angle = np.arctan2(rpos[0], rpos[2])
    rot = R.from_euler('yxz', [angle, 0, np.pi])
    quat = rot.as_quat()

    return np.roll(quat, -1)



if __name__ == "__main__":

    episodes = []
    for i, traj in enumerate(data):
        # episode_id = str(i)
        scene_id = f"data/scene_datasets/mp3d/{traj['scan']}/{traj['scan']}.glb"
        viewpoints = [
            viewpoints_dict[viewpoint]["position"] for viewpoint in traj["nav_steps"] if viewpoint in viewpoints_dict
        ]
        viewpoints = np.array(viewpoints)
        viewpoints[:, 1] -= 1.35

        dialog_history = traj["dialog_history"]
        for j in range(0, len(dialog_history), 2):
            dialog_navigator = dialog_history[j]
            dialog_oracle = dialog_history[j + 1]

            nav_idx = dialog_navigator["nav_idx"]
            start_position = viewpoints[nav_idx]
            if nav_idx + 1 < len(viewpoints):
                start_rotation = heading2rotation(viewpoints[nav_idx + 1] - viewpoints[nav_idx])
            else:
                start_rotation = heading2rotation(viewpoints[nav_idx] - viewpoints[nav_idx - 1])
            
            episode = {
                "episode_id": str(i * 100 + j // 2),
                "scene_id": scene_id,
                "start_position": start_position.tolist(),
                "start_rotation": start_rotation.tolist(),
                "goals": [
                    {
                        "position": [0, 0, 0],
                        "radius": 0.5,
                    }
                ],
                "dialog": [
                    dialog_navigator,
                    dialog_oracle
                ],
                "target": traj["target"],
                "shortest_paths": None,
                "start_room": None
            }

            episodes.append(episode)
        print(f"Finished {i+1}/{len(data)}")

    traj_data = {
        "episodes": episodes,
    }

    with open(f"{data_path}/{version}/{split}_dialog.json", "w") as f:
        json.dump(traj_data, f)

    with gzip.open(f"{data_path}/{version}/{split}_dialog.json.gz", "wt") as f:
        json.dump(traj_data, f)


"""
data: list of 1299
data[0].keys(): ['navigator', 'oracle_quality', 'nav_camera', 'R2R_spl', 'oracle_mturk', 'R2R_oracle_spl', 'planner_nav_steps', 'oracle_avg_quality', 'navigator_avg_quality', 'idx', 'navigator_quality', 'end_panos', 'target', 'problem', 'scan', 'dialog_history', 'navigator_mturk', 'stop_history', 'start_pano', 'nav_steps', 'oracle']
"""
