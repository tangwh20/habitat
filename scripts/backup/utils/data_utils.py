
import numpy as np
import pickle

from pathlib import Path

from dataclasses import dataclass

@dataclass
class EvalConfig:
    # output settings
    output_root_dir = Path("/home/tangwenhao/Workspace/habitat/outputs/eval_openvla")
    device: str = "cuda:1"

    # training settings
    batch_size: int = 1
    shuffle_buffer_size: int = 100_000                              # Dataloader shuffle buffer size (can reduce if OOM)

    # model settings
    run_root_dir = Path("/data/jiyufeng/openvla/lora-instruct-scratch/run")
    exp_id: str = "openvla-7b+sacson+b16+lr-0.0005+lora-r32+dropout-0.0"
    run_dir = run_root_dir / exp_id

    # dataset settings
    dataset_name: str = "sacson"
    data_root_dir = Path("/data/jiyufeng/openvla/datasets/")


def yaw_rotmat(yaw: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
    )

def to_local_coords(
    positions: np.ndarray, curr_pos: np.ndarray, curr_yaw: float
) -> np.ndarray:
    """
    Convert positions to local coordinates

    Args:
        positions (np.ndarray): positions to convert
        curr_pos (np.ndarray): current position
        curr_yaw (float): current yaw
    Returns:
        np.ndarray: positions in local coordinates
    """
    rotmat = yaw_rotmat(curr_yaw)
    if positions.shape[-1] == 2:
        rotmat = rotmat[:2, :2]
    elif positions.shape[-1] == 3:
        pass
    else:
        raise ValueError

    return (positions - curr_pos).dot(rotmat).astype(np.float32)

def get_gt_step_actions(traj_path: Path, step: int, len_traj_pred: int = 8, end_slack: int = 3) -> np.ndarray:
    with open((traj_path / "traj_data.pkl"), "rb") as f:
        traj_data = pickle.load(f)

    # calculate start and end index
    start_index = step
    end_index = step + len_traj_pred + 1
    if end_index >= len(traj_data["position"]) - len_traj_pred - end_slack:
        return

    # load position and yaw
    yaws = traj_data["yaw"][start_index:end_index].astype(np.float32)
    positions = traj_data["position"][start_index:end_index].astype(np.float32)

    # compute relative (x, y) coordinates of next n positions in current position
    waypoints = to_local_coords(positions, positions[0], yaws[0])
    actions = waypoints[1:]

    return actions