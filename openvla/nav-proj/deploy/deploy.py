import draccus
import json
import time
import torch
from python.robot_interface import RobotInterface

from dataclasses import dataclass
from pathlib import Path
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


@dataclass
class EvalConfig:
    # model settings
    run_root_dir = Path("/media/yufeng/openvla/lora-instruct-scratch/run")
    exp_id: str = "openvla-7b+sacson+b16+lr-0.0005+lora-r32+dropout-0.0"
    run_dir = run_root_dir / exp_id
    device: str = "cuda:0"
    dataset_name: str = "sacson"

    # test settings
    sleep_interval_sec: int = 3
    ip: str = "10.100.0.5"


@draccus.wrap()
def main(cfg: EvalConfig) -> None:
    # settings
    device = cfg.device
    run_dir = cfg.run_dir
    ip = cfg.ip

    # load model from checkpoint
    processor = AutoProcessor.from_pretrained(run_dir, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        run_dir,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)

    # load dataset statistics
    with open(Path(run_dir) / "dataset_statistics.json", "r") as f:
        vla.norm_stats = json.load(f)

    # initialize robot interface
    homebot = RobotInterface(ip)

    # while True:
    image_array, timestamp = homebot.read_once_compressed(order="bgr", view="arm", disable_depth=True)
    image = Image.fromarray(image_array)
    # TODO: get instruction from input
    instruction = "move forward"

    # predict actions
    inputs = processor(instruction, image).to(device, dtype=torch.bfloat16)
    action_pred = vla.predict_action(**inputs, unnorm_key="sacson", do_sample=False).reshape(8, 2)

    # visualize actions
    homebot.visualize_waypoints(action_pred)

    # # sleep
    # time.sleep(cfg.sleep_interval_sec)


if __name__ == "__main__":
    main()
