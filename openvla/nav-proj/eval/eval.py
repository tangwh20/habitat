import draccus
import json
import glob
import moviepy.video.io.ImageSequenceClip
import multiprocessing
import numpy as np
import random
import torch

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from PIL import Image

import matplotlib.pyplot as plt
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers.image_processing_utils import BatchFeature
from utils.data_utils import get_gt_step_actions


class EvalType(IntEnum):
    RANDOM_SAMPLE = 1         # random step, random traj from specified data split (default to test)
    SINGLE_STEP = 2           # specified step, specified traj (use defaulf if not provided)
    SINGLE_TRAJ = 3           # all steps in specified traj (use defaulf if not provided)
    SINGLE_SPLIT = 4          # all steps in all traj from specified data split (default to test)
    ENTIRE_DATASET = 5        # entire dataset

@dataclass
class EvalConfig:
    # eval settings
    eval_type: EvalType = EvalType.RANDOM_SAMPLE
    eval_split: str = "test"
    device: str = "cuda:0"

    # dataset name trajectory name & step for eval
    dataset_name: str = "sacson"
    traj_name: str = "Dec-12-2022-bww8_00000034_1"

    # training settings
    run_root_dir: Path = Path("/data/jiyufeng/openvla/lora/run")
    exp_id: str = "openvla-7b+sacson+b16+lr-0.0005+lora-r32+dropout-0.0"
    run_dir: Path = run_root_dir / exp_id

    # eval input & output settings
    data_split_dir: Path = Path("/data/jiyufeng/nomad_dataset/data_splits")
    data_root_dir: Path = Path("/data/jiyufeng/nomad_dataset")
    output_root_dir: Path = Path("/data/jiyufeng/openvla/eval")
    video_fps = 4    

@draccus.wrap()
def eval_model(cfg: EvalConfig) -> None:
    # data dirs
    data_split_dir = cfg.data_split_dir / cfg.dataset_name
    data_dir = cfg.data_root_dir / cfg.dataset_name
    output_dir = cfg.output_root_dir / cfg.dataset_name

    # load from checkpoints directly
    processor = AutoProcessor.from_pretrained(cfg.run_dir, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.run_dir,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(cfg.device)

    # load dataset statistics
    with open(Path(cfg.run_dir) / "dataset_statistics.json", "r") as f:
        vla.norm_stats = json.load(f)

    # format prompt
    instruction = "continue the trajectory"
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"

    def eval_step(step_path: Path):
        assert(step_path.exists() == True, f"Evaluating step {step_path} but does not exist!")
        traj_name = step_path.parent.name
        step = int(step_path.stem)

        # get ground truth and pred actions
        gt_actions = get_gt_step_actions(traj_path=step_path.parent, step=step)
        if gt_actions is None:
            return

        # load images in context window
        image_path = data_dir / traj_name / f"{step}.jpg"
        image = Image.open(image_path)
        
        # predict actions
        inputs = processor(prompt, image).to(cfg.device, dtype=torch.bfloat16)
        pred_actions = vla.predict_action(**inputs, unnorm_key="sacson", do_sample=False).reshape(8,2)

        # show last image in plot
        fig, axs = plt.subplots(1, 3)
        
        axs[0].imshow(image)
        axs[0].axis('off')
        axs[0].set_title(f"step_{step}")

        def plot_actions(ax, actions, color="b"):
            lim = np.max(np.abs(actions))
            # switch x and y axis, as x represents forward movement in real world
            ax.plot(-actions[:, 1], actions[:, 0], f"{color}o")  # 'o' means circles
            ax.plot(-actions[:, 1], actions[:, 0], f"{color}-")  # '-' means solid line
            # mark start spot as green and end spot as red 
            ax.plot(-actions[0, 1], actions[0, 0], f"go")  # 'go' means green circles
            ax.plot(-actions[-1, 1], actions[-1, 0], f"ro")  # 'go' means red circles
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_box_aspect(aspect=1)

        # plot actions
        plot_actions(axs[1], gt_actions)
        axs[1].set_title("gt_actions")
        plot_actions(axs[2], pred_actions)
        axs[2].set_title("pred_actions")

        # set title and layout
        fig.suptitle(f'Trajectory: {traj_name}', fontsize=12)
        fig.tight_layout()

        # save and close
        if not Path(output_dir / traj_name).exists():
            Path(output_dir / traj_name).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(output_dir / traj_name / f"{traj_name}_step_{step}.jpg"), dpi=300)
        plt.close()

    def eval_traj(traj_path: Path):
        print("evaluating traj: ", traj_path)
        step_paths = list(traj_path.glob("*.jpg"))
        step_paths.sort()
        for step_path in step_paths:
            eval_step(step_path)
        # generate video file
        # traj_out_path = Path(cfg.output_root_dir / traj_path.name)
        # image_paths = list(traj_out_path.glob("*.jpg"))
        # image_files = [str(path) for path in image_paths]
        # clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=cfg.fps)
        # clip.write_videofile(str(cfg.output_root_dir / f'{traj_path.name}.mp4'))
        # print("finished generating video for trajectory: ", traj_path.name)

    # eval
    if cfg.eval_type == EvalType.RANDOM_SAMPLE:
        with open(Path(data_split_dir / cfg.eval_split / "traj_names.txt"), "rb") as f:
            traj_names = f.read().decode("utf-8").splitlines() 
        traj_path = Path(data_dir / random.choice(traj_names))
        step_path = random.choice(list(traj_path.glob("*.jpg")))
        eval_step(step_path)
    elif cfg.eval_type == EvalType.SINGLE_STEP:
        eval_step(data_dir/ cfg.traj_name / f"{cfg.step}.jpg")
    elif cfg.eval_type == EvalType.SINGLE_TRAJ:
        eval_traj(data_dir / cfg.traj_name)
    elif cfg.eval_type == EvalType.SINGLE_SPLIT:
        with open(Path(data_split_dir / cfg.eval_split / "traj_names.txt"), "rb") as f:
            traj_names = f.read().decode("utf-8").splitlines()
        traj_paths = [Path(data_dir / str(traj_name)) for traj_name in traj_names]
        for traj_path in traj_paths:
            eval_traj(traj_path)
    elif cfg.eval_type == EvalType.ENTIRE_DATASET:
        traj_paths = list(data_dir.iterdir())
        for traj_path in traj_paths:
            eval_traj(traj_path)
    else:
        raise KeyError("Not supported evaluation type: ", cfg.eval_type)


if __name__ == "__main__":
    eval_model()