import draccus
import glob
import io
import json
import numpy as np
import shutil
import torch
import tqdm
from typing import Any, Dict, Tuple, Type

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from PIL import Image

import matplotlib.pyplot as plt
import tensorflow as tf
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset


class EvalType(IntEnum):
    SINGLE_SPLIT = 4          # all steps in all traj from specified data split (default to test)
    ENTIRE_DATASET = 5        # entire dataset

@dataclass
class EvalConfig:
    # output settings
    output_root_dir = Path("/data/jiyufeng/openvla/eval/lora-instruct-scratch")
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


@dataclass
class EvalRLDSTransform(RLDSBatchTransform):

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        action = rlds_batch["action"][0]
        image = rlds_batch["observation"]["image_primary"][0]
        language_instruction = rlds_batch["task"]["language_instruction"].decode().lower()
        return dict(action=action, image=image, language_instruction=language_instruction)


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


def visualize_result(
    image: Image.Image,
    action_gt: np.ndarray, 
    action_pred: np.ndarray, 
    instruction: str, 
    save_path: Path):
    fig, axs = plt.subplots(1, 3)
    # show instruction as title
    fig.suptitle(
        instruction,
        horizontalalignment="center",
        verticalalignment="top",
        fontsize=12,
        wrap=True,
    )
    # show observation image
    axs[0].imshow(image)
    axs[0].axis('off')
    # plot actions
    plot_actions(axs[1], action_gt)
    axs[1].set_title("action_gt")
    plot_actions(axs[2], action_pred)
    axs[2].set_title("action_pred")
    # save and close
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close()



@draccus.wrap()
def eval_model(cfg: EvalConfig) -> None:
    # configs
    device_id = cfg.device
    run_dir = cfg.run_dir
    dataset_dir = cfg.data_root_dir / cfg.dataset_name
    output_dir = cfg.output_root_dir / cfg.dataset_name
    if output_dir.exists():
        shutil.rmtree(cfg.output_root_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # load from checkpoints directly
    processor = AutoProcessor.from_pretrained(run_dir, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        run_dir,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device_id)

    # load dataset statistics
    with open(Path(run_dir) / "dataset_statistics.json", "r") as f:
        vla.norm_stats = json.load(f)

    # load rlds dataset
    tfrecords = list(dataset_dir.rglob(f"*val.tfrecord*"))
    raw_dataset = tf.data.TFRecordDataset(tfrecords)

    # iterate each trajectory through the entire dataset
    for i, raw_record in tqdm.tqdm(enumerate(raw_dataset.as_numpy_iterator())):
        example = tf.train.Example()
        example.ParseFromString(raw_record)

        # get trajectory folder
        traj_path = Path(
            example.features.feature.get("episode_metadata/traj_folder")
            .bytes_list.value[0]
            .decode("utf-8")
        )

        # create output directory
        traj_name = traj_path.name
        traj_output_dir = Path(output_dir) / traj_name
        if not traj_output_dir.exists():
            Path(traj_output_dir).mkdir(parents=True, exist_ok=True)

        # get images list
        images = example.features.feature.get("steps/observation/image").bytes_list.value

        # get actions list
        actions = example.features.feature.get("steps/action").float_list.value
        actions = np.asarray(actions).reshape(-1, 8, 2)
        # length of actions and images should be the same
        assert actions.shape[0] == len(images)

        # get instructions and reasoning
        instructions = example.features.feature.get(
            "steps/language_instruction"
        ).bytes_list.value
        reasonings = example.features.feature.get(
            "steps/language_reasoning"
        ).bytes_list.value

        # traverse through each step
        for step in range(len(images)):
            image = Image.open(io.BytesIO(images[step]))
            # convert to numpy and inspect image size
            image_array = np.asarray(image)
            assert image_array.shape == (96, 96, 3)

            # compute action predictions
            instruction = instructions[step].decode().lower()
            inputs = processor(instruction, image).to(device_id, dtype=torch.bfloat16)
            action_pred = vla.predict_action(**inputs, unnorm_key="sacson", do_sample=False).reshape(8,2)
   
            # visualize step
            visualize_result(
                image=image,
                action_gt=actions[step],
                action_pred=action_pred,
                instruction=instruction,
                save_path=traj_output_dir / f"{traj_name}_step_{step}.jpg",
            )
        


if __name__ == "__main__":
    eval_model()