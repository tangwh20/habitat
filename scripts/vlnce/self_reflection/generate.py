import json
import os
import numpy as np
import torch

from pathlib import Path
from PIL import Image
from tqdm import tqdm

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.float16,
    ).to("cuda:0")
# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

def visualize_repsonse(
    context_image: Image.Image,
    task: str, 
    response: str,
    save_path: Path,
):
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    grid = gridspec.GridSpec(3, 4, figure=fig)  # 4 rows, 8 columns

    # Plot last image on the left
    ax_context_image = fig.add_subplot(grid[1:2, 0:1])
    ax_context_image.imshow(context_image)
    ax_context_image.axis("off")
    ax_context_image.set_title("Context Image")

    # Plot task + response
    ax_text = fig.add_subplot(grid[1:2, 1:4])
    ax_text.axis("off")
    ax_text.text(
        0,
        0.5,
        f"Task:\n{task}\n\n{response}",
        wrap=True,
        ha="left",
        va="center",
        fontsize=12,
    )

    # plot save_path
    fig.suptitle(
        save_path.name,
        horizontalalignment="center",
        verticalalignment="top",
        fontsize=6
    )

    # Tight layout and save
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    fig.savefig(save_path, dpi=300)
    plt.close()

def generate_reasoning_for_dataset(dataset_root: Path, output_dir: Path, max_trajs: int = None):
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    traj_dirs = sorted([p for p in dataset_root.iterdir() if p.is_dir()])
    if max_trajs:
        traj_dirs = traj_dirs[:max_trajs]

    for traj_path in tqdm(traj_dirs, desc="Processing Trajectories"):
        data_path = traj_path / "data.json"
        images_dir = traj_path / "images"
        if not data_path.exists() or not images_dir.exists():
            continue

        with open(data_path, "r") as f:
            data = json.load(f)

        positions = np.array(data["position"])  # (T, 2)
        yaws = np.array(data["yaw"])  # (T,)
        task_list = data["task"]

        obs_list = []
        reasoning_list = []
        actions_list = []

        for i in range(len(positions)):
            # Load context image
            context_image = Image.open(images_dir / f"{i}.png")

            # generate reasoning and instruction from Qwen
            response = generate_response_from_qwen(context_image, task_list[i])

            obs_list.append(response.get("OBSERVATION", ""))
            reasoning_list.append(response.get("REASONING", ""))
            actions_list.append(response.get("ACTIONS", []))

            # save response visualization
            # save_name = f"{traj_path.name}_step{i}.png"
            # save_path = output_dir / save_name
            # visualize_repsonse(
            #     context_image=context_image,
            #     task=task_list[i],
            #     response=response,
            #     save_path=save_path,
            # )

        # save the aggregated JSON
        output_traj_dir = output_dir / traj_path.name
        output_traj_dir.mkdir(parents=True, exist_ok=True)
        with open(output_traj_dir / "qwen.json", "w") as f:
            json.dump({
                "observation": obs_list,
                "reasoning": reasoning_list,
                "actions": actions_list
            }, f, indent=2)

def generate_response_from_qwen(image, task):
    # Build dynamic prompt using the provided image and task
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"You are a navigation agent. Given this image and the following goal: '{task}', output a JSON object with the following keys:\n\n\"OBSERVATION\": A description of what you see,\n\"REASONING\": A plan for reaching the goal,\n\"ACTIONS\": A list of exactly 8 actions to take immediately toward the goal.\n\nEach action must be one of: \"FORWARD 0.25M\", \"TURN LEFT 15 DEGREES\", \"TURN RIGHT 15 DEGREES\", or \"STOP\".\n\nReturn a raw JSON string with no commentary or code block markers (such as ```json)."},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda:0")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=1028)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # Parse the output text as JSON
    import re
    cleaned = re.sub(r"^```(?:json|python)?\n|\n```$", "", output_text[0].strip())
    try:
        return json.loads(cleaned)
    except Exception as e:
        print("Failed to parse JSON:", e)
        return {"OBSERVATION": "", "REASONING": "", "ACTIONS": []}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize subtask dataset")
    parser.add_argument("--dataset_root", type=Path, required=True, help="Path to the subtask dataset root directory")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save visualizations")
    parser.add_argument("--max_trajs", type=int, default=None, help="Maximum number of trajectories to visualize")

    args = parser.parse_args()

    generate_reasoning_for_dataset(args.dataset_root, args.output_dir, args.max_trajs)