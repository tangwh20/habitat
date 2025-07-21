import json
import os
import numpy as np
import torch

# from pathlib import Path
from typing import List, Dict, Any
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


base_path = "/data1/tangwenhao/datasets/vlnce/data/train_old/subtask"

def visualize_repsonse(
    context_image: Image.Image,
    nav_images: List[Image.Image],
    task: str, 
    reasoning: str,
    actions: List[str],
    response: str,
    save_path: str,
):
    # Create figure
    fig = plt.figure(figsize=(12, 6))
    grid = gridspec.GridSpec(2, 8, figure=fig)  # 2 rows, 8 columns

    # Plot navigation images on the top row
    for i, img in enumerate(nav_images):
        ax = fig.add_subplot(grid[0, i])
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"Step {i+1}")

    # Plot last image on the left
    ax_context_image = fig.add_subplot(grid[1:2, 0:2])
    ax_context_image.imshow(context_image)
    ax_context_image.axis("off")
    ax_context_image.set_title("Context Image")

    # Plot task + response
    ax_text = fig.add_subplot(grid[1:2, 2:8])
    ax_text.axis("off")
    ax_text.text(
        0,
        0.5,
        f"Task:\n{task}\nReasoning:\n{reasoning}\nActions:\n{actions}\n\n{response}",
        wrap=True,
        ha="left",
        va="center",
        fontsize=12,
    )

    # plot save_path
    fig.suptitle(
        save_path,
        horizontalalignment="center",
        verticalalignment="top",
        fontsize=6
    )

    # Tight layout and save
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    fig.savefig(save_path, dpi=300)
    plt.close()


def generate_reasoning_for_episode(episode_id: str, output_dir: str):
    episode_path = os.path.join(base_path, episode_id)
    data_path = os.path.join(episode_path, "data.json")
    images_path = os.path.join(episode_path, "images")

    output_path = os.path.join(output_dir, episode_id)
    qwen_data_path = os.path.join(output_path, "qwen.json")
    qwen_stepwise_path = os.path.join(output_path, "stepwise")

    os.makedirs(output_path, exist_ok=True)
    if not os.path.exists(data_path) or not os.path.exists(images_path):
        print(f"Skipping {episode_id}, missing data or images.")
        return
    
    with open(data_path, "r") as f:
        data = json.load(f)
    with open(qwen_data_path, "r") as f:
        qwen_data = json.load(f)

    positions = np.array(data["position"])  # (T, 2)
    yaws = np.array(data["yaw"])  # (T,)
    task_list = data["task"]

    obs_list = qwen_data["observation"]
    reasoning_list = qwen_data["reasoning"]
    actions_list = qwen_data["actions"]
    
    outputs = []
    for i in range(len(os.listdir(qwen_stepwise_path))):
        # Load context image
        context_image = Image.open(os.path.join(images_path, f"{i}.png"))

        # Load navigation images
        navigation_images = []
        for j in range(len(os.listdir(os.path.join(qwen_stepwise_path, f"step_{i}", "images")))):
            img_path = os.path.join(qwen_stepwise_path, f"step_{i}", "images", f"{j}.png")
            navigation_images.append(Image.open(img_path))
        
        # Generate verdict and explanation from Qwen
        response = generate_response_from_qwen(
            image=context_image,
            task=task_list[i],
            images=navigation_images,
            reasoning=reasoning_list[i],
            actions=actions_list[i]
        )

        outputs.append({
            "step": i,
            "task": task_list[i],
            "observation": obs_list[i],
            "reasoning": reasoning_list[i],
            "actions": actions_list[i],
            "response": response
        })

        # Save response visualization
        save_path = os.path.join(output_path, "visualizations", f"step_{i}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        visualize_repsonse(
            context_image=context_image,
            nav_images=navigation_images,
            task=task_list[i],
            reasoning=reasoning_list[i],
            actions=actions_list[i],
            response=response,
            save_path=save_path
        )

    # Save the aggregated JSON
    with open(os.path.join(output_path, "qwen_aggregated.json"), "w") as f:
        json.dump(outputs, f, indent=2)


def generate_response_from_qwen(image, task, images=None, reasoning=None, actions=None):
    # Build dynamic prompt using the provided image and task
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "image", "image": image},
    #             {"type": "text", "text": f"You are a navigation agent. Given this image and the following goal: '{task}', output a JSON object with the following keys:\n\n\"OBSERVATION\": A description of what you see,\n\"REASONING\": A plan for reaching the goal,\n\"ACTIONS\": A list of exactly 8 actions to take immediately toward the goal.\n\nEach action must be one of: \"FORWARD 0.25M\", \"TURN LEFT 15 DEGREES\", \"TURN RIGHT 15 DEGREES\", or \"STOP\".\n\nReturn a raw JSON string with no commentary or code block markers (such as ```json)."},
    #         ],
    #     }
    # ]

    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are an AI assistant that evaluates a visual navigation agent's performance.\n"
                "You will be given:\n"
                "- The Context: A image of the agent's current observation.\n"
                "- The Goal: A task description that the agent aims to achieve.\n"
                "- The Agent's Reasoning: The agent's analysis on the observation and reasoning on how to achieve the task.\n"
                "- The Agent's Actions: A list of actions with corresponding observation images at each step. \n"
                "Your task is to analyze the agent's actions and judge its effectiveness. Provide your response as a single JSON object with the following two keys:\n"
                "\"ANALYSIS\": An analysis of what happened during action execution, and judge on the outcome based on the given task.\n"
                "\"VERDICT\": Your judgment on whether the agent successfully accomplished the task. Must be one of three strings: \"SUCCESS\", \"FAILURE\", or \"UNSURE\".\n"
                "\"REFLECTION\": If the verdiction is failure, reflect on what was wrong with the reasoning process or action planning.\n"
                "Ensure your response is a valid JSON object with no additional text or formatting.\n"},
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"The navigation agent is given this current observation image and the following goal: '{task}'.\n"
                "The agent's trajectory is as follows:\n"},
            ] + [
                {"type": "image", "image": img} for img in images
            ] + [
                {"type": "text", "text": f"The agent's reasoning: {reasoning}\n"
                f"The agent's actions: {actions}\n"}
            ]
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
        return {"OBSERVATION": "", "REASONING": "", "ACTIONS": []} if images is None else {"VERDICT": "UNSURE", "EXPLANATION": "Failed to parse JSON response."}


if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser(description="Visualize subtask dataset")
    # parser.add_argument("--dataset_root", type=str, required=True, help="Path to the subtask dataset root directory")
    # parser.add_argument("--output_dir", type=str, required=True, help="Directory to save visualizations")
    # parser.add_argument("--max_trajs", type=int, default=None, help="Maximum number of trajectories to visualize")

    # args = parser.parse_args()

    # generate_reasoning_for_dataset(args.dataset_root, args.output_dir, args.max_trajs)

    episode_id = "1LXtFkjw3qL_129"
    output_dir = "outputs/self_reflection_test"
    generate_reasoning_for_episode(episode_id, output_dir)
