import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import List

from chat_wrapper import ChatGPT
from template import (
    INTRO_FULL,
    RESPONSE_TEMPALTE_FULL,
    FORMAT_ACTION,
)

TURN_ANGLE = 15 # by default

SYSTEM_PROMPT = INTRO_FULL + RESPONSE_TEMPALTE_FULL
GENERATION_PROMPT = "Single instruction examples:\n" + "\n".join(FORMAT_ACTION)

def process_waypoints(waypoints: np.ndarray, actions: np.ndarray):
    if waypoints.shape[0] == 1:
        return np.array([[0, 0]])
    positions = waypoints[:, [0, 2]]
    positions[:, 1] = -positions[:, 1]  # flip y-axis

    positions = positions - positions[0]  # relative positions

    heading = (positions[1] - positions[0]) / np.linalg.norm(positions[1] - positions[0])

    rot_matrix = np.array([
        [heading[0], heading[1]],
        [-heading[1], heading[0]]
    ])

    positions = np.dot(positions, rot_matrix.T) # rotate relative position to x-axis facing direction

    first_forward_idx = np.where(actions == 1)[0][0]
    turn_angle = ((actions[:first_forward_idx] == 2).sum() - 
                  (actions[:first_forward_idx] == 3).sum()) * TURN_ANGLE
    if turn_angle != 0:
        rotation_matrix = np.array([
            [np.cos(np.radians(turn_angle)), -np.sin(np.radians(turn_angle))],
            [np.sin(np.radians(turn_angle)), np.cos(np.radians(turn_angle))]
        ])
        positions = np.dot(positions, rotation_matrix.T)

    return positions

def plot_result(
        output_path: str,
        image1: np.ndarray, 
        image2: np.ndarray,
        waypoints: np.ndarray, 
        actions: np.ndarray,
        frames: List[int],
        reasoning: str, 
        instruction: str
    ):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))
    ax1.imshow(image1)
    ax1.set_title("Initial Front View")
    ax1.axis("off")

    ax2.imshow(image2)
    ax2.set_title("Final Front View")
    ax2.axis("off")

    ax3.plot(-waypoints[:, 1], waypoints[:, 0], "bo-", markersize=5)
    ax3.plot(-waypoints[0, 1], waypoints[0, 0], "ro", markersize=10)
    ax3.plot(-waypoints[-1, 1], waypoints[-1, 0], "go", markersize=10)
    ax3.set_title("Waypoints")
    ax3.set_xlim(-4, 4)
    ax3.set_ylim(-4, 4)
    ax3.set_box_aspect(aspect=1)

    plt.figtext(
        0.0,
        0.0,
        reasoning,
        wrap=True,
        horizontalalignment="left",
        verticalalignment="bottom",
        fontsize=10,
    )
    
    plt.suptitle(f"{instruction}\nActions: {actions}\nStarting Frame: {frames[0]}, Ending Frame: {frames[1]}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    model_name = "gpt-4o"
    # base_path = "/home/tangwenhao/Workspace/habitat/outputs/vlnce"
    # output_base_path = "/home/tangwenhao/Workspace/habitat/outputs/vlnce_subtask/full_4o"
    base_path = "/data1/tangwenhao/datasets/vlnce_split"
    output_base_path = "/data1/tangwenhao/datasets/vlnce_subtask"

    chat = ChatGPT(model_name=model_name, system_prompt=SYSTEM_PROMPT)

    total_tokens = 0

    scene_ids = os.listdir(base_path)
    scene_ids = [scene_id for scene_id in scene_ids if scene_id != "videos"]
    # scene_ids = ["b8cTxDM8gDG_7026"]
    max_num = 100
    for ii, scene_id in enumerate(scene_ids[:max_num]):
        scene_path = f"{base_path}/{scene_id}"
        output_path = f"{output_base_path}/{scene_id}"
        if os.path.exists(output_path):
            print(f"Scene {scene_id} already processed!")
            continue
        os.makedirs(output_path)
        print(f"Processing scene {scene_id}...")

        with open(f"{scene_path}/actions.json", "r") as f:
            data = json.load(f)
            instruction = data["instruction"]
            waypoints = np.array(data["waypoints"])
            actions = np.array(data["actions"])

        num_images = len(os.listdir(f"{scene_path}/images"))
        get_image_name = lambda step: f"step_{step}.png" if f"step_{step}.png" in os.listdir(f"{scene_path}/images") else f"step_{step}_ref.png"

        start_step = 0
        skip_step = 2
        while num_images // skip_step > 50:
            skip_step += 1
        reference_views = [
            np.array(Image.open(f"{scene_path}/images/{get_image_name(step)}")) for step in range(start_step, num_images, skip_step)
        ]

        user_prompt = f"Given overall instruction: {instruction}\n" + \
            GENERATION_PROMPT

        response, usage = chat.send_message(reference_views, user_prompt)
        total_tokens += usage.total_tokens

        print("Prompt: ", user_prompt)
        print("Response: ", response)
        print("Usage: ", usage)

        try:
            output = json.loads(response)
        except:
            print("Failed to parse output!")
            continue

        # output = {
        #     "reasoning": "1) The overall instruction involves turning right, navigating through the kitchen to reach the main foyer, and stopping near the front door. This can be split into three subtasks: a) Turn right and navigate through the kitchen, b) Move into the main foyer, c) Wait near the front door. 2) Based on the views, frames 1-12 show navigation through the kitchen, frames 13-24 show the transition and stop at the main foyer, and frames 25-36 show approaching and waiting near the front door. 3) Identified the respective starting and ending frames for each subtask based on these observations.",
        #     "instruction": [
        #         "Turn right and move along the pathway through the kitchen island. Stop when you reach the edge of the kitchen near the cabinets.",
        #         "Move forward into the main foyer. Stop when you reach the open area near the stairs and foyer furniture.",
        #         "Approach the front door and wait near the entrance."
        #     ],
        #     "frames": [
        #         [1, 12],
        #         [13, 24],
        #         [25, 36]
        #     ]
        # }

        
        reasoning = output["reasoning"]
        sub_instructions = output["instruction"]
        if output["frames"][-1][-1] > len(reference_views):
            output["frames"][-1][-1] = len(reference_views)
        sub_frames = (skip_step * (np.array(output["frames"]) - 1) + start_step).astype(int).tolist()
        assert len(sub_instructions) == len(sub_frames), \
            "Number of sub-instructions does not match number of sub-trajectories!"
        
        subtasks = []
        for i in range(len(sub_instructions)):
            start_step, end_step = sub_frames[i]
            waypoints_start_step = (actions[:start_step] == 1).sum()
            waypoints_end_step = (actions[:end_step] == 1).sum() + 1
            waypoints_subtask = waypoints[waypoints_start_step:waypoints_end_step]
            actions_subtask = actions[start_step:end_step]
            positions = process_waypoints(waypoints_subtask, actions_subtask)

            subtask = {
                "subtask_id": i,
                "subtask_instruction": sub_instructions[i],
                "subtask_frame_ids": sub_frames[i],
                "world_positions": waypoints_subtask.tolist(),
                "relative_positions": positions.tolist(),
                "actions": actions[start_step:end_step].tolist(),
            }

            subtasks.append(subtask)

            try:
                plot_result(
                    f"{output_path}/subtask_{i}.png",
                    image1=reference_views[output["frames"][i][0] - 1],
                    image2=reference_views[output["frames"][i][1] - 1],
                    waypoints=positions,
                    actions=actions_subtask,
                    frames=sub_frames[i],
                    reasoning=reasoning,
                    instruction=sub_instructions[i]
                )
            except Exception as e:
                print(f"Failed to plot subtask {i}: {e}")

        with open(f"{output_path}/subtasks.json", "w") as f:
            json.dump(
                {
                    "scene_id": scene_id,
                    "overall_instruction": instruction,
                    "reasoning": reasoning,
                    "subtasks": subtasks
                }, f
            )

        with open("/home/tangwenhao/Workspace/habitat/total_tokens.txt", "a") as f:
            f.write(f"{ii} trajectory: {scene_id}, total tokens: {total_tokens}\n")

