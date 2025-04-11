import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from chat_wrapper import ChatGPT
from template import (
    INTRO,
    RESPONSE_TEMPALTE,
    FORMAT_ACTION,
)

SYSTEM_PROMPT = INTRO + RESPONSE_TEMPALTE
GENERATION_PROMPT = "Single instruction examples:\n" + "\n".join(FORMAT_ACTION)

def process_waypoints(waypoints: np.ndarray):
    positions = waypoints[:, [0, 2]]
    positions[:, 1] = -positions[:, 1]  # flip y-axis

    positions = positions - positions[0]  # relative positions

    heading = (positions[1] - positions[0]) / np.linalg.norm(positions[1] - positions[0])

    rot_matrix = np.array([
        [heading[0], heading[1]],
        [-heading[1], heading[0]]
    ])

    positions = np.dot(positions, rot_matrix.T) # rotate relative position to x-axis facing direction

    return positions

def plot_result(
        output_path: str,
        image1: np.ndarray, 
        image2: np.ndarray,
        waypoints: np.ndarray, 
        actions: np.ndarray,
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
    
    plt.suptitle(f"{instruction}\nActions: {actions}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    model_name = "gpt-4o"
    base_path = "/home/tangwenhao/Workspace/habitat/outputs/vlnce"
    output_base_path = "/home/tangwenhao/Workspace/habitat/outputs/vlnce_subtask"

    chat = ChatGPT(model_name=model_name, system_prompt=SYSTEM_PROMPT)

    scene_ids = os.listdir(base_path)
    scene_ids = [scene_id for scene_id in scene_ids if scene_id != "videos"]
    for scene_id in scene_ids:
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
            reference_waypoint_steps = np.array(data["reference_waypoint_steps"])
            reference_action_steps = np.array(data["reference_action_steps"])

        reference_views = [
            np.array(Image.open(f"{scene_path}/images/step_{step}_ref.png")) for step in reference_action_steps
        ]

        user_prompt = f"Given overall instruction: {instruction}\n" + \
            f"Given the number of subtasks: {len(reference_waypoint_steps) - 1}\n" + \
            GENERATION_PROMPT

        response, usage = chat.send_message(reference_views, user_prompt)

        print("Prompt: ", user_prompt)
        print("Response: ", response)
        print("Usage: ", usage)

        try:
            output = json.loads(response)
        except:
            print("Failed to parse output!")
        
        reasoning = output["reasoning"]
        sub_instructions = output["instruction"]

        assert len(sub_instructions) == len(reference_waypoint_steps) - 1, \
            "Number of sub-instructions does not match number of sub-trajectories!"
        
        subtasks = []
        for i in range(len(reference_waypoint_steps) - 1):
            start_step = reference_waypoint_steps[i]
            end_step = reference_waypoint_steps[i + 1]
            waypoints_subtask = waypoints[start_step:end_step + 1]
            positions = process_waypoints(waypoints_subtask)

            subtask = {
                "subtask_id": i,
                "subtask_instruction": sub_instructions[i],
                "world_positions": waypoints_subtask.tolist(),
                "relative_positions": positions.tolist(),
                "actions": actions[reference_action_steps[i]:reference_action_steps[i + 1] + 1].tolist(),
            }

            subtasks.append(subtask)

            plot_result(
                f"{output_path}/subtask_{i}.png",
                image1=np.array(Image.open(f"{scene_path}/images/step_{reference_action_steps[i]}_ref.png")),
                image2=np.array(Image.open(f"{scene_path}/images/step_{reference_action_steps[i + 1]}_ref.png")),
                waypoints=positions,
                actions=actions[reference_action_steps[i]:reference_action_steps[i + 1] + 1],
                reasoning=reasoning,
                instruction=sub_instructions[i]
            )

        with open(f"{output_path}/subtasks.json", "w") as f:
            json.dump(
                {
                    "scene_id": scene_id,
                    "overall_instruction": instruction,
                    "subtasks": subtasks
                }, f
            )

