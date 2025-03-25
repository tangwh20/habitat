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
GENERATION_PROMPT = "Examples:\n" + "\n".join(FORMAT_ACTION)

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

    scene_id = "JeFG25nYj2p_7880"

    chat = ChatGPT(model_name=model_name, system_prompt=SYSTEM_PROMPT)

    scene_path = f"{base_path}/{scene_id}"
    output_path = f"{output_base_path}/{scene_id}"
    os.makedirs(output_path, exist_ok=True)

    with open(f"{scene_path}/actions.json", "r") as f:
        data = json.load(f)
        instruction = data["instruction"]
        waypoints = np.array(data["waypoints"])
        actions = np.array(data["actions"])
        reference_waypoint_steps = np.array(data["reference_waypoint_steps"])
        reference_action_steps = np.array(data["reference_action_steps"])

    for i in range(len(reference_waypoint_steps) - 1):
        if os.path.exists(f"{output_path}/subtask_{i}.json"):
            print(f"Subtask {i} already exists!")
            continue
        start_step = reference_waypoint_steps[i]
        end_step = reference_waypoint_steps[i + 1]
        waypoints_subtask = waypoints[start_step:end_step + 1]
        positions = process_waypoints(waypoints_subtask)

        start_view = np.array(Image.open(f"{scene_path}/images/step_{reference_action_steps[i]}_ref.png"))
        end_view = np.array(Image.open(f"{scene_path}/images/step_{reference_action_steps[i + 1]}_ref.png"))

        user_prompt = f"Given overall instruction: {instruction}\n" + \
            f"Given list of locations: {positions.tolist()}\n" + GENERATION_PROMPT

        response = chat.send_message([start_view, end_view], user_prompt)

        print("Response: ", response)
        print("Actions: ", actions[reference_action_steps[i]:reference_action_steps[i + 1] + 1])

        try:
            output = json.loads(response)
        except:
            print("Failed to parse output!")
            continue
        reasoning = output["reasoning"]
        sub_instruction = output["instruction"]

        with open(f"{output_path}/subtask_{i}.json", "w") as f:
            json.dump(
                {
                    "overall_instruction": instruction,
                    "waypoints": positions.tolist(),
                    "actions": actions[reference_action_steps[i]:reference_action_steps[i + 1] + 1].tolist(),
                    "reasoning": reasoning,
                    "instruction": sub_instruction
                }, f
            )

        # with open(f"{output_path}/subtask_{i}.json", "r") as f:
        #     output = json.load(f)
        #     reasoning = output["reasoning"]
        #     sub_instruction = output["instruction"]

        plot_result(
            f"{output_path}/subtask_{i}.png",
            image1=start_view,
            image2=end_view,
            waypoints=positions,
            actions=actions[reference_action_steps[i]:reference_action_steps[i + 1] + 1],
            reasoning=reasoning,
            instruction=sub_instruction
        )

        # break