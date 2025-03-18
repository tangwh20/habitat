from typing import List, Union
import numpy as np
import json
import os
from PIL import Image
from matplotlib import pyplot as plt

from chat_wrapper import ChatGPT
from template import (
    INTRO_OBS_AND_ACTION_STRING,
    GENERATION_GUIDE,
    RESPONSE_TEMPALTE_REASON_BY_STEPS,
    FORMAT_ACTION,
)

SYSTEM_PROMPT = (
    INTRO_OBS_AND_ACTION_STRING
    + GENERATION_GUIDE
    + RESPONSE_TEMPALTE_REASON_BY_STEPS
)
GENERATION_PROMPT = "Examples:\n" + "\n".join(FORMAT_ACTION)

STEP_LENGTH = 8


def load_data(data_path: str):    
    actions_path = f"{data_path}/actions.json"
    with open(actions_path, "r") as f:
        data = json.load(f)
        actions = np.array(data["actions"])
        positions = np.array(data["positions"])
        rotations = np.array(data["rotations"])

    images = [
        np.array(Image.open(f"{data_path}/{i}.png"))
        for i in range(len(actions))
    ]

    return actions, positions, rotations, images

def gen_content(
    images: List[np.ndarray],
    positions: np.ndarray,
    start_step: int = 0,
):
    origin_step = start_step
    while actions[origin_step] != 1:
        origin_step += 1

    pos_xy = positions[:, [0, 2]]
    pos_xy[:, 1] = -pos_xy[:, 1] # flip y-axis

    origin_xy = pos_xy[origin_step]
    origin_next_xy = pos_xy[origin_step + 1]
    origin_face = (origin_next_xy - origin_xy) / np.linalg.norm(origin_next_xy - origin_xy)

    rot_mat = np.array([
        [origin_face[0], origin_face[1]],
        [-origin_face[1], origin_face[0]]
    ]) # rotate relative position to x-axis facing direction

    current_step = origin_step
    waypoints = [np.array([0, 0])]
    while current_step < len(actions) - 1 and len(waypoints) < STEP_LENGTH:
        current_step += 1
        if actions[current_step] == 1:
            current_pos = pos_xy[current_step]
            current_xy = rot_mat @ (current_pos - origin_xy)
            waypoints.append(current_xy)

    waypoints = np.array(waypoints)

    image = images[origin_step]
    user_prompt = (
        f"Given list of actions: {waypoints.tolist()}\n" + GENERATION_PROMPT
    )

    return image, waypoints, (origin_step, current_step)


def gen_instruction(
    chat: ChatGPT,
    images: Union[List[np.ndarray], np.ndarray],
    waypoints: np.ndarray
):
    user_prompt = (
        f"Given list of actions: {waypoints.tolist()}\n" + GENERATION_PROMPT
    )
    return chat.send_message(images, user_prompt)


def plot_result(
        output_path: str,
        image: np.ndarray, 
        waypoints: np.ndarray, 
        actions: np.ndarray,
        reasoning: str, 
        instruction: str
    ):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(image)
    ax1.set_title("Initial Front View")
    ax1.axis("off")

    ax2.plot(-waypoints[:, 1], waypoints[:, 0], "ro-")
    ax2.set_title("Waypoints")
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    ax2.set_box_aspect(aspect=1)

    plt.figtext(
        0.0,
        0.0,
        reasoning,
        wrap=True,
        horizontalalignment="left",
        verticalalignment="bottom",
        fontsize=6,
    )
    
    plt.suptitle(f"{instruction}\nActions: {actions}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    model_name = "gpt-4o" # "o3-mini" # 
    base_path = "/data1/tangwenhao/datasets/split"
    output_base_path = f"/home/tangwenhao/Workspace/habitat/outputs/{model_name}"

    max_traj_num = 100

    chat = ChatGPT(model_name=model_name, system_prompt=SYSTEM_PROMPT)

    traj_ids = os.listdir(base_path)
    for traj_id in traj_ids[:max_traj_num]:
        data_path = f"{base_path}/{traj_id}"
        output_path = f"{output_base_path}/{traj_id}"
        if os.path.exists(output_path):
            print(f"Trajectory {traj_id} already processed!")
            continue
        os.makedirs(output_path)
        actions, positions, rotations, images = load_data(data_path)

        for start_step in range(0, len(actions), STEP_LENGTH):
            if not (1 in actions[start_step:]):
                break
            image, waypoints, (origin_step, current_step) = gen_content(images, positions, start_step=start_step)
            output_text = gen_instruction(chat, image, waypoints)
            print(f"Actions: {actions[origin_step:current_step+1]}")
            print(f"Output: {output_text}")

            output = json.loads(output_text)
            reasoning = output["reasoning"]
            instruction = output["instruction"]

            with open(f"{output_path}/start_{start_step}.json", "w") as f:
                json.dump({
                    "image": image.tolist(),
                    "waypoints": waypoints.tolist(),
                    "actions": actions[origin_step:current_step+1].tolist(),
                    "reasoning": reasoning,
                    "instruction": instruction
                }, f)

            plot_result(
                f"{output_path}/start_{start_step}.png", 
                image=image,
                waypoints=waypoints,
                actions=actions[origin_step:current_step+1],
                reasoning=reasoning,
                instruction=instruction
            )
        
        print(f"Trajectory {traj_id} done!")
        print("=====================================")

