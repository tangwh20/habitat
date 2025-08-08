import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union
from PIL import Image
import cv2
import base64

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from template import (
    INTRO_OBS_AND_ACTION_STRING,
    GENERATION_GUIDE,
    RESPONSE_TEMPALTE_REASON_BY_STEPS,
    FORMAT_ACTION,
)

STEP_LENGTH = 8


def _prepare_prompts():    
    system_prompt = (
        # "System:\n" +
        INTRO_OBS_AND_ACTION_STRING + 
        GENERATION_GUIDE + 
        RESPONSE_TEMPALTE_REASON_BY_STEPS
    )
    generation_prompt = "Examples:\n" + "\n".join(FORMAT_ACTION)
    request_prompt = "Please provide a brief natural language instruction to summarize the whole sequence of actions based on the given images."

    return system_prompt, generation_prompt, request_prompt

def _generate_image_content(image: Union[Image.Image, np.ndarray]):
    # convert image to BytesIO and encode in openai image format
    image_arr = np.array(image) if type(image) is not np.ndarray else image
    encoded_image = cv2.imencode(".jpg", image_arr)[1]
    base64_image = base64.b64encode(encoded_image.tobytes()).decode("utf-8")

    return dict(
        {
            "type": "image",
            "image": f"data:image/jpeg;base64,{base64_image}",
        },
    )


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


def load_model():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    return model, processor


# ========== Functions for one step ==========
def generate_message(
        images: List[np.ndarray], 
        positions: np.ndarray, 
        actions: np.ndarray, 
        start_step: int = 0
    ):
    system_prompt, generation_prompt, request_prompt = _prepare_prompts()

    origin_step = start_step
    while actions[origin_step] != 1:
        origin_step += 1

    get_xy = lambda pos: np.array([pos[0], -pos[2]])

    origin_xy = get_xy(positions[origin_step])
    origin_next_xy = get_xy(positions[origin_step + 1])
    origin_direction = (origin_next_xy - origin_xy) / np.linalg.norm(origin_next_xy - origin_xy)
    origin_transform_matrix = np.array([[origin_direction[0], origin_direction[1]], 
                                        [-origin_direction[1], origin_direction[0]]])
    
    get_local_xy = lambda pos: np.dot(origin_transform_matrix, get_xy(pos) - origin_xy)
    
    current_step = origin_step
    waypoints = [get_local_xy(positions[current_step])]
    while current_step < len(actions) and len(waypoints) < STEP_LENGTH:
        if actions[current_step] == 1:
            waypoints.append(get_local_xy(positions[current_step + 1]))
        current_step += 1
    waypoints = np.array(waypoints)

    # image_content = images[start_step:start_step+STEP_LENGTH]
    image_content = [_generate_image_content(images[origin_step])]
    # breakpoint()

    # actions_prompt = f"Actions the robot actually took: {actions[start_step:start_step+STEP_LENGTH]}\n"
    waypoints_prompt = f"Given list of actions: {waypoints.tolist()}\n"
    text_content_user = [
        {
            "type": "text",
            "text": '\n'.join([waypoints_prompt, generation_prompt])
        }
    ]
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": image_content + text_content_user,
        }
    ]

    return messages, waypoints, (origin_step, current_step)


def inference(model, processor, messages):    
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
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text


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
