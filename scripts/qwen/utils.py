import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from template import (
    INTRO_8_OBS_AND_ACTION_STRING,
    GENERATION_GUIDE,
    RESPONSE_TEMPALTE_REASON_BY_STEPS,
    FORMAT_ACTION,
)

STEP_LENGTH = 8


def _prepare_prompts():    
    system_prompt = (
        "System:\n" +
        INTRO_8_OBS_AND_ACTION_STRING + 
        GENERATION_GUIDE + 
        RESPONSE_TEMPALTE_REASON_BY_STEPS
    )
    generation_prompt = "Examples:\n" + "\n".join(FORMAT_ACTION)
    request_prompt = "Please provide a brief natural language instruction to summarize the whole sequence of actions based on the given images."

    return system_prompt, generation_prompt, request_prompt


def load_data(data_path: str):    
    actions_path = f"{data_path}/actions.txt"
    with open(actions_path, "r") as f:
        actions = f.read().strip().split("\n")
        actions = [int(action) for action in actions]

    images = [
        {
            "type": "image",
            "image": f"{data_path}/{i}.png",
        }
        for i in range(len(actions))
    ]

    return actions, images


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
def generate_message(images: list, actions: list, start_step: int = 0):
    system_prompt, generation_prompt, request_prompt = _prepare_prompts()

    image_content = images[start_step:start_step+STEP_LENGTH]

    actions_prompt = f"Actions the robot actually took: {actions[start_step:start_step+STEP_LENGTH]}\n"
    text_content_user = [
        {
            "type": "text",
            "text": '\n'.join([system_prompt, generation_prompt, actions_prompt, request_prompt])
        }
    ]
    messages = [
        {
            "role": "user",
            "content": image_content + text_content_user,
        }
    ]

    return messages


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

    return image_inputs, output_text


def plot_result(images: List, actions: List, output_text: str, output_path: str = "output", start_step: int = 0):
    """
    Plot all images of the trajectory in one figure, and record the output text.

    Args:
        images (List): List of images of the trajectory.
        actions (List): List of actions of the trajectory.
        output_text (str): The output text of the trajectory.
        output_path (str): The path to save the output.
        start_step (int): The start step of the trajectory.
    """
    fig, axs = plt.subplots(2, STEP_LENGTH // 2, figsize=(8, 5))
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(np.array(images[i]))
        ax.title.set_text(f"Step {i}")
        ax.axis("off")
    output = json.loads(output_text)
    plt.suptitle(f"{output['instruction']}\nActions: {actions}")
    plt.tight_layout()

    plt.savefig(f"{output_path}/start_{start_step}.png")
    plt.close()
    
