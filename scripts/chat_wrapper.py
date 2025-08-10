import base64
import cv2
import numpy as np
import io
import os
import json
import torch
from typing import Union, List

from abc import ABC, abstractmethod
from PIL import Image


with open(os.path.join(os.path.dirname(__file__), "_openai_api.json"), "r") as f:
    api_info = json.load(f)
os.environ["OPENAI_API_VERSION"] = api_info["openai_api_version"]
os.environ["AZURE_OPENAI_API_KEY"] = api_info["azure_openai_api_key"]
os.environ["AZURE_OPENAI_ENDPOINT"] = api_info["azure_openai_endpoint"]


MODEL_COST = {
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.4, 1.6),
    "gpt-4.1-nano": (0.1, 0.4),
    "gpt-5-mini": (0.25, 2.0),
    "Qwen/Qwen2.5-VL-7B-Instruct": (0.0, 0.0),
}


class PromptCounter:
    def __init__(self, model_name: str = "gpt-4o"):
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost_per_million = MODEL_COST[model_name]

    def add_usage(self, usage):
        self.total_tokens += usage.total_tokens
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
    
    def add_usage_from_counter(self, counter: 'PromptCounter'):
        self.total_tokens += counter.total_tokens
        self.prompt_tokens += counter.prompt_tokens
        self.completion_tokens += counter.completion_tokens

    def get_usage(self):
        prompt_cost = self.prompt_tokens / 1e6 * self.cost_per_million[0]
        completion_cost = self.completion_tokens / 1e6 * self.cost_per_million[1]
        return {
            'total_tokens': self.total_tokens,
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'cost': {
                'prompt_cost': prompt_cost,
                'completion_cost': completion_cost,
                'total_cost': prompt_cost + completion_cost,
            }
        }
    

class ChatWrapper(ABC):

    @abstractmethod
    def __init__(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def send_message(self, **args):
        raise NotImplementedError


class ChatGPT(ChatWrapper):

    def __init__(self, model_name: str = "gpt-4o", system_prompt: str = "") -> None:
        # lazy import
        from openai import AzureOpenAI

        self.chat = AzureOpenAI()
        self.model_name = model_name
        self.system_prompt = system_prompt

    def _generate_image_content(self, image: Union[Image.Image, np.ndarray]):
        image_arr = np.array(image) if type(image) is not np.ndarray else image
        encoded_image = cv2.imencode(".jpg", image_arr)[1]
        base64_image = base64.b64encode(encoded_image.tobytes()).decode("utf-8")

        return dict(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            },
        )

    def send_message(
            self,
            images: Union[Image.Image, np.ndarray, List[np.ndarray]],
            user_prompt: str,
            verbose=False,
    ):
        image_content = (
            [self._generate_image_content(m) for m in images]
            if isinstance(images, list)
            else [self._generate_image_content(images)]
        )

        completion = self.chat.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": image_content
                               + [
                                   {"type": "text", "text": user_prompt},
                               ],
                },
            ],
            response_format={"type": "json_object"},
        )
        response = completion.choices[0].message
        if verbose:
            print("Response: ", response.content)
        return response.content, completion.usage


class Qwen2_5Omni(ChatWrapper):
    def __init__(self, model_name: str = "Qwen/Qwen2.5-Omni-7B", device: str = "cuda" if torch.cuda.is_available() else "cpu") -> None:
        # lazy import
        from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
        
        self.device = device
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_name)
        self.model_name = model_name

    def _prepare_conversation(self, images: Union[Image.Image, np.ndarray, List[np.ndarray]], user_prompt: str) -> List[dict]:
        """Prepare the conversation format for the model."""
        # Convert single image to list for consistent processing
        if not isinstance(images, list):
            images = [images]
            
        # Convert numpy arrays to PIL Images if needed
        processed_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            processed_images.append(img)
            
        return [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ],
            },
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": img} for img in processed_images],
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

    def send_message(
            self,
            images: Union[Image.Image, np.ndarray, List[np.ndarray]],
            user_prompt: str,
            verbose=False,
    ):
        # Prepare conversation
        conversation = self._prepare_conversation(images, user_prompt)
        
        # Process inputs
        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            text_ids = self.model.generate(**inputs, max_new_tokens=256)
            response = self.processor.batch_decode(
                text_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
        
        if verbose:
            print("Response: ", response)
            
        # Create a usage object similar to OpenAI's format
        usage = type('Usage', (), {
            'total_tokens': len(response.split()),
            'prompt_tokens': len(user_prompt.split()),
            'completion_tokens': len(response.split()) - len(user_prompt.split())
        })
        
        return response, usage

class Qwen2_5VL(ChatWrapper):
    def __init__(
        self, 
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct", 
        device: str = "cuda" if torch.cuda.is_available() else "cpu", 
        system_prompt: str = ""
    ) -> None:
        # lazy import
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info
        
        self.device = device
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        #     "Qwen/Qwen2.5-VL-7B-Instruct",
        #     torch_dtype=torch.bfloat16,
        #     attn_implementation="flash_attention_2",
        #     device_map="auto",
        # )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model_name = model_name
        self.process_vision_info = process_vision_info
        self.system_prompt = system_prompt

    def _prepare_conversation(self, images: Union[Image.Image, np.ndarray, List[np.ndarray]], user_prompt: str) -> List[dict]:
        """Prepare the conversation format for the model."""
        # Convert single image to list for consistent processing
        if not isinstance(images, list):
            images = [images]
            
        # Convert numpy arrays to PIL Images if needed
        processed_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            processed_images.append(img)
        
        # Start with system message if system prompt is provided
        messages = []
        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}]
            })
            
        # Add user message with images and prompt
        messages.append({
            "role": "user",
            "content": [
                *[{"type": "image", "image": img} for img in processed_images],
                {"type": "text", "text": user_prompt},
            ],
        })
            
        return messages

    def send_message(
            self,
            images: Union[Image.Image, np.ndarray, List[np.ndarray]],
            user_prompt: str,
            verbose=False,
    ):
        # Prepare conversation
        messages = self._prepare_conversation(images, user_prompt)
        
        # Process inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
        
        if verbose:
            print("Response: ", response)

        # Replace \n in the response with a space
        response = response.replace("\n", " ")
            
        # Create a usage object similar to OpenAI's format
        usage = type('Usage', (), {
            'total_tokens': len(response.split()),
            'prompt_tokens': len(user_prompt.split()),
            'completion_tokens': len(response.split()) - len(user_prompt.split())
        })
        
        return response, usage

if __name__ == "__main__":
    # Example usage with Qwen2.5-VL
    vl = Qwen2_5VL(model_name="Qwen/Qwen2.5-VL-7B-Instruct")
    
    # Load example images
    image_paths = [
        "/home/tangwenhao/Workspace/habitat/outputs/tutorials/split_test/5LpN3gDmAk7_73870/0.png",
        "/home/tangwenhao/Workspace/habitat/outputs/tutorials/split_test/5LpN3gDmAk7_73870/1.png",
        "/home/tangwenhao/Workspace/habitat/outputs/tutorials/split_test/5LpN3gDmAk7_73870/2.png",
    ]
    images = [Image.open(image_path) for image_path in image_paths]
    
    # Example prompt
    prompt = "Please describe the scene in the images in detail."
    
    # Get response
    response, usage = vl.send_message(images=images, user_prompt=prompt, verbose=True)
