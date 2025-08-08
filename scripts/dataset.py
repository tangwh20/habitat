import os
import json
import re
import quaternion
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Union, Optional
from matplotlib import pyplot as plt

from chat_wrapper import ChatGPT, PromptCounter

from template import TEMPLATES, EXAMPLES, ACTION_MAP

MODEL_NAME = "gpt-4.1-mini"
ACTION_LENGTH = 8
BASE_PATH = "/home/tangwenhao/Workspace/habitat"


class Chats:
    split_chat: ChatGPT
    history_chat: ChatGPT
    instruction_chat: ChatGPT
    reasoning_chat: ChatGPT
    reflection_chat: ChatGPT


class Episode:
    def __init__(
        self,
        task_type: str,
        episode_id: str,
        chats: Chats
    ):
        assert task_type in ["vlnce", "objectnav"], "Invalid task type"
        self.task_type = task_type
        self.episode_id = episode_id
        self.chats = chats

        self.scene_id, self.episode_num = self.episode_id.split('_')

        self.counter = PromptCounter(model_name=MODEL_NAME)

        self.storage: Dict[str, List] = {}


    def load_data(self, input_data_type: str, data_path: str):
        """
        Load data from the specified path. Data should be organized as:
        - data_path/
            - scene_id/
                - episode_id/
                    - images/
                    - data.json
        """
        assert input_data_type in ["base", "trajectory", "instruction", "task"], "Invalid existing data type"
        if input_data_type == "base":
            self._load_base_data(data_path)
        else:
            with open(os.path.join(data_path, self.scene_id, self.episode_num, "data.json"), "r") as f:
                self.storage = json.load(f)

    
    def _load_base_data(self, data_path: str):
        self.base_path = data_path

        with open(os.path.join(self.base_path, self.scene_id, self.episode_num, "data.json"), "r") as f:
            data = json.load(f)
        if self.task_type == "vlnce":
            self.instruction = data["instruction"]
        elif self.task_type == "objectnav":
            self.object_goal = data["object_goal"]
            self.object_goal_id = data["object_goal_id"]
        self.actions = data["actions"]
        self.positions = data["positions"]
        self.rotations = data["rotations"]
        self.distances = data["distances"]
        self.collisions = data["collisions"]

        self.images = []
        images_path = os.path.join(data_path, self.scene_id, self.episode_num, "images")
        for step in range(len(os.listdir(images_path))):
            image_path = os.path.join(images_path, f"{step}.png")
            self.images.append(Image.open(image_path).convert("RGB"))

        num_steps = len(self.images)
        self.storage = {
            "position": [None] * num_steps,
            "yaw": [None] * num_steps,
            "instruction": [None] * num_steps,
            "task": [None] * num_steps,
            "history": [None] * num_steps,
            "reasoning": [None] * num_steps,
            "decision": [None] * num_steps,
            "reflection": [None] * num_steps
        }


    def save_data(self, output_data_type: str, output_path: str):
        """
        Save data to the specified path. Data will be organized as:
        - output_path/
            - scene_id/
                - episode_id/
                    - images/ # soft link to original images
                    - data.json
        """
        assert output_data_type in ["trajectory", "instruction", "task"], "Invalid output data type"
        output_dir = os.path.join(output_path, self.scene_id, self.episode_num)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "data.json"), "w") as f:
            if output_data_type == "trajectory":
                json.dump({
                    "position": self.storage["position"],
                    "yaw": self.storage["yaw"],
                }, f, indent=4)
            elif output_data_type == "instruction":
                json.dump({
                    "position": self.storage["position"],
                    "yaw": self.storage["yaw"],
                    "instruction": self.storage["instruction"],
                    "decision": self.storage["decision"],
                }, f, indent=4)
            elif output_data_type == "task":
                json.dump(self.storage, f, indent=4)

        images_src = os.path.join(self.base_path, self.scene_id, self.episode_num, "images")
        images_dst = os.path.join(output_dir, "images")
        if not os.path.exists(images_dst):
            os.symlink(images_src, images_dst)

    
    def visualize_data(self, visualize_path: str, step: Optional[int] = None):
        """
        Visualize the task data and save the plot to the specified path.
        """
        os.makedirs(visualize_path, exist_ok=True)

        num_steps = len(self.images)
        for k, v in self.storage.items():
            if len(v) != num_steps:
                raise ValueError(f"Storage key '{k}' has inconsistent length: {len(v)} vs {num_steps}")
            
        steps = range(num_steps) if step is None else [step]
        for i in steps:
            if any([v[i] is None for v in self.storage.values()]):
                continue

            position, yaw = self._calculate_pos_and_yaw(
                self.positions[i:i+ACTION_LENGTH],
                self.rotations[i:i+ACTION_LENGTH]
            )

            # Create a figure with specific size
            fig = plt.figure(figsize=(9, 9))
        
            # Create three subplots: top text, image, bottom text
            gs = fig.add_gridspec(3, 2, height_ratios=[2, 4, 3])
            
            # Top text area
            top_text = f"Step: {i}\n" + \
                f"Subtask: {self.storage['task'][i]}\n" + \
                f"Decision: {self.storage['decision'][i]}\n" + \
                f"Instruction: {self.storage['instruction'][i]}"
            ax_top = fig.add_subplot(gs[0, :])
            ax_top.axis('off')
            ax_top.text(0.1, 0.5, top_text, wrap=True, fontsize=14, va='center', ha='left')

            # Image area
            ax_img = fig.add_subplot(gs[1, 0])
            ax_img.imshow(self.images[i])
            ax_img.axis('off')

            # Trajectory area
            ax_traj = fig.add_subplot(gs[1, 1])
            ax_traj.plot(-position[:, 1], position[:, 0], marker='o', markersize=3, label="Trajectory")
            ax_traj.quiver(-position[:-1, 1], position[:-1, 0], 
                        np.cos(yaw[:-1] + np.pi / 2), np.sin(yaw[:-1] + np.pi / 2),
                        angles='xy', scale_units='xy', scale=5, color='r', label='Yaw Direction')
            ax_traj.set_xlim(-2, 2)
            ax_traj.set_ylim(-2, 2)
            ax_traj.set_aspect('equal', adjustable='box')
            ax_traj.set_title("Trajectory and Yaw Direction")
            ax_traj.legend(loc='lower right')
            ax_traj.grid()
            
            # Bottom text area
            bottom_text = f"History: {self.storage['history'][i]}\n" + \
                f"Reasoning: {self.storage['reasoning'][i]}\n" + \
                f"Reflection: {self.storage['reflection'][i]}"
            ax_bottom = fig.add_subplot(gs[2, :])
            ax_bottom.axis('off')
            ax_bottom.text(0., 0.5, bottom_text, wrap=True, fontsize=12, va='center', ha='left')

            # Save the figure
            output_path = os.path.join(visualize_path, self.scene_id, self.episode_num, f"{i}.png")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.tight_layout()
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            
                    



    # ================================================================================
    # ========================== Data Generation Methods =============================
    # ================================================================================
    def generate_trajectory(self):
        position = self.storage["position"]
        yaw = self.storage["yaw"]
        if None not in position and None not in yaw:
            return
        
        position, yaw = self._calculate_pos_and_yaw(self.positions, self.rotations)
        self.storage["position"] = position.tolist()
        self.storage["yaw"] = yaw.tolist()
    
    def generate_onestep_instruction(self, idx: int):
        positions = np.array(self.positions)
        rotations = np.array(self.rotations)

        positions = np.concatenate([positions, np.repeat(positions[-1:], ACTION_LENGTH, axis=0)], axis=0)
        rotations = np.concatenate([rotations, np.repeat(rotations[-1:], ACTION_LENGTH, axis=0)], axis=0)

        image = self.images[idx]

        relative_positions, relative_yaws = self._calculate_pos_and_yaw(
            positions[idx : idx + ACTION_LENGTH], rotations[idx : idx + ACTION_LENGTH]
        )
        user_prompt = f"Given list of positions: {relative_positions}\n" + \
                      f"Given list of yaw angles: {relative_yaws}" + EXAMPLES["instruction"]

        response, usage = self.chats.instruction_chat.send_message(image, user_prompt)
        self.counter.add_usage(usage)
        # print(f"Instruction chat usage: {self.counter.get_usage()}")

        output = json.loads(response)
        instruction_with_object = output["instruction_with_object"]
        instruction_without_object = output["instruction_without_object"]

        self.storage["instruction"][idx] = instruction_without_object
        self.storage["decision"][idx] = instruction_with_object

    def generate_task_history(self):
        num_total_images = len(self.images)
        # stride = max(2, num_total_images // 50 + 1)
        # image_index = np.arange(0, num_total_images, stride)

        if self.task_type == "vlnce":
            pass
        elif self.task_type == "objectnav":
            self.storage["task"] = [f"Find the {self.object_goal}"] * num_total_images

            for i in range(num_total_images):
                image = self.images[i]
                history = self.storage["history"][i-1] if i > 0 else ""
                action = ACTION_MAP[self.actions[i-1]] if i > 0 else "START"
                task = self.storage["task"][i]
                user_prompt = f"Given the previous history: {history}\n" + \
                              f"Given the last action: {action}\n" + \
                              f"Given the overall task: {task}\n"
                response, usage = self.chats.history_chat.send_message(image, user_prompt)
                self.counter.add_usage(usage)
                # print(f"History chat usage: {self.counter.get_usage()}")

                try:
                    output = json.loads(response)
                    self.storage["history"][i] = output["history"]
                except json.JSONDecodeError:
                    print(f"Error decoding JSON response at index {i}: {response}")
                    self.storage["history"][i] = response.strip()
                print(f"Finished generating history for image {i+1}/{num_total_images}")

    def generate_onestep_reasoning(self, idx: int):
        image = self.images[idx]
        task = self.storage["task"][idx]
        history = self.storage["history"][idx]
        decision = self.storage["decision"][idx]

        user_prompt = (
            f"Given high-level task: {task}. \n" + 
            f"Given navigation history: {history}. \n" +
            f"Given low-level movement instruction: {decision}. \n"
        )
        response, usage = self.chats.reasoning_chat.send_message(image, user_prompt)
        self.counter.add_usage(usage)
        # print(f"Reasoning chat usage: {self.counter.get_usage()}")
        
        try:
            output = json.loads(response)
            self.storage["reasoning"][idx] = output["reasoning"]
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON response at index {idx}: {response}")
            self.storage["reasoning"][idx] = response.strip()

    def generate_onestep_reflection(self, idx: int):
        image = self.images[idx]
        task = self.storage["task"][idx]
        history = self.storage["history"][idx]
        reasoning = self.storage["reasoning"][idx]
        decision = self.storage["decision"][idx]

        user_prompt = (
            f"Given high-level goal: {task}. \n" +
            f"Given navigation history: {history}. \n" +
            f"Given reasoning trace: {reasoning}. \n" +
            f"Given low-level movement instruction: {decision}. \n"
        )

        response, usage = self.chats.reflection_chat.send_message(image, user_prompt)
        self.counter.add_usage(usage)
        # print(f"Reflection chat usage: {self.counter.get_usage()}")

        try:
            output = json.loads(response)
            self.storage["reflection"][idx] = output
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON response at index {idx}: {response}")
            self.storage["reflection"][idx] = response.strip()


    @staticmethod
    def _calculate_pos_and_yaw(
        positions: List[List[float]], 
        rotations: List[List[float]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Home robot coordinate system: +z = up
        Habitat coordinate system: +y = up

        We map the coordinate systems by assuming the +x axis is shared
        x -> x
        y -> -z
        rz -> -ry
        """
        qs = quaternion.from_float_array(np.array(rotations))
        yaws = quaternion.as_rotation_vector(qs)[:, 1] + np.pi / 2.0

        positions = np.array(positions)
        positions = np.concatenate([positions[:, 0:1], -positions[:, 2:3]], axis=1)

        rotate_angle = -yaws[0]
        rotation_matrix = np.array([
            [np.cos(rotate_angle), -np.sin(rotate_angle)],
            [np.sin(rotate_angle), np.cos(rotate_angle)]
        ])
        
        relative_positions = positions - positions[0]
        relative_positions = relative_positions[:, :2] @ rotation_matrix.T
        relative_yaws = yaws - yaws[0]
        relative_yaws = (relative_yaws + np.pi) % (2 * np.pi) - np.pi

        return relative_positions, relative_yaws
    

if __name__ == "__main__":
    # Example usage
    chats = Chats()
    chats.split_chat = ChatGPT(model_name=MODEL_NAME, system_prompt=TEMPLATES["split"])
    chats.history_chat = ChatGPT(model_name=MODEL_NAME, system_prompt=TEMPLATES["history"])
    chats.instruction_chat = ChatGPT(model_name=MODEL_NAME, system_prompt=TEMPLATES["instruction"])
    chats.reasoning_chat = ChatGPT(model_name=MODEL_NAME, system_prompt=TEMPLATES["reasoning"])
    chats.reflection_chat = ChatGPT(model_name=MODEL_NAME, system_prompt=TEMPLATES["reflection"])


    episode = Episode(task_type="objectnav", episode_id="1LXtFkjw3qL_357", chats=chats)
    episode.load_data(input_data_type="base", data_path=os.path.join(BASE_PATH, "outputs/data/objectnav/data_raw"))
    episode.load_data(input_data_type="task", data_path=os.path.join(BASE_PATH, "outputs/data/objectnav/data_task"))
    # episode.generate_trajectory()
    # episode.generate_task_history()

    idx = 20
    episode.generate_onestep_instruction(idx=idx)
    episode.generate_onestep_reasoning(idx=idx)
    episode.generate_onestep_reflection(idx=idx)

    episode.save_data(output_data_type="task", output_path=os.path.join(BASE_PATH, "outputs/data/objectnav/data_task"))
    episode.visualize_data(step=idx, visualize_path=os.path.join(BASE_PATH, "outputs/data/objectnav/visualize"))

    print(f"Usage: {episode.counter.get_usage()}")
