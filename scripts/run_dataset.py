import os
import random

from chat_wrapper import ChatGPT, PromptCounter
from template import TEMPLATES
from dataset import Episode, Chats

MODEL_NAME = "gpt-4.1-mini"
DATA_BASE_PATH = "/data1/tangwenhao/datasets"

chats = Chats()
chats.split_chat = ChatGPT(model_name=MODEL_NAME, system_prompt=TEMPLATES["split"])
chats.history_chat = ChatGPT(model_name=MODEL_NAME, system_prompt=TEMPLATES["history"])
chats.instruction_chat = ChatGPT(model_name=MODEL_NAME, system_prompt=TEMPLATES["instruction"])
chats.reasoning_chat = ChatGPT(model_name=MODEL_NAME, system_prompt=TEMPLATES["reasoning"])
chats.reflection_chat = ChatGPT(model_name=MODEL_NAME, system_prompt=TEMPLATES["reflection"])


def generate_common_data(task_type: str, episode_id: str):
    episode = Episode(task_type=task_type, episode_id=episode_id, chats=chats)
    episode.load_data(input_data_type="base", data_path=os.path.join(DATA_BASE_PATH, f"{task_type}/data_raw"))
    episode.generate_trajectory()
    episode.generate_task_history()
    episode.save_data(output_data_type="task", output_path=os.path.join(DATA_BASE_PATH, f"{task_type}/data_task"))

    return episode.counter


def generate_stepwise_data(task_type: str, episode_id: str, step: int):
    episode = Episode(task_type=task_type, episode_id=episode_id, chats=chats)
    episode.load_data(input_data_type="base", data_path=os.path.join(DATA_BASE_PATH, f"{task_type}/data_raw"))
    episode.load_data(input_data_type="task", data_path=os.path.join(DATA_BASE_PATH, f"{task_type}/data_task"))
    episode.generate_onestep_instruction(idx=step)
    episode.generate_onestep_reasoning(idx=step)
    episode.generate_onestep_reflection(idx=step)
    episode.save_data(output_data_type="task", output_path=os.path.join(DATA_BASE_PATH, f"{task_type}/data_task")) # TODO: Change to custom output path if needed

    return episode.counter


def visualize_stepwise_data(task_type: str, episode_id: str, step: int):
    episode = Episode(task_type=task_type, episode_id=episode_id, chats=chats)
    episode.load_data(input_data_type="base", data_path=os.path.join(DATA_BASE_PATH, f"{task_type}/data_raw"))
    episode.load_data(input_data_type="task", data_path=os.path.join(DATA_BASE_PATH, f"{task_type}/data_task"))
    episode.visualize_data(step=step, visualize_path=os.path.join(DATA_BASE_PATH, f"{task_type}/visualize"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process dataset")
    parser.add_argument("--task_type", type=str, required=True, choices=["vlnce", "objectnav"], help="Type of task")
    parser.add_argument("--scene_id", type=str, required=True, help="ID of the scene")
    args = parser.parse_args()

    # logging
    import time
    log_path = f"logs/{args.task_type}/{time.strftime('%m%d')}/{args.scene_id}.log"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_f = open(log_path, "a")
    log_f.write(f"Start processing scene: {args.scene_id} at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_f.flush()

    # scene_path = f"/data1/tangwenhao/datasets/vlnce/data_test/preview_newdata_subtask/{args.scene_id}"
    scene_path = os.path.join(DATA_BASE_PATH, f"{args.task_type}/data_task/{args.scene_id}")
    total_counter = PromptCounter(model_name=MODEL_NAME)
    for episode_num in os.listdir(scene_path):
        episode_id = f"{args.scene_id}_{episode_num}"
        try:
            # episode_counter = generate_common_data(args.task_type, episode_id) # Uncomment to generate common data

            # TODO: idx selection (currently random)
            num_steps = len(os.listdir(os.path.join(scene_path, episode_num, "images")))
            rand_idx = random.randint(0, num_steps - 1)
            episode_counter = generate_stepwise_data(args.task_type, episode_id, rand_idx)

            log_f.write(f"Episode {episode_id} usage: {episode_counter.get_usage()}\n")
            log_f.flush()
            total_counter.add_usage_from_counter(episode_counter)

        except Exception as e:
            log_f.write(f"Error processing episode {episode_id}: {e}\n")
            log_f.flush()


    log_f.write(f"Total usage for scene {args.scene_id}: {total_counter.get_usage()}\n")
    log_f.close()
