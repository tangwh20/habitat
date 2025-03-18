import os
import json

from typing import List

def load_data(data_path: str):
    instructions = []
    for name in os.listdir(data_path):
        if not name.endswith(".json"):
            continue
        with open(f"{data_path}/{name}", "r") as f:
            data = json.load(f)
            instructions.append(data["instruction"])
    return instructions

def count_instruction(instructions: List[str]):
    instructions_count = []
    for instruction in list(set(instructions)):
        instructions_count.append((instruction, instructions.count(instruction)))
    instructions_count = sorted(instructions_count, key=lambda x: x[1], reverse=True)
    instructions_dict = {
        instruction: count
        for instruction, count in instructions_count
    }
    return instructions_dict

if __name__ == "__main__":
    base_path = "/home/tangwenhao/Workspace/habitat/outputs/gpt-4o"
    instructions = []
    for traj_id in os.listdir(base_path):
        data_path = f"{base_path}/{traj_id}"
        instructions.extend(load_data(data_path))
        print(f"Trajectory {traj_id} processed!")
    instructions_dict = {"total": len(instructions)}
    instructions_dict.update(count_instruction(instructions))

    with open(f"{base_path.split('/')[-1]}_count.json", "w") as f:
        json.dump(instructions_dict, f, indent=4)

