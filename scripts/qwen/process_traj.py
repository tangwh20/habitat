import os
import json

from utils import (
    STEP_LENGTH,
    load_data, 
    load_model, 
    generate_message, 
    inference, 
    plot_result
)


def process_traj(data_path: str, model, processor):
    actions, positions, rotations, images = load_data(data_path)

    for start_step in range(0, len(actions) - STEP_LENGTH, STEP_LENGTH):
        messages, waypoints, (origin_step, current_step) = generate_message(images, positions, actions, start_step)
        output_text = inference(model, processor, messages)

        output_path = f"outputs/qwen_test/{data_path.split('/')[-1]}"
        os.makedirs(output_path, exist_ok=True)

        try:
            output_text = json.loads(output_text[0])
        except:
            print(f"Error in trajectory {data_path.split('/')[-1]} at start step {start_step}")
            with open(f"{output_path}/start_{start_step}_output.txt", "w") as f:
                f.write(output_text[0])
            continue

        with open(f"{output_path}/start_{start_step}.json", "w") as f:
            json.dump({
                "image": images[origin_step].tolist(),
                "waypoints": waypoints.tolist(),
                "actions": actions[origin_step:current_step+1].tolist(),
                "reasoning": output_text["reasoning"],
                "instruction": output_text["instruction"]
            }, f)

        # Save image content
        plot_result(
            f"{output_path}/start_{start_step}.png",
            image=images[origin_step],
            waypoints=waypoints,
            actions=actions[origin_step:current_step+1],
            reasoning=output_text["reasoning"],
            instruction=output_text["instruction"]
        )

        print(f"Start step: {start_step}")
        print(f"Actions: {actions[origin_step:current_step+1]}")
        print(f"Output: {output_text}")
        print("=====================================")


if __name__ == "__main__":
    model, processor = load_model()

    base_path = "/home/tangwenhao/Workspace/habitat/outputs/tutorials/split_test"
    for data_path in os.listdir(base_path):
        data_path = "dhjEzFoUFzH_19151"
        process_traj(f"{base_path}/{data_path}", model, processor)
        print(f"Finished processing {data_path}")
        print("=====================================")
        break