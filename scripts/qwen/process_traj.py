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
        messages, waypoints, (origin_step, current_step) = generate_message(images, positions, rotations, actions, start_step)
        image_inputs, output_text = inference(model, processor, messages)

        output_path = f"outputs/qwen/{data_path.split('/')[-1]}"
        os.makedirs(output_path, exist_ok=True)

        # Save output text
        with open(f"{output_path}/start_{start_step}_output.json", "w") as f:
            f.write(output_text[0])

        # Save waypoints
        with open(f"{output_path}/start_{start_step}_waypoints.json", "w") as f:
            json.dump([waypoint.tolist() for waypoint in waypoints], f)

        # Save image content
        try:
            plot_result(
                images=image_inputs, 
                actions=actions[origin_step:current_step+1], 
                waypoints=waypoints,
                output_text=output_text[0], 
                output_path=output_path,
                start_step=start_step
            )
        except Exception as e:
            print(f"Error: {e}")

        print(f"Start step: {start_step}")
        print(f"Actions: {actions[origin_step:current_step+1]}")
        print(f"Output: {output_text}")
        print("=====================================")


if __name__ == "__main__":
    model, processor = load_model()

    base_path = "/home/tangwenhao/Workspace/habitat/outputs/tutorials/split_test"
    for data_path in os.listdir(base_path):
        process_traj(f"{base_path}/{data_path}", model, processor)
        print(f"Finished processing {data_path}")
        print("=====================================")
        # break