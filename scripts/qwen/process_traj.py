import os

from utils import (
    STEP_LENGTH,
    load_data, 
    load_model, 
    generate_message, 
    inference, 
    plot_result
)


def process_traj(data_path: str, model, processor):
    actions, images = load_data(data_path)

    for start_step in range(0, len(actions) - STEP_LENGTH, STEP_LENGTH):
        messages = generate_message(images, actions, start_step)
        image_inputs, output_text = inference(model, processor, messages)

        output_path = f"outputs/{data_path.split('/')[-1]}"
        os.makedirs(output_path, exist_ok=True)

        # Save output text
        with open(f"{output_path}/start_{start_step}.json", "w") as f:
            f.write(output_text[0])

        # Save image content
        plot_result(
            images=image_inputs, 
            actions=actions[start_step:start_step+STEP_LENGTH], 
            output_text=output_text[0], 
            output_path=output_path,
            start_step=start_step
        )

        print(f"Start step: {start_step}")
        print(f"Actions: {actions[start_step:start_step+STEP_LENGTH]}")
        print(f"Output: {output_text}")
        print("=====================================")


if __name__ == "__main__":
    model, processor = load_model()

    base_path = "/home/tangwenhao/Workspace/habitat/outputs/tutorials/split_test"
    for data_path in os.listdir(base_path):
        process_traj(f"{base_path}/{data_path}", model, processor)
        print(f"Finished processing {data_path}")
        print("=====================================")