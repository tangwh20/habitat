import os
from tqdm import trange

from task import RolloutTask


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Habitat task rollout")
    parser.add_argument("--task_name", type=str, choices=["vlnce", "objectnav"], required=True, help="Task name")
    parser.add_argument("--split_num", type=int, default=None, help="Split number for vlnce task. If not provided, will process the entire dataset.")
    parser.add_argument("--scene_name", type=str, default=None, help="Scene name for objectnav task. If not provided, will process all scenes in the dataset.")
    parser.add_argument("--output_path", type=str, default=None, help="Output path for results")
    parser.add_argument("--save_video", action="store_true", help="Save video of the rollout")
    parser.add_argument("--remove_existing_output", action="store_true", help="Remove existing output directory")
    args = parser.parse_args()

    # Create error log file
    import time
    if args.split_num is not None:
        log_filepath = f"logs/{args.task_name}/parallel_{time.strftime('%m%d')}/{args.split_num}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log"
    elif args.scene_name is not None:
        log_filepath = f"logs/{args.task_name}/parallel_{time.strftime('%m%d')}/{args.scene_name}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log"
    else:
        log_filepath = f"logs/{args.task_name}/{time.strftime('%Y-%m-%d_%H-%M-%S')}.log"
    os.makedirs(os.path.dirname(log_filepath), exist_ok=True)
    log_f = open(log_filepath, "a")
    log_f.write(f"Start processing scene {args.scene_name} at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_f.flush()

    task = RolloutTask(**vars(args))

    num_episodes = task.dataset.num_episodes
    print(f"Number of episodes in the dataset: {num_episodes}") # 10,819 for vlnce, 2,632,422 for objectnav
    for i in trange(num_episodes, desc=f"Processing Scene {args.scene_name}:"):
        try:
            task.rollout()
        except Exception as e:
            print(f"Error in episode {i}: {e}")
            log_f.write(f"Error in episode {i}: {e}\n")
            log_f.flush()
            continue
        log_f.write(f"Finished episode {i}/{num_episodes}\n")
        log_f.flush()
    
    task.env.close()
    log_f.write(f"Finished processing scene {args.scene_name} at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_f.close()

    