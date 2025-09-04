#!/bin/bash

scene_path=/home/tangwenhao/Workspace/habitat/data/datasets/objectnav/mp3d/v1/train/content
scene_names=$(ls $scene_path)

# get length of scene_names
parallel --jobs ${#scene_names[@]} --bar python scripts/split_objectnav_dataset.py --scene_filename {} ::: $scene_names