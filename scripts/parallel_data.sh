#!/bin/bash

# scene_path="/data1/tangwenhao/datasets/vlnce/data_test/preview_newdata_reflection"
scene_path="/data1/tangwenhao/datasets/vlnce/data_raw_val"
scene_names=$(ls $scene_path)

parallel --jobs 11 --bar python scripts/run_dataset.py --task_type vlnce --scene_id {} ::: $scene_names
# parallel --jobs 11 --bar python scripts/run_dataset.py --task_type objectnav --scene_id {} ::: $scene_names