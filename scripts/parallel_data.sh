#!/bin/bash

scene_path="/data1/tangwenhao/datasets/vlnce/data_test/preview_newdata_reflection"
scene_names=$(ls $scene_path)

parallel --jobs 60 python scripts/run_dataset.py --task_type vlnce --scene_id {} ::: $scene_names