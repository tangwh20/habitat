#!/bin/bash
# This script runs the task in parallel using multiple processes.

# 56 scenes to process
# Each scene will be processed in a separate process.
scene_names=(
    "17DRP5sb8fy"                      
    "1LXtFkjw3qL"                    
    "1pXnuDYAj8r"                    
    "29hnd4uzFmX"                    
    "5LpN3gDmAk7"                    
    "5q7pvUzZiYa"
    "759xd9YjKW5"
    "7y3sRwLe3Va"
    "82sE5b5pLXE"
    "8WUmhLawc2A"
    "aayBHfsNo7d"
    "ac26ZMwG7aT"
    "B6ByNegPMKs"
    "b8cTxDM8gDG"
    "cV4RVeZvu5T"
    "D7G3Y4RVNrH"
    "D7N2EKCX4Sj"
    "dhjEzFoUFzH"
    "E9uDoFAP3SH"
    "e9zR4mvMWw7"
    "EDJbREhghzL"
    "GdvgFV5R1Z5"
    "gZ6f7yhEvPG"
    "HxpKQynjfin"
    "i5noydFURQK"
    "JeFG25nYj2p"
    "JF19kD82Mey"
    "jh4fc5c5qoQ"
    "kEZ7cmS4wCh"
    "mJXqzFtmKg4"
    "p5wJjkQkbXX"
    "Pm6F8kyY3z2"
    "pRbA3pwrgk9"
    "PuKPg4mmafe"
    "PX4nDJXEHrG"
    "qoiz87JEwZ2"
    "r1Q1Z4BcV1o"
    "r47D5H71a5s"
    "rPc6DW4iMge"
    "s8pcmisQ38h"
    "S9hNv5qa7GM"
    "sKLMLpTHeUy"
    "sT4fr6TAbpF"
    "ULsKaCPVFJR"
    "uNb9QFRL6hY"
    "ur6pFq6Qu1A"
    "Uxmj2M2itWa"
    "V2XKFyX4ASd"
    "VFuaQ6m2Qom"
    "VLzqgDo317F"
    "VVfe2KiqLaN"
    "Vvot9Ly1tCj"
    "vyrNrziPKCB"
    "XcA2TqTSSAj"
    "YmJkqBEsHnH"
    "ZMojNkEp431"
)

parallel --jobs 56 \
    python scripts/run_rollout.py \
    --task_name objectnav \
    --split train \
    --scene_name {1} \
    --output_path /data1/tangwenhao/datasets/objectnav/data_raw \
    ::: "${scene_names[@]}"

# parallel --jobs 20 \
#     python scripts/run_rollout.py \
#     --task_name vlnce \
#     --split train \
#     --split_num {1} \
#     --output_path /data1/tangwenhao/datasets/vlnce/data_raw \
#     ::: {0..19} # Process splits 0 to 19 in parallel