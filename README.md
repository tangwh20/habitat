# Habitat for OpenVLA

This repository contains the code for the Habitat for OpenVLA project. The project is a collaboration between the [OpenVLA](https://github.com/openvla/openvla) and [Habitat Lab](https://github.com/facebookresearch/habitat-lab). The goal of the project is to create a virtual learning environment for the construction industry.

## Installation

### Install habitat-lab

First install habitat-lab by following the instructions [here](habitat-lab/README.md), or by running the following commands:

```bash
# We require python>=3.9 and cmake>=3.14
conda create -n habitat python=3.9 cmake=3.14.0
conda activate habitat

# Install habitat-sim
conda install habitat-sim withbullet -c conda-forge -c aihabitat

# Install habitat-lab
cd habitat-lab
pip install -e habitat-lab

# Install habitat-baselines
pip install -e habitat-baselines
```

### Install openvla

Then install openvla by following the instructions [here](openvla/README.md), or by running the following commands:

```bash
conda activate habitat

# Install openvla
cd openvla
pip install -e .

# Install other dependencies
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation
```

## Datasets

For running example script on habitat-lab, you can download the required datasets by running the following commands:

```bash
# Download the datasets
python -m habitat_sim.utils.datasets_download --uids <replica_dataset> --data-path data
```

where `<replica_dataset>` should be replaced with the dataset you want to download. The required datasets are:

 - `habitat_test_scenes`
 - `habitat_test_pointnav_dataset`
 - `rearrange_dataset_v2`
 - `rearrange_pick_dataset_v0`
 - `replica_cad_dataset`
 - `hab_fetch`
 - `ycb`

After downloading the datasets, you can run the example script by running the following command:

```bash
python scripts/example_topdown.py
```

This example script will output top-down views in the `output` directory.

## Usage

### Evaluate OpenVLA

We provide a script to evaluate OpenVLA on the skokloster-castle scene. To run the evaluation script, run `scripts/evaluate_openvla_example.py`:

```bash
python scripts/evaluate_openvla_example.py
```

If you want to evaluate OpenVLA on a customized scene, you can modify the settings in the `example/example_episode.json` file. Then run the following command:

```bash
python scripts/example/example_gzip.py
python scripts/example/example.py
```

This will output a video of the OpenVLA agent navigating the scene in the `output` directory.

### Extract step-by-step rgb images

We provide a script to extract step-by-step rgb images in the `scripts/example_split.py` file. To run the script, run the following command:

```bash
python scripts/example_split.py
```

This will output step-by-step rgb images in the `output` directory for the first few episodes. The number of episodes can be modified in the `scripts/example_split.py` file.

### Use GPT-4o/Qwen2.5-VL to generate instructions

We provide instruction generation script using Qwen2.5-VL in the `scripts/{openai, qwen}/process_traj.py` file. 

```bash
# To use GPT-4o
pip install openai
python scripts/openai/process_traj.py

# To use Qwen2.5-VL
pip install qwen-vl-utils[decord]
python scripts/qwen/process_traj.py
```

We also provide a script to evaluate the quality of the generated instructions, that is, we count the times each instruction appears in the dataset.

```bash
python scripts/count_instruction.py
```