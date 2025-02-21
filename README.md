# Habitat for OpenVLA

This repository contains the code for the Habitat for OpenVLA project. The project is a collaboration between the [OpenVLA](https://github.com/openvla/openvla) and [Habitat Lab](https://github.com/facebookresearch/habitat-lab). The goal of the project is to create a virtual learning environment for the construction industry.

## Installation

First install habitat-lab by following the instructions [here](habitat-lab/README.md).

Then install dependencies for openvla project by running the following commands:

```bash
# Install pytorch (CUDA 12.1)
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
# Install other dependencies
pip install -r requirements.txt
```