# SwinCVS: A Unified Approach to Classifying Critical View of Safety Structures in Laparoscopic Cholecystectomy

**Authors**:  
Franciszek Nowak, Evangelos B. Mazomenos, Brian Davidson, Matthew J. Clarkson  

--- 

## Overview

Welcome. This repository provides code necessary for reproduction of the SwinCVS publication. The work proposes a SwinV2+LSTM based architecture called SwinCVS, to classify three Critical View of Safety (CVS) criteria from an open access Endoscapes2023 dataset.  

## Implemented models

- **SwinV2 Backbone**: Pure SwinV2 backbone. Can be run on random weights or initialised using provided ImageNet weights.
- **SwinCSV (E2E, with multiclassifier)**: SwinCVS with end-to-end training and multiclassifier. Backbone weights initialised on ImageNet.
- **SwinCSV (E2E, without multiclassifier)**: SwinCVS with end-to-end training, but without multiclassifier. Backbone weights initialised on ImageNet.
- **SwinCSV (Frozen, without multiclassifier)**: SwinCVS where image encoding backbone is frozen. Suggested backbone weights pretrained on Endoscapes.

## Installation

- Clone this repository
- Install dependencies from requirements.txt

## Usage

Script is run by executing SwinCVS.py from the root of the repository. Specific model training parameters are set within config/SwinCVS_config.yaml. Settings that specify model selection are:
- MODEL.LSTM: False - just SwinV2 backbone training, True - SwinCVS = SwinV2 with LSTM
- MODEL.E2E: False - backbone weights frozen, True - End-to-end training
- MODEL.MULTICLASSIFIER: False - does not add an additional classifier after backbone, True - adds a classifier after backbone, before SLTM 
- MODEL.INFERENCE: False - allows for training, True - skips all training, performs only testing on provided weihts
- BACKBONE.PRETRAINED: 'str' - which backbone weights to load, imagenet or endoscapes

## Code Release

**Note**: The code for this project will be released soon following the publication of our paper. Stay tuned for updates!

## Citation

If you use this work in your research, please cite our paper:
(details soon)

