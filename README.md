# MM-Fit - Getting Started
___
This repository contains starter code to facilitate further research using the MM-Fit dataset. The MM-Fit dataset is a 
substantial collection of inertial sensor data from smartphones, smartwatches and earbuds worn by participants while 
performing full-body workouts, and time-synchronised multi-viewpoint RGB-D video, with 2D and 3D pose estimates. To 
download the MM-Fit dataset, please visit the [MM-Fit website](https://mmfit.github.io/).

To get started with the MM-Fit dataset we provide an Exploratory Data Analysis Jupyter notebook, along with a number of helper functions which may be useful when working with the dataset.

We also provide sample code for training an end-to-end multimodal network for activity recognition using the MM-Fit dataset.
___
## Installation Guide
To install the project using Conda, follow these instructions.

First, clone the repository. Then download and extract the MM-Fit dataset from [here](https://mmfit.github.io/).

Conda environment setup:
```
conda env create -f environment.yml
conda activate mm-fit
```
To run the notebook:
```
jupyter lab
```
To train a multimodal network for activity recognition using MM-Fit run:
```
python train_multimodal_ar.py --data "mm-fit/" --num_classes 11 --lr 0.001 --epochs 25 --batch_size 128 --ae_layers 3 --ae_hidden_units 1000 --embedding_units 1000 --ae_dropout 0.0 --window_length 5 --window_stride 0.2 --layers 3 --hidden_units 100 --dropout 0.0
```
___
## Citation
To cite the MM-Fit dataset, please use the following reference (bibtex record provided below):

David Strömbäck, Sangxia Huang, and Valentin Radu. 2020. MM-Fit: Multimodal Deep Learning for Automatic Exercise Logging across Sensing Devices. Proc. ACM Interact. Mob. Wearable Ubiquitous Technol. 4, 4, Article 168 (December 2020), 22 pages. DOI:https://doi.org/10.1145/3432701
```
@article{mmfit_2020,
author = {Str\"{o}mb\"{a}ck, David and Huang, Sangxia and Radu, Valentin},
title = {MM-Fit: Multimodal Deep Learning for Automatic Exercise Logging across Sensing Devices},
year = {2020},
issue_date = {December 2020},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {4},
number = {4},
url = {https://doi.org/10.1145/3432701},
doi = {10.1145/3432701},
journal = {Proc. ACM Interact. Mob. Wearable Ubiquitous Technol.},
month = dec,
articleno = {168},
numpages = {22}
}
```