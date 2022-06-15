"""
Project paths, system dependent

Copyright (c) 2022 Magdalena Fuentes, Bea Steers, Luca Bondi(Robert Bosch GmbH), Julia Wilkins
All rights reserved.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""

import os
from pathlib import Path

# Workspace folder, shared
workspace = Path(os.getenv('WORKSPACE',Path(__file__).parent))  # The work folder for artifacts

#Workspace subfolders
index_folder = workspace.joinpath('index')
features_folder = workspace.joinpath('feature')
logs_folder = workspace.joinpath('logs')
checkpoints_folder = workspace.joinpath('checkpoints')
predictions_folder = workspace.joinpath('predictions')
results_folder = workspace.joinpath('results')
data_folder = workspace.joinpath('data')

# Datasets root folders
#urbansas_root = Path(os.getenv('URBANSAS_ROOT', '../urbansas_dataset/')) # The dataset folder
urbansas_root = Path(os.getenv('URBANSAS_ROOT', '/scratch/mf3734/share/urbansas/dataset'))

# Dataset class name to root map
dataset_root_map = {
    'Urbansas': urbansas_root
}


def get_cache_folder(dataset:str, sr:int or float) -> Path:
    """Get cache folder

    Args:
        dataset: dataset class name
        sr: sampling rate [Hz]

    Returns:
        Path to cache folder
    """

    cache_folder = data_folder.joinpath(dataset, f'sr-{int(sr):d}')
    return cache_folder