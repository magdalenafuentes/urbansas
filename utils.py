"""
Copyright (c) 2022 Magdalena Fuentes, Bea Steers, Luca Bondi(Robert Bosch GmbH), Julia Wilkins
All rights reserved.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""
from pathlib import Path

import pandas as pd

from project_paths import checkpoints_folder


def get_weights_path(config_name: str, epoch: int = None, metric: str = None) -> (Path,int,str):
    """
    Get weights path for a specific configuration

    Args:
        config_name: configuration name, i.e. subfolder name of checkpoint_folder
        epoch: epoch number, as from history.csv
        metric: select epoch based on metric

    Returns:
        (Path to checkpoint, epoch, mode)
    """

    if epoch is None and metric is None:
        raise ValueError('Either epoch or metric must be provided')

    checkpoint_dir = checkpoints_folder.joinpath(config_name)
    history_path = checkpoint_dir.joinpath('history.csv')

    # Determine prediction epoch
    if epoch is None:

        history = pd.read_csv(history_path)
        if metric not in history:
            raise ValueError(f'{metric} not found in history. Valid values for this model are: {list(history.columns)}')

        mode = 'min' if 'loss' in metric else 'max'
        idx = history[metric].argmax() if mode == 'max' else history[metric].argmin()
        epoch = history['epoch'][idx]

        model_paths = list(checkpoint_dir.glob(f'epoch_{epoch:04d}.h5'))
        if len(model_paths):
            model_path = model_paths[-1]
        else:
            model_path = checkpoint_dir.joinpath(f'best_{metric}.h5')

    else:
        model_path = list(checkpoint_dir.glob(f'epoch_{epoch:04d}.h5'))[-1]
        mode = None

    return model_path, epoch, mode
