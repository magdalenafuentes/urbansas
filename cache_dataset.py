"""
Cache a resampled version of the dataset

Copyright (c) 2022 Magdalena Fuentes, Bea Steers, Luca Bondi(Robert Bosch GmbH), Julia Wilkins
All rights reserved.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""
import numpy as np
import argparse
import json
from inspect import getmembers, isclass
from pathlib import Path
from tqdm.auto import tqdm
import librosa
from soundfile import SoundFile

import index
import project_params

from project_paths import index_folder, dataset_root_map, get_cache_folder


def check_integrity(y: np.ndarray, sr: float, file_dict: dict) -> bool:
    """Check integrity of audio file

    Args:
        y: audio file as read by librosa (channels, samples)
        sr: audio file sample rate
        file_dict: attributes for file from index

    Returns:
        bool: true if integrity ok, else otherwise
    """

    expected_channels = file_dict['channels']
    actual_channels = y.shape[0]

    expected_duration = file_dict['duration']
    actual_duration = y.shape[1]/sr

    return (expected_channels == actual_channels) and (np.abs(expected_duration-actual_duration) < 0.01)


def cache_dataset(*,
                  dataset: str,
                  sr: float,
                  ):
    """Cache dataset

    Args:
        dataset: dataset name
        sr: sample rate [Hz]

    """

    index_path = index_folder.joinpath(f'{dataset}.json')
    root = dataset_root_map[dataset]
    cache_folder = get_cache_folder(dataset,sr)
    print(f'Cache folder: {cache_folder}')

    # Load dataset index
    with Path(index_path).open('r') as f:
        index_dict = json.load(f)

    for uid, file_dict in tqdm(index_dict.items(), desc=f'Caching {dataset} at {sr}Hz'):
        resampled_audio_path = cache_folder.joinpath(
            file_dict['path']).with_suffix('.wav')
        original_audio_path = root.joinpath(file_dict['path'])

        if resampled_audio_path.exists():
            # Check audio file is valid
            file_y, file_sr = librosa.load(
                str(resampled_audio_path), sr=None, mono=False)
            if (file_sr != sr) or not check_integrity(file_y, file_sr, file_dict):
                tqdm.write(
                    f'Cached file {resampled_audio_path} is corrupted, recreating.')
                resampled_audio_path.unlink()

        if not resampled_audio_path.exists():
            # Resample and store audio
            file_y, file_sr = librosa.load(
                str(original_audio_path), mono=False, sr=sr, res_type='kaiser_fast')
            resampled_audio_path.parent.mkdir(exist_ok=True, parents=True)
            with SoundFile(str(resampled_audio_path), mode='w', samplerate=int(sr), channels=file_y.shape[0], subtype='PCM_16') as f:
                f.write(file_y.T)
            if not check_integrity(file_y, file_sr, file_dict):
                raise RuntimeError(
                    f'Error while generating cache for file: {resampled_audio_path}')

    print('Completed')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=sorted([el[0] for el in getmembers(index, isclass)]),
                        default='Urbansas')
    parser.add_argument('--sr', type=int, default=project_params.sr)
    args = parser.parse_args()
    cache_dataset(**vars(args))
