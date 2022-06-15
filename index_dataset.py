"""
Index one or more datasets

Copyright (c) 2022 Magdalena Fuentes, Bea Steers, Luca Bondi(Robert Bosch GmbH), Julia Wilkins
All rights reserved.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import argparse
from pathlib import Path
from inspect import getmembers, isclass

import index
import project_params
import project_paths


def main(*, datasets, overwrite:bool, urbansas_path:str, fov:float, frame_width:float, index_max_box_only:bool):

    """
    See audio_visual_dataset.py.
    
    Arguments:
        datasets: datasets used to create the index.
        urbansas_path: path to urbansas dataset, containing audio/ folder and annotations/ folders.
        fov: field of view.
        frame_width: width of video frame.
        index_max_box_only: boolean indicating if only the top N bounding boxes
            per class/frame will be used in the index.
        overwrite: set to overwrite exising index

    """

    print(f'Indexing {len(datasets)} datasets')
    audio_metadata_path = Path(os.path.join(urbansas_path, 'annotations', 'audio_annotations.csv'))
    video_metadata_path =  Path(os.path.join(urbansas_path, 'annotations', 'video_annotations.csv'))
    audio_folder = Path(os.path.join(urbansas_path, 'audio'))
    for dataset_name in datasets:
        dataset_class = getattr(index, dataset_name)(overwrite=overwrite)
        dataset_class.index(audio_metadata_path, video_metadata_path, audio_folder,
                            fov, frame_width, index_max_box_only)
    print('Completed')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, nargs='+', choices=sorted([el[0] for el in getmembers(index, isclass)]),
                        default=sorted([el[0] for el in getmembers(index, isclass)]))
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--urbansas_path', type=str, default=Path(project_paths.urbansas_root))
    parser.add_argument('--fov', type=int, default=project_params.fov)
    parser.add_argument('--frame_width', type=int, default=project_params.frame_width)
    parser.add_argument('--index_max_box_only', type=bool, default=project_params.index_max_box_only)

    args = parser.parse_args()

    main(**vars(args))
