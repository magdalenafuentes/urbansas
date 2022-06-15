"""
Copyright (c) 2022 Magdalena Fuentes, Bea Steers, Luca Bondi(Robert Bosch GmbH), Julia Wilkins
All rights reserved.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""
import os
import glob
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import project_paths
import project_params
from eval import load_predictions, load_index, score_file, compute_file_gt

np.random.seed(project_params.seed)


def main(config_name:str, index_path:str,
        labels_period:float, point_sources:bool,
        file_duration:float=10, tau=0.05):
    """
    Load ground truth and predictions and compute IOU and GIOU evaluation scores.

    Arguments:
        config_name: path to predictions (h5 file)
        index_path: path to index (.json file)
        labels_period: number of annotations per second.
        point_sources: True if model is pointwise, False if it boxwise.
        file_duration: length in seconds of video files.

    Returns:
        None, writes out a results_XX.csv with results.
    """
    results_path = project_paths.results_folder
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    predictions_path = os.path.join(project_paths.predictions_folder, config_name)
    print(predictions_path)

    classes = sorted(['bus', 'car', 'motorbike', 'truck'])

    if point_sources:
        source_type = 'point_sources'
    else:
        source_type = 'box_sources'

    # load index (which has the ground truth)
    index = load_index(index_path)

    # load model predictions
    for pred_path in glob.glob(os.path.join(predictions_path, "*")):
        print(f'Loading predictions in {pred_path}')
        pred = load_predictions(pred_path)
        print(f'Predictions length {len(pred)}')

        name = pred_path.split('/')[-1]

        if len(pred)<1:
            continue

        scoring_df = pd.DataFrame({})

        # evaluate all files present in the predictions
        for uid in tqdm(list(pred.keys())):

            gth_uid = compute_file_gt(index, uid, classes=classes, point_sources=point_sources)
            pred_uid = pred[uid]['pred']

            # scoring_dict[uid].append(curr_tau_scores)
            df_score_all = score_file(gth_uid, pred_uid, uid, tau, class_list=classes, model_type=source_type,
                                      labels_period=labels_period, file_duration=file_duration, frame_type='all',
                                      name=name)
            df_score_active = score_file(gth_uid, pred_uid, uid, tau, class_list=classes, model_type=source_type,
                                      labels_period=labels_period, file_duration=file_duration, frame_type='active',
                                      name=name)
            df_score_inactive = score_file(gth_uid, pred_uid, uid, tau, class_list=classes, model_type=source_type,
                                      labels_period=labels_period, file_duration=file_duration, frame_type='inactive',
                                      name=name)

            scoring_df = pd.concat([scoring_df,df_score_all])
            scoring_df = pd.concat([scoring_df, df_score_active])
            scoring_df = pd.concat([scoring_df, df_score_inactive])

        outpath = os.path.join(results_path, f'results_{config_name}.csv')
        scoring_df.to_csv(outpath)
        print(f'Predictions csv saved in {outpath}')

        print(scoring_df.groupby(['tau', 'frame_type', 'score']).mean())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str, default=project_paths.predictions_folder)
    parser.add_argument('--index_path', type=str, default=project_paths.index_folder.joinpath('Urbansas.json'))
    parser.add_argument('--labels_period', type=float, default=project_params.labels_period)
    parser.add_argument('--file_duration', type=int, default=10)
    parser.add_argument('--point_sources', action='store_true')


    args = parser.parse_args()

    main(**vars(args))
