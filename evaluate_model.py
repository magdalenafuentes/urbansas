"""
Copyright (c) 2022 Magdalena Fuentes, Bea Steers, Luca Bondi(Robert Bosch GmbH), Julia Wilkins
All rights reserved.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""
import os
import time
import glob
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from eval import load_predictions, load_index, score_file, get_random_prediction
import project_paths
import project_params

np.random.seed(project_params.seed)


def main(predictions_path:str, index_path:str, results_path:str,
        fov:float, num_regions:int, labels_period:float, point_sources:bool,
         random_scores=False, file_duration:float=10):
    """
    Load ground truth and predictions and compute IOU and GIOU evaluation scores.

    Arguments:
        predictions_path: path to predictions (h5 file)
        index_path: path to index (.json file)
        results_path: path to write out results CSV.
        fov: field of view of camera in video.
        num_regions: number of horizontal regions to divide video into.
        labels_period: number of annotations per second.
        point_sources: True if model is pointwise, False if it boxwise.
        random_scores: if True random scores are computed according to point_sources.
        file_duration: length in seconds of video files.

    Returns:
        None, writes out a results_XX.csv with results.
    """
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    classes = sorted(['bus', 'car', 'motorbike', 'truck'])
    tau_range = np.arange(0.0, 0.5, 0.05)

    if point_sources or 'pointwise' in predictions_path:
        source_type = 'pointwise'
    else:
        source_type = 'boxwise'

    print(f'Evaluating with {source_type} gth sources, {num_regions} number of regions')


    if point_sources:
        source_type = 'pointwise'
    else:
        source_type = 'boxwise'

    print(f'Evaluating with {source_type} gth sources, {num_regions} number of regions')


    # load model predictions
    for pred_path in glob.glob(os.path.join(predictions_path, "*")):
        print(f'Loading predictions in {pred_path}')
        pred = load_predictions(pred_path)
        print(f'Predictions length {len(pred)}')

        if len(pred)<1:
            continue

        # load index (which has the ground truth)
        index = load_index(index_path)

        scoring_df = pd.DataFrame({})

        random_df = pd.DataFrame({})


        # evaluate all files present in the predictions
        errors = []
        for uid in tqdm(list(pred.keys())):

            for tau in tau_range:

                # scoring_dict[uid].append(curr_tau_scores)
                df_score_all = score_file(index, pred, uid, tau, source_type, class_list=classes, labels_period=labels_period, fov=fov,
                                      frame_type='all', num_regions=num_regions, file_duration=file_duration)
                df_score_active = score_file(index, pred, uid, tau, source_type, class_list=classes, labels_period=labels_period, fov=fov,
                                         frame_type='active', num_regions=num_regions, file_duration=file_duration)
                df_score_inactive = score_file(index, pred, uid, tau, source_type, class_list=classes, labels_period=labels_period, fov=fov,
                                           frame_type='inactive', num_regions=num_regions, file_duration=file_duration)

                if df_score_all is None:
                    errors.append([uid])

                scoring_df = pd.concat([scoring_df,df_score_all])
                scoring_df = pd.concat([scoring_df, df_score_active])
                scoring_df = pd.concat([scoring_df, df_score_inactive])


                if random_scores:

                    rand_pred = {uid:get_random_prediction(p['pred'].shape[0],
                                                                    p['pred'].shape[1], p['pred'].shape[2],
                                                                     point_sources=point_sources)
                                 for uid, p in pred.items()}

                    df_score_all = score_file(index, rand_pred, uid, tau, source_type, class_list=classes,
                                              labels_period=labels_period, fov=fov,
                                              frame_type='all', num_regions=num_regions, file_duration=file_duration)
                    df_score_active = score_file(index, rand_pred, uid, tau, source_type, class_list=classes,
                                                 labels_period=labels_period, fov=fov,
                                                 frame_type='active', num_regions=num_regions,
                                                 file_duration=file_duration)
                    df_score_inactive = score_file(index, rand_pred, uid, tau, source_type, class_list=classes,
                                                   labels_period=labels_period, fov=fov,
                                                   frame_type='inactive', num_regions=num_regions,
                                                   file_duration=file_duration)

                    random_df = pd.concat([random_df, df_score_all])
                    random_df = pd.concat([random_df, df_score_active])
                    random_df = pd.concat([random_df, df_score_inactive])

        datetime = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
        scoring_df.to_csv(os.path.join(results_path, f'results_{datetime}_{source_type}.csv'))

        print(scoring_df.groupby(['tau', 'frame_type', 'score']).mean())

        if random_scores:
            random_df.to_csv(os.path.join(results_path, f'results_{datetime}_random_{source_type}.csv'))
            print(random_df.groupby(['tau', 'frame_type', 'score']).mean())

        print(f'ERRORS {errors}')
        print(f'Number of ERRORS {len(errors)}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions_path', type=str, default=project_paths.predictions_folder)
    parser.add_argument('--index_path', type=str, default=project_paths.index_folder.joinpath('Urbansas.json'))
    parser.add_argument('--results_path', type=str, default=project_paths.results_folder)
    parser.add_argument('--fov', type=float, default=project_params.fov)
    parser.add_argument('--num_regions', type=int, default=project_params.num_regions)
    parser.add_argument('--labels_period', type=float, default=project_params.labels_period)
    parser.add_argument('--file_duration', type=int, default=10)
    parser.add_argument('--point_sources', action='store_true')
    parser.add_argument('--random_scores', action='store_true')


    args = parser.parse_args()

    main(**vars(args))
