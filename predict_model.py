"""
Predict and evaluate a trained model

Copyright (c) 2022 Magdalena Fuentes, Bea Steers, Luca Bondi(Robert Bosch GmbH), Julia Wilkins
All rights reserved.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""
import argparse
import json
import sys
from inspect import getmembers, isclass
from typing import List, Literal

import h5py
import keras.models
import numpy as np
from scipy.special import expit
from tqdm.auto import tqdm
from losses import WeightedBinaryCrossentropy

import index
import project_params
import utils
from data.BatchRawDataset import BatchRawDataset
from models.layers import STFT, STFT2LogMel, STFT2GCCPhat
from project_paths import checkpoints_folder, predictions_folder, index_folder


def main(*,
         dataset: str or None,
         folds: List[int] or None,
         fold_key: Literal['city', 'location_id'] or None,
         config_name: str,
         epoch: int or None,
         metric: str or None,
         overwrite: str,
         loss: str,
         sr: int,
         batch_size: int,
         labels_period: float,
         point_sources: bool,
         filter_confirmed: bool,
         ):
    """
    Attributes:
        dataset: dataset to load audio+video annotations to predict on. if None uses val_dataset from train parameters
        folds: folds to predict on. If none, predict on val_folds if dataset == val_datasets, otherwise on all dataset
        fold_key: key used for dataset folds. If None, gathered from train parameters
        config_name: config name to load model from.
        epoch: epoch to load model from.
        metric: select epoch that maximizes/minimizes the given metric. If both epoch and metric are None
                uses monitor from train parameters
        overwrite: bool indicating if predictions will overwrite previous with same config
        loss: loss used in training
        sr: sampling rate of the audio signal used in training
        batch_size: batch size used in training
        labels_period: frequency of video annotation labels (e.g. 0.5 = two labels/second)
        point_sources: boolean indicating if we are using center-point or box-wise annotations.
        filter_confirmed: either bool or number: indicating if we want to filter data that isn't confirmed
            by audio annotations, and if it's passes as a number, what percent of the annotations
            should be filtered
    """
    checkpoint_dir = checkpoints_folder.joinpath(config_name)
    params_path = checkpoint_dir.joinpath('params.json')

    # Load training parameters
    with params_path.open() as f:
        train_params = json.load(f)

    dataset = train_params.get('val_dataset', dataset) if dataset is None else dataset
    fold_key = train_params.get('fold_key', fold_key) if fold_key is None else fold_key
    metric = train_params.get('monitor', metric) if metric is None and epoch is None else metric
    folds = train_params.get('val_folds', folds) if folds is None and dataset == train_params['val_dataset'] else folds

    # Determine prediction epoch
    model_path, epoch, _ = utils.get_weights_path(config_name=config_name, epoch=epoch, metric=metric)

    print(f'Prediction on {dataset}, folds {folds}, stratification: {fold_key}, epoch: {epoch}' +
          ('' if metric is None else f', metric: {metric}'))

    if folds is not None:
        predictions_file = predictions_folder.joinpath(config_name,
                                                       f'{dataset}-f{".".join(map(str, folds))}-epoch{epoch:03d}.h5')
    else:
        predictions_file = predictions_folder.joinpath(config_name, f'{dataset}-epoch{epoch:03d}.h5')

    # Load dataset
    db = BatchRawDataset(index_path=index_folder.joinpath(f'{dataset}.json'),
                         sr=sr,
                         batch_size=batch_size,
                         folds=folds,
                         in_dur=train_params['in_dur'],
                         classes=train_params['classes'],
                         num_regions=train_params['num_regions'],
                         labels_period=labels_period,
                         fov=train_params['fov'],
                         point_sources=point_sources,
                         filter_confirmed=filter_confirmed,
                         fold_key=fold_key)

    if overwrite or not predictions_file.exists():

        # Load pretrained model
        print(f'Loading model from: {model_path}')
        model_objects = {'STFT': STFT,
                        'STFT2LogMel': STFT2LogMel,
                        'STFT2GCCPhat': STFT2GCCPhat,
                        }
        if loss == 'WeightedBinaryCrossentropy':
            model_objects[loss] = getattr(sys.modules[__name__], loss)

        model = keras.models.load_model(model_path,
                                        custom_objects=model_objects)

        # Prediction loop
        uid_dict = {}
        for batch_dict, batch_gt in tqdm(iter(db), desc='Prediction'):
            batch_pred = model.predict(batch_dict)
            for idx in range(len(batch_pred)):
                uid = batch_dict['uid'][idx]
                if uid not in uid_dict:
                    uid_dict[uid] = {'time': [], 'pred': [], 'gt': []}
                res_uid = uid_dict[uid]
                res_uid['time'].append(batch_dict['out_time'][idx])
                res_uid['pred'].append(batch_pred[idx])
                res_uid['gt'].append(batch_gt[idx])

        # Concatenate, clean NoN timestamps, apply sigmoid to output
        for uid, res_uid in uid_dict.items():
            res_uid['time'] = np.concatenate(res_uid['time'])
            res_uid['pred'] = np.concatenate(res_uid['pred'])
            res_uid['gt'] = np.concatenate(res_uid['gt'])

            valid_mask = ~np.isnan(res_uid['time'])
            res_uid['pred'] = res_uid['pred'][valid_mask]
            res_uid['gt'] = res_uid['gt'][valid_mask]
            res_uid['time'] = res_uid['time'][valid_mask]

            res_uid['pred'] = expit(res_uid['pred'])

        # Store results
        print(f'Saving predictions at: {predictions_file}')
        predictions_file.parent.mkdir(exist_ok=True, parents=True)
        with h5py.File(predictions_file, 'w') as f:
            for uid, res_uid in uid_dict.items():
                f_uid = f.create_group(uid)
                f_uid['time'] = res_uid['time']
                f_uid['pred'] = res_uid['pred']
                f_uid['gt'] = res_uid['gt']

    else:
        print(f'Prediction already computed at: {predictions_file}')

    print('Completed')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=sorted([el[0] for el in getmembers(index, isclass)]),
                        required=False)
    parser.add_argument('--folds', type=int, required=False, nargs='+')
    parser.add_argument('--fold_key', type=str, required=False, )
    parser.add_argument('--config_name', type=str, required=True)
    parser.add_argument('--epoch', help='Epoch as from history.csv.', type=int)
    parser.add_argument('--metric', help='Select epoch with the best metric value', type=str)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--sr', type=int, default=project_params.sr)
    parser.add_argument('--batch_size', type=int, default=project_params.batch_size)
    parser.add_argument('--labels_period', type=float, default=project_params.labels_period)
    parser.add_argument('--point_sources', action='store_true')
    parser.add_argument('--filter_confirmed', action='store_true', default=False)
    parser.add_argument('--loss', type=str, default=project_params.loss)


    args = parser.parse_args()
    main(**vars(args))
