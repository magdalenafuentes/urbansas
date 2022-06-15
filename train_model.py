"""
Train NN model

Copyright (c) 2022 Magdalena Fuentes, Bea Steers, Luca Bondi(Robert Bosch GmbH), Julia Wilkins
All rights reserved.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""
import os

# ###*IMPORANT*: Have to do this line *before* importing tensorflow
os.environ['PYTHONHASHSEED'] = str(41)

import numpy as np
import random
import argparse
import json
import shutil
import inspect
from inspect import getmembers, isclass
from typing import List, Optional, Literal

import keras.callbacks
import tensorflow as tf
from pathlib import Path
from keras.metrics import AUC
import keras.losses

import losses
import utils
import index
import project_params
from data.BatchRawDataset import BatchRawDataset
from models.dcase_seld_raw import dcase_seld_regions
from project_paths import index_folder, logs_folder, checkpoints_folder


def main(*,
         train_dataset: str,
         val_dataset: str or None,
         train_folds: List[int] or None,
         val_folds: List[int] or None,
         fold_key: Literal['city', 'location_id'],
         classes: List[str] or None,
         model_name: str,
         loss: str,
         fl_gamma: float,
         monitor: str,
         num_regions: int,
         fov: float,
         config_name: str,
         max_epochs: int,
         patience: int,
         rlrop_patience: int,
         lr: float,
         batch_size: int,
         init: Optional[str],
         overwrite: bool,
         seed: int,
         trainable_mel: bool,
         sr: float,
         in_dur: float,
         labels_period: float,
         point_sources: bool,
         win_length: int,
         hop_length: int,
         n_mels: int,
         dcase_seld_params: dict,
         filter_nivs: bool,
         filter_offscreen: bool,
         filter_confirmed: bool,
         audio_filtered_labels: bool,
         weight_act: bool,
         optimizer: str,
         ):
    """
    Attributes:
         train_dataset: dataset path for training (str).
         val_dataset: str or None,
         train_folds: folds used for training.
         val_folds: folds used for validation.
         fold_key: key used for dataset folds
         classes: optional list of classes to train on
         model_name: str indicating model config.
         loss: loss function, see keras.losses.get
         fl_gamma: focal loss gamma
         monitor: monitor variable
         num_regions: number of horizontal regions to divide the image frames into.
         fov: camera field of view.
         config_name: config name that the training output will write to.
         max_epochs: max training epochs.
         patience: patience param.
         rlrop_patience: rlprop patience.
         lr: learning rate.
         batch_size: number of audio chunks per batch.
         init: path to pretrained model
         overwrite: bool indicating if we will overwrite training outputs.
         seed: seed for reproducibility.
         trainable_mel: bool,
         sr: desired sample rate for audio.
         in_dur: duration of each audio file chunk in the batch.
         labels_period: frequency of video annotation labels (e.g. 0.5 = two labels/second)
         point_sources: boolean indicating if we are using center-point or box-wise annotations.
         win_length: window length for stft,
         hop_length: hop length for stft.
         n_mels: number of mels.
         dcase_seld_params: dict
         filter_nivs: boolean indicating whether to filter files with non-identifiable
            vehicle sound + empty video annotations ground truth.
         filter_offscreen: boolean indicating whether to discard files with any
            "offscreen sound" audio annotations.
         filter_confirmed: either bool or number: indicating if we want to filter data that isn't confirmed
            by audio annotations, and if it's passes as a number, what percent of the annotations
            should be filtered
        audio_filtered_labels: boolean indicating if we want to filter to only video annotations
            that coincide with an audio annotation.
         weight_act: activate weighting based on activation density
         optimizer: optimizer

    """
    # Initialize rng
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

    # Parameters and folder
    main_args = inspect.getargvalues(inspect.currentframe())
    params = {k: main_args.locals[k] for k in main_args.args}

    log_dir = logs_folder.joinpath(config_name)
    checkpoint_dir = checkpoints_folder.joinpath(config_name)
    checkpoint_filepath = checkpoint_dir.joinpath('epoch{epoch:03d}.h5')
    csv_log_path = checkpoint_dir.joinpath('history.csv')
    params_path = checkpoint_dir.joinpath('params.json')

    if params_path.exists() and not overwrite:
        raise FileExistsError(f"Another training with config name '{config_name}' exists at: {checkpoint_dir}")

    # Clean previously saved files
    shutil.rmtree(log_dir, ignore_errors=True)
    shutil.rmtree(checkpoint_dir, ignore_errors=True)

    checkpoint_dir.mkdir(parents=True)
    log_dir.mkdir(parents=True)

    if model_name == 'dcase_seld_regions':

        # Load train dataset
        train_db = BatchRawDataset(index_path=index_folder.joinpath(f'{train_dataset}.json'),
                                   sr=sr,
                                   batch_size=batch_size,
                                   train=True,
                                   folds=train_folds,
                                   in_dur=in_dur,
                                   num_regions=num_regions,
                                   labels_period=labels_period,
                                   fov=fov,
                                   point_sources=point_sources,
                                   weight_act=weight_act,
                                   filter_nivs=filter_nivs,
                                   filter_offscreen=filter_offscreen,
                                   filter_confirmed=filter_confirmed,
                                   classes=classes, fold_key=fold_key,
                                   audio_filtered_labels=audio_filtered_labels
                                   )

        # Load val dataset
        val_db = BatchRawDataset(index_path=index_folder.joinpath(f'{val_dataset}.json'),
                                 sr=sr,
                                 batch_size=batch_size,
                                 classes=train_db.classes,
                                 folds=val_folds,
                                 in_dur=in_dur,
                                 num_regions=num_regions,
                                 labels_period=labels_period,
                                 fov=fov,
                                 point_sources=point_sources,
                                 filter_nivs=filter_nivs,
                                 filter_offscreen=filter_offscreen,
                                 filter_confirmed=filter_confirmed,
                                 fold_key=fold_key,
                                 audio_filtered_labels=audio_filtered_labels
                                 )

        model_params = dict(
            enable_gcc=True,
            trainable_mel=trainable_mel,
            num_regions=num_regions,
        )

        model = dcase_seld_regions(
            sr=sr,
            in_dur=in_dur,
            num_out_win=train_db.num_out_win,
            num_in_channels=train_db.num_in_channels,
            num_classes=train_db.num_classes,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            gcc_nc=n_mels,
            **model_params,
            **dcase_seld_params
        )
        params.update(model_params)
        params['classes'] = train_db.classes
    else:
        raise NotImplementedError(f'Unknown model: {model_name}')

    # Test run on datasets, to confirm things are working
    _ = next(iter(train_db))
    _ = next(iter(val_db))

    # Prepare tf Datasets
    tf_train_db = tf.data.Dataset.from_generator(
        generator=train_db,
        output_signature=train_db.output_signature
    )
    tf_val_db = tf.data.Dataset.from_generator(
        generator=val_db,
        output_signature=val_db.output_signature
    )

    # Compile model for training
    config_dict = {"from_logits": True}
    if loss == 'BinaryFocalCrossentropy':
        config_dict['gamma'] = fl_gamma
    loss_identifier = {"class_name": loss,
                       "config": config_dict}
    try:
        loss_fct = keras.losses.get(loss_identifier)
    except ValueError:
        # Manually defined loss functions
        if hasattr(losses, loss):
            loss_class = getattr(losses, loss)
            if loss.startswith('Weighted'):
                pos_weight = sum([v.size for v in train_db.gt.values()]) / \
                             sum([np.sum(v) for v in train_db.gt.values()])
                loss_fct = loss_class(pos_weight=pos_weight)
            else:
                loss_fct = loss_class()
        else:
            raise ValueError(f'Unknown loss function: {loss}')

    print('loss', loss)
    print(loss_fct)

    metrics = [
        AUC(from_logits=True, name='auc_roc', curve='ROC'),
        AUC(from_logits=True, name='auc_pr', curve='PR'),
    ]
    model.compile(optimizer=tf.keras.optimizers.get({'class_name': optimizer, 'config': {'learning_rate': lr}}),
                  loss=loss_fct,
                  metrics=metrics)

    # Load pre-trained model
    if init is not None:
        if Path(init).is_file():
            init = str(init)
        else:
            init = str(utils.get_weights_path(config_name=init, metric=monitor)[0])
        model.load_weights(init, by_name=True, skip_mismatch=True)

    # Prepare callbacks
    callbacks = [
        keras.callbacks.TensorBoard(log_dir=log_dir, profile_batch=0),
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.EarlyStopping(monitor=monitor, mode='min' if 'loss' in monitor else 'max',
                                      patience=patience, verbose=True),
        keras.callbacks.ReduceLROnPlateau(monitor=monitor, mode='min' if 'loss' in monitor else 'max',
                                          factor=0.1, patience=rlrop_patience, verbose=True),
        # Checkpoint everything
        #keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath.with_name('best_val_auc_pr.h5'),
        #                                monitor="val_auc_pr",
        #                                save_best_only=False),
        keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath.with_name('epoch_{epoch:04d}.h5')),

        keras.callbacks.CSVLogger(csv_log_path),
    ]

    # Write parameters
    with params_path.open('w') as f:
        json.dump(params, f, indent=2)

    # Fit
    model.fit(
        x=tf_train_db,
        epochs=max_epochs,
        callbacks=callbacks,
        validation_data=tf_val_db,
    )

    print('Completed')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset', type=str, choices=sorted([el[0] for el in getmembers(index, isclass)]),
                        default='Urbansas')
    parser.add_argument('--val_dataset', type=str, choices=sorted([el[0] for el in getmembers(index, isclass)]),
                        default='Urbansas')
    parser.add_argument('--train_folds', type=int, required=True, nargs='+')
    parser.add_argument('--val_folds', type=int, required=True, nargs='+')
    parser.add_argument('--fold_key', type=str, default=project_params.fold_key)
    parser.add_argument('--classes', nargs='+', type=str)
    parser.add_argument('--model_name', type=str, default='dcase_seld_regions')
    parser.add_argument('--loss', type=str, default=project_params.loss)
    parser.add_argument('--fl_gamma', type=float, default=2)
    parser.add_argument('--monitor', type=str, help='Quantity to monitor for ReduceLROnPlateau and EarlyStopping',
                        default='val_auc_pr')
    parser.add_argument('--num_regions', type=int, default=project_params.num_regions)
    parser.add_argument('--fov',
                        help='Maximum field of view, to avoid a lot of empty regions when dataset is limited, e.g. labels from video',
                        type=float, default=project_params.fov)
    parser.add_argument('--config_name', type=str, required=True)
    parser.add_argument('--init', type=str)
    parser.add_argument('--batch_size', type=int, default=project_params.batch_size)
    parser.add_argument('--max_epochs', type=int, default=project_params.max_epochs)
    parser.add_argument('--lr', help='Initial learning rate', type=float, default=project_params.lr)
    parser.add_argument('--patience', help='Stop training after patience epochs without validation loss improvements',
                        type=int, default=project_params.patience)
    parser.add_argument('--rlrop_patience',
                        help='Reduce learning rate after rlrop_patience epochs without validation loss improvements',
                        type=int, default=project_params.rlrop_patience)
    parser.add_argument('--seed', help='Random seed', type=int, default=project_params.seed)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--trainable_mel', action='store_true')
    parser.add_argument('--sr', type=int, default=project_params.sr)
    parser.add_argument('--in_dur', type=int, default=project_params.in_dur)
    parser.add_argument('--labels_period', type=float, default=project_params.labels_period)
    parser.add_argument('--point_sources', action='store_true')
    parser.add_argument('--win_length', type=int, default=project_params.win_length)
    parser.add_argument('--hop_length', type=int, default=project_params.hop_length)
    parser.add_argument('--n_mels', type=int, default=project_params.n_mels)
    parser.add_argument('--dcase_seld_params', type=int, default=project_params.dcase_seld_params)
    parser.add_argument('--filter_nivs', action='store_true', default=False)
    parser.add_argument('--filter_offscreen', action='store_true', default=False)
    parser.add_argument('--filter_confirmed', action='store_true', default=False)
    parser.add_argument('--audio_filtered_labels', action='store_true', default=False)
    parser.add_argument('--weight_act', action='store_true', default=False)
    parser.add_argument('--optimizer', type=str, default='adam')

    args = parser.parse_args()
    main(**vars(args))

