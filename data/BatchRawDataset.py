"""
Copyright (c) 2022 Magdalena Fuentes, Bea Steers, Luca Bondi(Robert Bosch GmbH), Julia Wilkins
All rights reserved.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""
import json
import random
from pathlib import Path
from typing import List, Literal

import numpy as np
import scipy.interpolate
import tensorflow as tf
import tensorflow_io as tfio

from project_paths import get_cache_folder
from project_params import class_distinction


def is_number(x) -> bool:
    """
    Determine if x is an integer or a float
    
    Args:
        x:

    Returns: True if x is int or float, False otherwise

    """
    return isinstance(x, (int, float)) and not isinstance(x, bool)


class BatchRawDataset:
    """
    Batch dataset with regions output, azimuth only.

    Attributes:
        index_path: path to index file.
        sr: desired sample rate for audio.
        in_dur: duration of each audio file chunk in the batch.
        labels_period: frequency of video annotation labels (e.g. 0.5 = two labels/second)
        num_regions: number of horizontal regions to divide the image frames into.
        fov: camera field of view.
        batch_size: number of audio chunks per batch.
        train: boolean indicating if we're at the training stage.
        classes: list of object classes.
        point_sources: boolean indicating if we are using center-point or box-wise annotations.
        audio_filtered_labels: boolean indicating if we want to filter to only video annotations
            that coincide with an audio annotation.
        filter_confirmed: either bool or number: indicating if we want to filter data that isn't confirmed 
            by audio annotations, and if it's passes as a number, what percent of the annotations 
            should be filtered?
        folds: list of folds used to extract portions of the data from the index.
        fold_key: key used for dataset folds
        filter_nivs: boolean indicating whether to filter files with non-identifiable
            vehicle sound + empty video annotations ground truth.
        filter_offscreen: boolean indicating whether to discard files with any 
            "offscreen sound" audio annotations.
    """

    def __init__(self, *,
                 index_path: Path or str,
                 sr: float,
                 in_dur: float,
                 labels_period: float,
                 num_regions: int,
                 fov: float,
                 batch_size: int,
                 train: bool = False,
                 classes: List[str] = None,
                 point_sources: bool = True,
                 audio_filtered_labels: bool = False,
                 filter_confirmed: bool or int = False,
                 folds: List[int] or None = None,
                 fold_key: Literal['city','location_id'],
                 filter_nivs: bool = False,
                 filter_offscreen: bool = False,
                 weight_act: bool = False,
                 ):
        """Initializes BatchRawDataset class with above args."""
        print('Filtering nivs', filter_nivs)
        print('Filtering offscreen', filter_offscreen)
        print('Using video annot confirmed by audio (with some overlap)', filter_confirmed)
        print('Using only video annot that overlap with audio', audio_filtered_labels)

        # Load dataset index, containing GT
        with Path(index_path).open('r') as f:
            self.index = json.load(f)

        # Determine dataset name
        self.dataset = Path(index_path).stem

        # Filter with folds, if provided
        self.folds = folds
        if folds is not None:
            self.index = {k: v for k, v in self.index.items() if v[f'fold_{fold_key}'] in self.folds}

        # Parse arguments
        self.sr = float(sr)
        self.in_dur = float(in_dur)
        self.num_in_samples = int(np.floor(in_dur * sr))
        self.num_regions = int(num_regions)
        self.batch_size = int(batch_size)
        self.train = bool(train)
        self.labels_period = labels_period
        self.fov = float(fov)
        self.point_sources = bool(point_sources)
        self.audio_filtered_labels = bool(audio_filtered_labels)
        self.filter_confirmed = filter_confirmed
        self.filter_nivs = bool(filter_nivs)
        self.filter_offscreen = bool(filter_offscreen)

        # Here we suppose all files in a dataset have the same number of channels
        self.num_in_channels = self.index[list(self.index.keys())[0]]['channels']
        self.num_out_win = int(np.floor(self.in_dur / self.labels_period))

        # Determine the available classes
        self.classes = classes

        # Distinguish between classes
        if class_distinction:
            if self.classes is None:
                self.classes = set()
                for uid, file_dict in self.index.items():
                    for event in file_dict['events']:
                        self.classes.add(event['label'])
                # Discard audio label for off_screen_vehicle, it will not appear in ground truth
                # as audio is only used as a filter for video labels
                self.classes.discard('offscreen')
                self.classes.discard('-1')
                self.classes = sorted(self.classes)
        # Don't distinguish between vehicle classes
        else:
            self.classes = ['vehicle']

        # Store cache folder
        self.cache_folder = get_cache_folder(self.dataset, self.sr)

        # Precompute and store ground truth for each file.
        # At training we extract a random segment of in_dur duration from each file, shuffling files order
        # At validation/test, we extract all segments of in_dur duration from each file, with no shuffling
        self.gt = {
            uid: self.compute_file_gt(uid) for uid in self.index.keys()
        }

        # Prepare statistics for weighting
        self.weight_act = weight_act
        # Mapping between number of activations in GT and weight for the sample, for weight_act strategy
        self.act2weight = {}
        if self.weight_act:
            num_out_frames_per_win = int(self.in_dur / self.labels_period)
            gt_windows = np.stack([
                self.gt[uid][idx0:idx0 + num_out_frames_per_win] for uid in self.gt for idx0 in
                np.arange(0, len(self.gt[uid]) - num_out_frames_per_win + 1, num_out_frames_per_win)
            ])
            gt_act = np.sum(gt_windows, axis=(1, 2, 3))
            counts, edges = np.histogram(gt_act, bins=np.arange(np.prod(gt_windows.shape[1:]) + 2))
            for numact, count in zip(edges[:-1], counts):
                self.act2weight[numact] = 1 / (max(count, 1))

    def compute_file_gt(self, uid: str) -> np.ndarray:
        """
        Compute ground truth for a given file

        Args:
            uid: File uid, i.e. key of self.index

        Returns: (num_out_frames, num_classes, num_regions)
        """

        # Retrieve file information
        file_dict = self.index[uid]
        file_dur = file_dict['duration']
        file_out_tax = np.arange(0, file_dur, self.labels_period)

        num_out_win = len(file_out_tax)
        file_gt = np.zeros((num_out_win, self.num_classes, self.num_regions), np.float32)
        for event in file_dict['events']:

            # Ignore event if class is not relevant
            if event['label'] not in self.classes:
                continue
                
            if event.get('source') == 'audio':
                # Ignore audio labels in training
                continue
            # If filter_confirmed is passed as a number, that's a threshold for
            # what proportion of audio confirmations we want to require
            # If it's a bool, proceed with the rest of the logic
            if is_number(self.filter_confirmed):
                # If this event's confirmation threshold is less than the required threshold, skip it
                if event.get('amount_confirmed', 1) <= self.filter_confirmed:
                    continue
            # If filter_confirmed == True and this event as a whole is not confirmed, skip
            elif self.filter_confirmed and not event.get('confirmed', True):
                continue
            # Otherwise proceed (i.e. "keep this event")

            event_mask = (file_out_tax >= event['time'][0]) & (file_out_tax <= event['time'][-1])
            if np.sum(event_mask) == 0:
                # Ignore event if no intersection between event and
                continue

            event_win_idx0 = np.flatnonzero(event_mask)[0]
            event_tax = file_out_tax[event_mask]
            
            if class_distinction:
                label_idx = self.classes.index(event['label'])
            else:
                label_idx = 0

            # we have different versions of the azimuth values, saved with different suffixes
            # audio_filtered_labels means that we want azimuths to be filtered
            # to only those that coincide with an audio label
            key_sfx = '_filtered' if self.audio_filtered_labels else ''

            # skip events spanning only one frame
            if not len(event['time' + key_sfx]) >= 2:
                continue

            # bounding boxes take up a single region at the center of the box

            # skip events spanning only one frame
            if not len(event['time' + key_sfx]) >= 2:
                continue

            if self.point_sources:
                az_idx_left = az_idx_right = self.convert_azimuth_to_indexes(
                    event['time' + key_sfx], event['azimuth' + key_sfx], event_tax)
            # bounding boxes take up all regions from left to right of its box
            else:
                az_idx_left = self.convert_azimuth_to_indexes(
                    event['time' + key_sfx], event['azimuth_left' + key_sfx], event_tax)
                az_idx_right = self.convert_azimuth_to_indexes(
                    event['time' + key_sfx], event['azimuth_right' + key_sfx], event_tax)

            # fill the ground truth mask
            for win_idx, (i_az_l, i_az_r) in enumerate(zip(az_idx_left, az_idx_right)):
                file_gt[event_win_idx0 + win_idx, label_idx, i_az_l:i_az_r + 1] += 1

        # As multiple events of the same class can activate the same azimuth index, here we clip
        file_gt = np.clip(file_gt, 0, 1)

        return file_gt

    def retrieve_win_gt(self, uid: str, offset: float) -> np.ndarray:
        """
        Retrieve ground truth for a specific window, identified via uid and offset

        Args:
            uid: File uid
            offset: offset from file start [s]

        Returns: (num_out_frames, num_classes, num_regions)

        """

        win_out_tax = np.arange(self.num_out_win) / self.labels_period + offset

        win_gt = np.zeros((len(win_out_tax), self.num_classes, self.num_regions), np.float32)
        win_out_idx0 = int(offset / self.labels_period)
        win_gt_precomp = self.gt[uid][win_out_idx0:win_out_idx0 + len(win_out_tax)]
        win_gt[:len(win_gt_precomp)] = win_gt_precomp

        return win_gt

    def is_win_valid(self, uid: str, offset: float) -> bool:
        """
        Determine if a window is good or not, based on filter criteria

        Args:
            uid: File uid
            offset: offset from file start [s]

        Returns: True if window passes filters' selection, False otherwise
        """

        file_dict = self.index[uid]
        win_gt = self.retrieve_win_gt(uid=uid, offset=offset)

        if np.sum(win_gt) == 0:
            # Discard windows labeled as NIVS and empty gth (no confirmed annotations)
            if file_dict['non_identifiable_vehicle_sound'] and self.filter_nivs:

                return False

        #if np.sum(file_gt) == 0:
         #   return
            # Discard files labeled as NIVS and empty gth (no confirmed annotations)
            # if file_dict['non_identifiable_vehicle_sound'] and self.filter_nivs:
            #     return

        # Discard files with any off_screen_sound audio labels
        if 'offscreen' in np.unique([event['label'] for event in file_dict['events']
                                     if event['source'] == 'audio']).tolist() and self.filter_offscreen:
            return False

        return True

    def retrieve_win(self, uid: str, offset: float) -> dict:
        """
        Retrieve window data, indices, ground truth

        Args:
            uid: File uid
            offset: offset from file start [s]

        Returns: dictionary with attributes
                 audio: (num_in_channels, num_in_samples)
                 gt: (num_out_frames, num_classes, num_regions)
                 in_time: (num_in_samples)
                 out_time: (num_out_frames)
                 uid: uid of file, if window returned
        """

        # Load window ground truth
        win_gt = self.retrieve_win_gt(uid=uid, offset=offset)

        # Load resampled audio, pad, and check the sample rate just in case
        win_y = np.zeros((self.num_in_channels, self.num_in_samples))
        audio_path = self.cache_folder.joinpath(self.index[uid]['path'])

        # This is a lazy loader, (num_samples, num_channels)
        y_load = tfio.audio.AudioIOTensor(str(audio_path))
        file_sr = y_load.rate
        if int(file_sr) != int(self.sr):
            raise ValueError(f'Expected cache file {audio_path} to be at {self.sr}Hz, but it is at {file_sr}Hz')

        win_y_load = tf.transpose(y_load[int(offset*self.sr):int((offset+self.in_dur)*self.sr)])/(2**15)
        win_y[:, :win_y_load.shape[1]] = win_y_load

        # Time indexes, at input and output
        win_in_tax = np.arange(self.num_in_samples) / self.sr + offset
        win_out_tax = np.arange(self.num_out_win) * self.labels_period + offset
        win_out_tax[np.where(win_out_tax>len(y_load)/file_sr)] = np.nan

        win_weight = 1
        if self.weight_act:
            win_weight *= self.act2weight.get(int(np.sum(win_gt)), 1)

        assert win_y.shape[1] == len(win_in_tax)
        assert win_gt.shape[0] == len(win_out_tax)

        return {
            'audio': win_y,
            'gt': win_gt,
            'in_time': win_in_tax,
            'out_time': win_out_tax,
            'weight': win_weight,
            'uid': uid,
        }

    def convert_azimuth_to_indexes(self, time: np.ndarray, azimuth: np.ndarray,
                                   event_tax: np.ndarray = None) -> np.ndarray:
        """
        Given azimuth values (-fov/2, fov/2), convert them to gt mask indices.
        
        Args:
            time: the timestamps for each azimuth value.
            azimuth: the azimuth values.
            event_tax: upsampled timestamps that we want to interpolate to (see retrieve_file)

        Returns: 
            np.ndarray: ground truth mask indices with values in range [0,num_regions].
        """

        _time, _azimuth = np.asarray(time), np.asarray(azimuth)
        if event_tax is not None:
            _azimuth = scipy.interpolate.interp1d(
                _time, _azimuth, fill_value='extrapolate')(event_tax)
        # Turn azimuth into region index. Azimuth is natively in -180 and 180. Transform into the given fov
        idxs = np.round(
            (_azimuth + self.fov / 2) / self.fov * (self.num_regions - 1)
        ).astype(int)
        # Clip as datasets could have angles outside the fov
        idxs = np.clip(idxs, 0, self.num_regions - 1)
        return idxs

    def __iter__(self):
        """
        Iterator that generates batches of samples

        Yields: dict, gt
                dict with keys:
                    audio: (batch_size, num_in_channels, num_in_samples)
                    in_time: (batch_size, num_in_frames)
                    out_time: (batch_size, num_out_frames)
                    uid: (batch_size)
                gt: (batch_size, num_out_time_frames, num_classes, num_regions)

        """

        # Determine sequence of files and offsets. Each file is characterize by its uid
        uid_offset_list = []
        if self.train:
            # At training, randomly select as many windows as non-overlapping windows for each file

            for uid in self.index.keys():
                for _ in range(int(np.ceil(self.index[uid]['duration'] / self.in_dur))):
                    offset = np.random.uniform(0, self.index[uid]['duration'])
                    if self.is_win_valid(uid, offset):
                        uid_offset_list.append((uid, offset))
            random.shuffle(uid_offset_list)
        else:
            # At validation/testing, select all non-overlapping windows for all files
            for uid in self.index.keys():
                for offset in np.arange(0, self.index[uid]['duration'], self.in_dur):
                    uid_offset_list.append((uid, offset))

        while len(uid_offset_list):
            num_el_batch = 0
            batch_dict = {
                'audio': [],
                'gt': [],
                'weight': [],
                'in_time': [],
                'out_time': [],
                'uid': []
            }
            while (num_el_batch < self.batch_size) and len(uid_offset_list):
                uid, offset = uid_offset_list.pop(0)
                sample_dict = self.retrieve_win(uid, offset)
                for k, v in sample_dict.items():
                    batch_dict[k].append(v)
                num_el_batch += 1

            for k in batch_dict:
                batch_dict[k] = np.stack(batch_dict[k])

            # Keras 2.6 doesn't support yet dictionary for gt output, only for multiple inputs.
            batch_gt = batch_dict['gt']
            batch_weight = np.expand_dims(batch_dict['weight'],
                                          axis=(1, 2))  # Needed to broadcast correctly at training
            del batch_dict['gt']
            del batch_dict['weight']

            if self.train:
                yield batch_dict, batch_gt, batch_weight
            else:
                yield batch_dict, batch_gt

    def __call__(self):
        return iter(self)

    @property
    def num_classes(self) -> int:
        """
        Retrieve the number of classes

        Returns:

        """
        return len(self.classes)

    @property
    def audio_shape(self) -> tuple:
        """
        Features shapes
        (batch, num channels, num_in_samples, )

        Returns:

        """
        return None, self.num_in_channels, self.num_in_samples,

    @property
    def gt_shape(self) -> tuple:
        """
        Ground truth shape
        (batch, num out time frames, num features, num channels)

        Returns:

        """
        return None, self.num_out_win, self.num_classes, self.num_regions

    @property
    def output_signature(self) -> tuple:
        out_dict = ({
                        'audio': tf.TensorSpec(shape=self.audio_shape, dtype=tf.float32, name='feat'),
                        'in_time': tf.TensorSpec(shape=(None, self.num_in_samples), dtype=tf.float32, name='in_time'),
                        'out_time': tf.TensorSpec(shape=self.gt_shape[:2], dtype=tf.float32, name='out_time'),
                        'uid': tf.TensorSpec(shape=(None,), dtype=tf.int32, name='uid'),
                    },
                    tf.TensorSpec(shape=self.gt_shape, dtype=tf.float32, name='gt'),
        )

        if self.train:
            out_dict += (tf.TensorSpec(shape=(None, 1, 1), dtype=tf.float32, name='weight'),)

        return out_dict
