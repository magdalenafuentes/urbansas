"""
Copyright (c) 2022 Magdalena Fuentes, Bea Steers, Luca Bondi(Robert Bosch GmbH), Julia Wilkins
All rights reserved.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""

import cv2
import librosa
import numpy as np
import pandas as pd
import librosa.display
from pathlib import Path
from itertools import cycle
import matplotlib.pyplot as plt

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

# Viz utils

# For loading visual annotations
FPS = 2
COLUMNS = ['frame_id', 'track_id', 'x', 'y', 'w', 'h', 'not ignored', 'class_id', 'visibility']

# For visualizing bboxes
COLORS = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
]
INVISIBLE = 255, 255, 255
IS_ONE_BASED = False
LABELS_PERIOD = 0.5
CLASSES = ['bus', 'car', 'motorbike', 'offscreen', 'truck']


class Cycled(dict):
    def __init__(self, available, *a, **kw):
        self.available = available
        super().__init__(*a, **kw)

    def __missing__(self, key):
        value = self[key] = self.available[len(self) % len(self.available)]
        return value


def norm_col(color_tuple, max_col=255):
    return tuple(i / max_col for i in color_tuple)


colors_ = Cycled(COLORS)
shift = 2
COLOR_BARS = {cl: norm_col(co) for cl, co in zip(CLASSES, COLORS[shift:len(CLASSES) + shift])}


def draw_boxes_on_img(im, boxes, thickness=2, font_weight=None, font_size=0.6, colors=colors_):
    im = np.copy(im)
    for _, box in boxes.iterrows():
        color = colors[box.label] if box.visibility else INVISIBLE
        x, y, w, h = map(int, [box.x, box.y, box.w, box.h])
        label = '{}({})'.format(box.label, box.confidence) if 'confidence' in box else box.label
        # cv2.rectangle(image, start_point, end_point, color, thickness)
        cv2.rectangle(im, (x, y), (x + w, y + h), color, thickness)
        # cv2.putText(image, text, origin, font, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
        cv2.putText(im, label, (x + 3, y + h - 3), 0, font_size, color, font_weight or thickness)
    return im


def draw_detections(im, i, df, *a, **kw):
    '''Draws detections on a single frame.'''
    return draw_boxes_on_img(im, df[df.frame_id == i + IS_ONE_BASED], *a, **kw)


def draw_annot_on_spec(y, sr, file_audio_anot, file_video_annot, start=0, end=10):
    fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex='all')
    # fig.tight_layout(pad=3.0)

    # Plot waveform
    plt.sca(axs[0])
    plt.title("Original Two-channel Audio Waveform")
    librosa.display.waveshow(y[0, :], sr=sr, color='black')
    plt.xlabel('')
    plt.grid()
    plt.sca(axs[1])
    librosa.display.waveshow(y[1, :], sr=sr, color='black')
    plt.xlabel('')
    plt.grid()

    # Plot spectrogram
    plt.sca(axs[2])
    plt.title("Audio Spectrogram with Annotations")
    y_stft = librosa.stft(y[0])
    y_specgram = librosa.amplitude_to_db(np.abs(y_stft), ref=np.max)
    librosa.display.specshow(y_specgram, sr=sr, y_axis='mel', x_axis='time')

    # Add annotations
    bar_height = 0.3
    bar_bottom_it = cycle(2 ** np.arange(8, 15, bar_height + 0.1))

    # Add audio annotations
    plt.sca(axs[2])

    for _, event in file_audio_anot.iterrows():
        ev_start = event['start']
        ev_end = event['end']

        if ev_start < end and ev_end > start:
            ctr = (ev_start + ev_end) / 2
            width = ev_end - ev_start
            bottom = next(bar_bottom_it)
            height = bar_height * bottom
            plt.bar(x=ctr, width=width, bottom=bottom, height=height, alpha=0.5, color=COLOR_BARS[event['label']])
            plt.text(ev_start + 0.05, bottom + height / 2, f'[audio]:{event["label"]}',
                     color='w', va='center')

    # Add video annotations
    plt.sca(axs[2])
    events = file_video_annot.groupby(['track_id', 'label']).agg({'time': ['min', 'max']}).reset_index()
    for _, event in events.iterrows():
        ev_start = event['time']['min']
        ev_end = event['time']['max'] + LABELS_PERIOD

        if ev_start < end and ev_end > start:
            ctr = (ev_start + ev_end) / 2
            width = ev_end - ev_start
            bottom = next(bar_bottom_it)
            height = bar_height * bottom
            plt.bar(x=ctr, width=width, bottom=bottom, height=height, alpha=0.5, color=COLOR_BARS[event['label'][0]])
            plt.text(ev_start + 0.05, bottom + height / 2, f'[video] {event["label"][0]}',
                     color='w', va='center')
