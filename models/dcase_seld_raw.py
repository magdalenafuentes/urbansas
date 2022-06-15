"""
DCASE SELD raw input model, regions version

Copyright (c) 2022 Magdalena Fuentes, Bea Steers, Luca Bondi(Robert Bosch GmbH), Julia Wilkins
All rights reserved.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""
from typing import List

from models.layers import STFT, STFT2LogMel, STFT2GCCPhat
from keras import Input, Model
from keras.layers import BatchNormalization, Concatenate, Permute

from models.dcase_seld_common import dcase_seld_core


def dcase_seld_regions(*, sr: float,
                       in_dur: float, num_in_channels: int,
                       win_length: int, hop_length: int, n_mels: int,
                       mel_fmin: float = 0, mel_fmax: float = None, gcc_nc: int,
                       num_out_win: int, dropout_rate: float, nb_cnn2d_filt: int,
                       f_pool_size: List[int], t_pool_size: List[int],
                       rnn_size: List[int], fnn_size: List[int], num_classes: int, num_regions: int,
                       trainable_mel: bool = False,
                       enable_gcc: bool = False, ):
    """
    Get DCASE 2021 SELD raw input model. Adapted to be channel last, regions output

    """
    # model definition
    # name is important as data loader passes a dictionary
    raw_audio = Input(shape=(num_in_channels, int(in_dur*sr)), name='audio')

    # Prepare features layers
    stft = STFT(win_length=win_length, n_fft=win_length, hop_length=hop_length,pad_end=True)(raw_audio)

    # LogMel
    mel = STFT2LogMel(sr=sr, n_fft=win_length, n_mels=n_mels, fmin=mel_fmin, fmax=mel_fmax,
                      trainable=trainable_mel)(stft)
    mel = BatchNormalization(name='mel_bn')(mel)

    feat_list = [mel]

    # GCC Phat
    if enable_gcc:
        gcc_l = STFT2GCCPhat(max_coeff=gcc_nc)
        gcc = gcc_l(stft)
        gcc = BatchNormalization(name='gcc_bn')(gcc)
        feat_list.append(gcc)

    # Concatenate along channels and change to channel last
    feat = Concatenate(axis=1)(feat_list)
    feat = Permute((2, 3, 1))(feat)

    # CNN
    out = dcase_seld_core(spec_cnn=feat, f_pool_size=f_pool_size, t_pool_size=t_pool_size,
                          nb_cnn2d_filt=nb_cnn2d_filt, rnn_size=rnn_size, fnn_size=fnn_size, num_classes=num_classes,
                          dropout_rate=dropout_rate, num_out_time_frames=num_out_win, num_regions=num_regions)

    model = Model(inputs=raw_audio, outputs=out, name='dcase_seld_regions')
    model.summary()
    return model
