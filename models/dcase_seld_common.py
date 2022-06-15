"""
Copyright (c) 2022 Magdalena Fuentes, Bea Steers, Luca Bondi(Robert Bosch GmbH), Julia Wilkins
All rights reserved.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, Reshape, Bidirectional, GRU, \
    TimeDistributed, Dense


def dcase_seld_core(*, spec_cnn, f_pool_size, t_pool_size, nb_cnn2d_filt, rnn_size, fnn_size, num_classes, dropout_rate,
                    num_out_time_frames, num_regions):
    """
    Core DCASE SELD model, with regions output

    This function is adapted from DCASE 2021: Sound Event Localization and Detection with Directional Interference
      (https://github.com/sharathadavanne/seld-dcase2021)
    Copyright (c) 2021 Tampere University and its licensors, licensed under the CC BY-NC license
    cf. 3rd-party-licenses.txt file in the root directory of this source tree.

    """
    for i, convCnt in enumerate(f_pool_size):
        spec_cnn = Conv2D(filters=nb_cnn2d_filt, kernel_size=(3, 3), padding='same', name=f'spec_conv2d_{i}')(spec_cnn)
        spec_cnn = BatchNormalization(name=f'spec_bn_{i}')(spec_cnn)
        spec_cnn = Activation('relu')(spec_cnn)
        spec_cnn = MaxPooling2D(pool_size=(t_pool_size[i], f_pool_size[i]))(spec_cnn)
        spec_cnn = Dropout(dropout_rate)(spec_cnn)
    # RNN
    spec_rnn = Reshape((num_out_time_frames, -1))(spec_cnn)
    for i, nb_rnn_filt in enumerate(rnn_size):
        spec_rnn = Bidirectional(
            GRU(nb_rnn_filt, activation='tanh', dropout=dropout_rate, return_sequences=True, name=f'rnn_gru_{i}'),
            merge_mode='mul', name=f'rnn_grub_{i}'
        )(spec_rnn)
    # FC - DOA
    doa = spec_rnn
    for nb_fnn_filt in fnn_size:
        doa = TimeDistributed(Dense(nb_fnn_filt, name=f'fnn_dense'), name=f'fnn_td_dense')(doa)
        doa = Dropout(dropout_rate)(doa)
    doa = TimeDistributed(Dense(num_classes * num_regions, name=f'out_dense'), name=f'out_td_dense')(doa)
    doa = Reshape((-1, num_classes, num_regions), name='regions_out')(doa)
    return doa