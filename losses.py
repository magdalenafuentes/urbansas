"""
Copyright (c) 2022 Magdalena Fuentes, Bea Steers, Luca Bondi(Robert Bosch GmbH), Julia Wilkins
All rights reserved.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""
import tensorflow as tf
from tensorflow.keras.losses import Loss


class WeightedBinaryCrossentropy(Loss):
    """
    Binary crossentropy where positive samples are weighted more/less than negative samples.
    Requires logits
    """

    def __init__(self, pos_weight: float, **kwargs):
        kwargs.setdefault('name', self.__class__.__name__)
        super().__init__(**kwargs)
        self._pos_weight = float(pos_weight)

    def get_config(self):
        config = super().get_config()
        config.update({
            'pos_weight': self._pos_weight,
        })
        return config

    def call(self, y_true, y_pred):
        return tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, self._pos_weight)
