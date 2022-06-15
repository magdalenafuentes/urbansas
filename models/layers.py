"""
Copyright (c) 2022 Magdalena Fuentes, Bea Steers, Luca Bondi(Robert Bosch GmbH), Julia Wilkins
All rights reserved.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""
from typing import Iterable

import tensorflow as tf
from tensorflow.keras.layers import Layer


class STFT(Layer):

    def __init__(self, *, win_length: int = None, hop_length: int, n_fft: int, pad_end: bool = False,
                 **kwargs):
        """
        Extract STFT from time domain tensor

        Args:
            win_length: window length (optional)
            hop_length: hop length
            n_fft: fft size
            pad_end: pad end of input to not discard data
            kwargs: passed to tf.keras.layers.Layer constructor
        """
        kwargs.setdefault('name', self.__class__.__name__)
        super(STFT, self).__init__(**kwargs)

        self.hop_length = int(hop_length)
        self.n_fft = int(n_fft)
        self.win_length = int(win_length) if win_length is not None else self.n_fft
        self.n_fft = int(n_fft)
        self.pad_end = bool(pad_end)

    def get_config(self):
        config = dict(
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            pad_end=self.pad_end,
        )
        base_config = super(STFT, self).get_config()
        return {**base_config, **config}

    def call(self, x: tf.Tensor, **kwargs):
        """

        Args:
            x: time-domain float32 tensor (N,S) or (N,ch,S)
                N: batch size
                ch: number of channels, optional
                S: number of samples

        Returns: (N,T,F) or (N,ch,T,F) complex64 tensor
                N: batch size
                ch: number of channels, optional
                T: time frames
                F: nfft/2+1
        """

        num_channels = x.shape[1]
        stft = tf.stack([tf.signal.stft(signals=x[:, ch],
                                        frame_length=self.win_length,
                                        frame_step=self.hop_length,
                                        fft_length=self.n_fft,
                                        window_fn=tf.signal.hann_window,
                                        pad_end=self.pad_end
                                        ) for ch in range(num_channels)],
                        axis=1)
        return stft


class STFT2LogMel(Layer):
    def __init__(self, *,
                 sr: float, n_fft: int, n_mels: int, fmin: float = 0, fmax: float = None,
                 amin: float = 1e-10, ref_value: float = 1., top_db: float = 80., **kwargs):
        """
        Extract Log Mel Spectrogram from STFT

        Args:
            sr: sample rate
            n_fft: fft size
            n_mels: num mel bins
            fmin: min mel frequency
            fmax: max mel frequency
            amin: offset to prevent numerical errors in log
            ref_value: reference value for conversion to dB
            top_db: maximum dynamic in dB

        """
        kwargs.setdefault('name', self.__class__.__name__)
        super(STFT2LogMel, self).__init__(**kwargs)

        self.n_fft = int(n_fft)
        self.sr = float(sr)
        self.n_mels = int(n_mels)
        self.n_fft = int(n_fft)
        self.fmin = float(fmin)
        self.fmax = float(fmax) if fmax is not None else self.sr / 2
        self.top_db = float(top_db)
        self.amin = float(amin)
        self.ref_value = float(ref_value)

        self.mel_basis = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.n_mels, num_spectrogram_bins=n_fft // 2 + 1, sample_rate=sr,
            lower_edge_hertz=self.fmin, upper_edge_hertz=self.fmax, dtype=tf.dtypes.float32,
        )

    def get_config(self):
        config = dict(
            sr=self.sr,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            amin=self.amin,
            ref_value=self.ref_value,
            top_db=self.top_db,
        )
        base_config = super(STFT2LogMel, self).get_config()
        return {**base_config, **config}

    def call(self, x, **kwargs):
        """
        Args:
            x: (N,ch,T,F) complex64 or float32 tensor
                N: batch size
                ch: number of channels, optional
                T: time frames
                F = nfft/2+1

        Returns: (N,ch,T,M) float32 tensor
                N: number of signals in the batch
                ch: number of channels, optional
                T: time frames
                M: mel frequencies
        """
        psd = tf.square(tf.abs(x))
        mel = tf.tensordot(a=psd, b=self.mel_basis, axes=[[3], [0]])
        mel = tf.math.log(mel+tf.keras.backend.epsilon())

        return mel


class STFT2GCCPhat(Layer):
    def __init__(self, max_coeff: int = None, **kwargs):
        """
        Extract GCC-Phat from multi-channel STFT

        Args:
            max_coeff: maximum number of coefficients, first max_coeff//2 and last max_coeff//2
            kwargs: passed to tf.keras.layers.Layer constructor
        """
        kwargs.setdefault('name', self.__class__.__name__)
        super(STFT2GCCPhat, self).__init__(**kwargs, )
        self.max_coeff = max_coeff

    def get_config(self):
        config = dict(
            max_coeff=self.max_coeff,
        )
        base_config = super(STFT2GCCPhat, self).get_config()
        return {**base_config, **config}

    def call(self, inputs: tf.Tensor or Iterable[tf.Tensor, tf.Tensor], **kwargs):
        """

        Args:
            inputs: STFT [, mask]
                STFT: N,ch,T,F complex64 tensor
                    N: batch size
                    ch: number of channels
                    T: time frames
                    F = nfft/2+1
                mask: N,T,F float32 tensor
                    N: number of signals in the batch
                    T: time frames
                    S: max_coeff or nfft

        Returns: N,comb,T,S float32 tensor
                N: number of signals in the batch
                comb: number of channels combinations
                T: time frames
                S: max_coeff or nfft
        """
        num_channels = inputs.shape[1]

        if num_channels < 2:
            raise ValueError(f'GCC-Phat requires at least two input channels')

        out_list = []
        for ch1 in range(num_channels - 1):
            for ch2 in range(ch1 + 1, num_channels):
                x1 = inputs[:, ch1]
                x2 = inputs[:, ch2]
                xcc = tf.math.angle(x1 * tf.math.conj(x2))
                xcc = tf.exp(1j * tf.cast(xcc, tf.complex64))
                gcc_phat = tf.signal.irfft(xcc)
                if self.max_coeff is not None:
                    gcc_phat = tf.concat([gcc_phat[:, :, -self.max_coeff // 2:], gcc_phat[:, :, :self.max_coeff // 2]],
                                         axis=2)
                out_list.append(gcc_phat)

        return tf.stack(out_list, axis=1)


class SumNormalizationLayer(Layer):
    def __init__(self):
        super(SumNormalizationLayer, self).__init__()

    def call(self, inputs):
        sums = tf.reduce_sum(inputs, axis=-1)
        return inputs / sums[...,None]

