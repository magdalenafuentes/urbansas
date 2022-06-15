"""
Copyright (c) 2022 Magdalena Fuentes, Bea Steers, Luca Bondi(Robert Bosch GmbH), Julia Wilkins
All rights reserved.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""
from pathlib import Path

import soundfile


def extract_audio_meta(audio_path: str or Path) -> dict:
    """
    Extract basic metadata from an audio file.

    Arguments:
        audio_path: path to audio file.
        
    Returns:
        dict: dictionary of audio metadata including:
            'sr': sample rate of audio
            'format': format of audio (i.e. .wav)
            'channels': number of channels in audio file.
            'duration': duration in seconds of audio file.

    """

    with soundfile.SoundFile(str(audio_path)) as f:
        audio_sr = f.samplerate
        audio_format = f.format
        audio_channels = f.channels
        audio_duration = float(f.frames) / f.samplerate

    out_dict = {
        'sr': int(audio_sr),
        'format': audio_format,
        'channels': audio_channels,
        'duration': audio_duration,
    }
    return out_dict
