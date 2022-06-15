"""
Copyright (c) 2022 Magdalena Fuentes, Bea Steers, Luca Bondi(Robert Bosch GmbH), Julia Wilkins
All rights reserved.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""
import json
import warnings
from pathlib import Path
from statistics import mode

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

from index.utils import extract_audio_meta
from project_paths import index_folder, urbansas_root

def process_video_track(track_df, fov=None, frame_width=None) -> pd.Series:
    """
    Processes video annotations data by cleaning up typing and
    converting bounding box positions to azimuth angles.

    Arguments: 
        track_df: video annotations for one video.
        fov: field of view.
        frame_width: width of video frame.

    Returns: 
        pd.Series: processed video track.
    """
    
    x = track_df.x.values
    w = track_df.w.values

    track_ds = pd.Series({
        'label': mode(track_df.label),
        'time': track_df.time.to_list(),
        'azimuth': list(pos2angle(x + w/2, fov=fov, frame_width=frame_width)),
        'azimuth_left': list(pos2angle(x, fov=fov, frame_width=frame_width)),
        'azimuth_right': list(pos2angle(x + w, fov=fov, frame_width=frame_width)),
        'visibility': track_df.visibility.to_list(),
    })
    return track_ds

def filter_for_largest_box(track_df: pd.DataFrame, top_n: int=1) -> pd.DataFrame:
    """
    Filter the video annotations for one file down to only the largest bounding
    box annotation for each class per frame.

    Arguments:
        track_df: video annotations for one video.
        top_n: top "N" largest bounding boxes to filter to.

    Returns:
        pd.DataFrame: video annotations filtered to the top N largest 
            bounding box annotations per class and frame.
    """

    trimmed_video_annotations = pd.DataFrame(columns=track_df.columns)

    for t in np.arange(0, 10, 0.5):
        for c in list(set(track_df['label'])):
            curr_records = track_df.loc[((track_df['label'] == c) & (track_df['time'] == t))]
            box_areas = []
            if len(curr_records) > 0:
                for i, r in curr_records.iterrows():
                    box_areas.append(r['w'] * r['h'])
                curr_records['box_area'] = box_areas
                sorted_records = curr_records.sort_values(by='box_area', ascending=False)
                top_records = sorted_records.head(top_n)
                top_records = top_records.drop(columns='box_area')
                trimmed_video_annotations = trimmed_video_annotations.append(top_records, ignore_index=True) 
    return trimmed_video_annotations

def pos2angle(x:np.ndarray, fov:float, frame_width:float) -> np.ndarray:
    """
    Given a position in reference to an image coordinate, convert it to an angle.
    
    Arguments:
        x: position in reference to an image coordinate.
        fov: field of video.
        frame_width: width of video frame.
    
    Returns:
        azimuth angle mapped from the given position.
    """
    return (x / frame_width - 1 / 2) * fov


class AudioVisualDataset:
    """
    Base indexing class for Audio-Visual Datasets. Creates a JSON file
    after processing annotations files that will be used in model training 
    and testing.

    Attributes:
        root: Path to dataset root.
        index_path: Path to index file
        overwrite: Set to True to overwrite existing index
    """

    def __init__(self, *, root: str or Path, index_path: Path or str = None, overwrite: bool = False):
        """
        Initialize indexer for AudioVisualDatasets.
        """

        self.root_folder = Path(root)
        self.index_path = index_folder.joinpath(self.__class__.__name__ + '.json') if index_path is None else Path(
            index_path)
        self.overwrite = bool(overwrite)

    def __call__(self):
        return self.index()

    def index(self, *a, **kw):
        raise NotImplementedError

    def _index(self, *, audio_metadata_path:str, video_metadata_path:str, audio_folder:str or Path,
               fov:float, frame_width:float, index_max_box_only:bool) -> dict:
        """
        Indexing to create a JSON dict where each entry represents a file and is characterized by a unique id.
        Attributes:
          - 'path': path relative to dataset root
          - 'events': list of sound events, each with attributes
             + 'label': textual label for the sample
             + 'time': list of timestamps for time-varying information in the file
             + 'azimuth': list of azimuth angles, same length as time [deg].
                          azimuth = 0  is in front, azimuth -90 is right, azimuth 90 is left
             + 'elevation': list of elevation angles, same length as time [deg].
                          elevation = 0  is in front, elevation -90 is bottom, elevation 90 is top
             + 'start': beginning of sound, if no time-varying information
             + 'end': end of sound, if no time-varying information

        Arguments:
            audio_metadata_path: path to the audio metadata CSV.
            video_metadata_path: path to the video metadata CSV.
            audio_folder: folder with audio files.
            fov: field of view.
            frame_width: width of video frame.
            index_max_box_only: boolean indicating if only the top N bounding boxes
                per class/frame will be used in the index.

        Returns: 
            dict: index dictionary
        """

        if not self.overwrite and self.index_path.exists():
            try:
                with self.index_path.open() as f:
                    index = json.load(f)
                    return index
            except json.JSONDecodeError:
                print(f'Error while decoding index. Re-indexing')
                self.index_path.unlink()

        if not self.overwrite and self.index_path.exists():
            raise FileExistsError(f'Index file for {self.__class__.__name__} exists at: {self.index_path}')

        index = {}

        # Load audio and video metadata
        df_audio = pd.read_csv(audio_metadata_path)
        df_video = pd.read_csv(video_metadata_path)


        # Compile filenames list
        filenames_list = sorted(set(df_audio['filename'].to_list() +
                                    df_video['filename'].to_list()))

        # Iterate on metadata and files
        for uid, filename in enumerate(tqdm(filenames_list,
                                            desc=f'Indexing {self.__class__.__name__}',
                                            leave=False)):

            audio_files = list(Path(audio_folder).glob(filename + '.*'))
            if len(audio_files) == 0:
                raise FileExistsError(f'Audio file not found: {filename}\n'
                                      f'Audio folder: {audio_folder}\n'
                                      f'Please download the full dataset first')
            if len(audio_files) > 1:
                raise FileExistsError(f'Multiple audio files found for: {filename}\n'
                                      f'Audio folder: {audio_folder}\n'
                                      f'Please clean your dataset folder first')
            audio_path = audio_files[0]
            metadata = extract_audio_meta(audio_path)

            if metadata['channels'] != 2:
                warnings.warn(f'Expected stereo file, but {filename} has {metadata["channels"]} channels')

            df_audio_file = df_audio[df_audio['filename'] == filename]
            df_video_file = df_video[df_video['filename'] == filename]

            if len(df_video_file):
                city = df_video_file['city'].iloc[0]
                location_id = df_video_file['location_id'].iloc[0]
            else:
                stem = Path(df_audio_file.iloc[0]['filename']).stem
                city = stem.split('-')[1] if stem.startswith('street-traffic') else 'montevideo'
                location_id = stem.split('_')[0][:-4] if city == 'montevideo' else '-'.join(stem.split('-')[2:4])

            if index_max_box_only:
                df_video_file = filter_for_largest_box(df_video_file, top_n=1)

            # Determine objects from video
            df_video_objects = df_video_file.groupby('track_id').apply(
                lambda df: process_video_track(df, fov=fov, frame_width=frame_width))

            # Index audio evdents
            audio_events_list = []
            file_meta = {}
            for _, ods in df_audio_file.iterrows():
                if ods['label'] != -1:
                    audio_events_list.append({
                        'label': ods['label'],
                        'start': ods['start'],
                        'end': ods['end'],
                        'source': 'audio'
                    })
                file_meta.update({
                    'non_identifiable_vehicle_sound': bool(ods['non_identifiable_vehicle_sound'])
                })

            # Index video events
            video_events_list = []
            for track_id, ods in df_video_objects.iterrows():
                event_dict = ods.to_dict()

                event_dict['source'] = 'video'
                event_dict['track_id'] = track_id
                # Check if video event is confirmed by audio.
                # This is true if at least half the timestamp from video labels are included in audio regions
                confirmed = [
                    any((t >= aev['start'] and t <= aev['end']) and
                        aev['label'] == event_dict['label']
                        for aev in audio_events_list)
                    for t in event_dict['time']
                ]
                event_dict['audio_confirmation'] = confirmed
                event_dict['time_filtered'] = [
                        t for t, c in zip(event_dict['time'], confirmed) if c]
                event_dict['azimuth_filtered'] = [
                        t for t, c in zip(event_dict['azimuth'], confirmed) if c]
                event_dict['azimuth_left_filtered'] = [
                        t for t, c in zip(event_dict['azimuth_left'], confirmed) if c]
                event_dict['azimuth_right_filtered'] = [
                        t for t, c in zip(event_dict['azimuth_right'], confirmed) if c]
                event_dict['amount_confirmed'] = np.mean(confirmed)
                event_dict['confirmed'] = bool(np.mean(confirmed) > 0)

                video_events_list.append(event_dict)

            # Store file information
            index[uid] = {
                'path': str(audio_path.relative_to(self.root_folder)),
                'events': audio_events_list + video_events_list,
                'city': city,
                'location_id': location_id,
                **file_meta
            }
            index[uid].update(metadata)

        # Digest
        print(f'Indexed {len(index)} files')

        return index

    def save_index(self, index:dict):
        """
        Writes index to <datasetname>.json.
        
        Arguments:
            index: dictionary containing index data.

        Returns:

        """
        # Save index
        self.index_path.parent.mkdir(exist_ok=True, parents=True)
        with self.index_path.open('w') as f:
            json.dump(index, f, indent=2)
        print(f'Index for {self.__class__.__name__} saved to: {self.index_path}')


class Urbansas(AudioVisualDataset):
    """
    Custom indexer for Urbansas dataset.

    Attributes:
        root: path to dataset root.
        index_path: path to index file
        overwrite: boolean, set to True to overwrite existing index.
    """

    def __init__(self, *, root: str or Path = None, index_path: Path or str = None, overwrite: bool = False):
        """
        Initialize indexer for Urbansas class.
        """
        super().__init__(
            root=urbansas_root if root is None else Path(root),
            index_path=index_path,
            overwrite=overwrite
        )

    def index(self, audio_metadata_path:str, video_metadata_path:str, audio_folder:str, fov:float,
              frame_width:float, index_max_box_only:bool):
        """
        Creates the index file for Urbansas and writes it out to Urbansas.json.

        Arguments:
            audio_metadata_path: path to the audio metadata CSV.
            video_metadata_path: path to the video metadata CSV.
            audio_folder: folder with audio files.
            fov: field of view.
            frame_width: width of video frame.
            index_max_box_only: boolean indicating if only the top N bounding boxes
                per class/frame will be used in the index.

        Returns: 
            dict: index dictionary
        """
        index = self._index(
            audio_metadata_path=audio_metadata_path,
            video_metadata_path=video_metadata_path,
            audio_folder=audio_folder,
            fov=fov,
            frame_width=frame_width,
            index_max_box_only=index_max_box_only
        )

        # Create folds for train-validation split. Stratified 5-fold strategy on cities/location
        print('Creating folds')
        df = pd.DataFrame.from_dict(index, orient='index')
        print(df.head())

        for key in ['city', 'location_id']:
            for fold, group in enumerate(
                    StratifiedKFold(n_splits=5, shuffle=True, random_state=41).split(df, df[key])):
                for uid in df.index[group[1]]:  # Test uid
                    index[uid][f'fold_{key}'] = fold

        self.save_index(index)
        return index
