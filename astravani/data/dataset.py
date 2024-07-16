from typing import List, Union, Optional

import torch
import json
import logging
# import numpy as np
# import torchaudio as ta

from pathlib import Path
from torch.utils.data import Dataset
from astravani.core import AudioSignal

from pydantic.dataclasses import dataclass


def read_manifest(path):
    return list(map(json.loads, open(path, 'r').readlines()))

def write_manifest(path, manifest, ensure_ascii=False):
    return open(path, 'w').writelines([json.dumps(x, ensure_ascii=ensure_ascii) + '\n' for x in manifest])

def update_tracker(tracker, data):
    tracker["data_point"] += 1
    tracker["time"] += data["duration"]


@dataclass
class AudioDataset(Dataset):
    """
    A PyTorch Dataset for handling audio data.

    Args:
        manifest_fpaths (Union[str, Path, List[str], List[Path]]): Path(s) to the manifest file(s) containing the audio data.
        sample_rate (int): The desired sample rate for the audio data.
        min_duration (float, optional): The minimum duration of audio data to include. Defaults to 0.58.
        max_duration (float, optional): The maximum duration of audio data to include. Defaults to 5.0.
        slice_audio (bool, optional): Whether to slice audio data that exceeds the maximum duration. Defaults to True.
        f_min (float, optional): The minimum frequency to include in the audio data. Defaults to 0.
        f_max (Optional[float], optional): The maximum frequency to include in the audio data. Defaults to None.
    """

    manifest_fpaths: Union[str, Path, List[str], List[Path]]
    sample_rate: int
    min_duration: float = 0.58
    max_duration: float = 5.0
    slice_audio: bool = True
    f_min: float = 0
    f_max: Optional[float] = None

    def __post_init__(self):
        self.max_frames = int(self.max_duration * self.sample_rate)
        if isinstance(self.manifest_fpaths, str):
            self.manifest_fpaths = [self.manifest_fpaths]
        
        if self.slice_audio:
            cond_fn = self._filter_for_sliced_ds
        else:
            cond_fn = self._filter_ds

        data = []
        total_data = {"data_point": 0, "time": 0.0}
        pruned_data = {"data_point": 0, "time": 0.0}
        filtered_data = {"data_point": 0, "time": 0.0}
        for file in self.manifest_fpaths:
            manifest_data = read_manifest(file)
            for item in manifest_data:
                update_tracker(total_data, item)
                if cond_fn(item):
                    data.append(item)
                    update_tracker(filtered_data, item)
                else:
                    update_tracker(pruned_data, item)
        
        self.data = data
        
        logging.info(f"TOTAL DATAPOINTS FOUND: {total_data['data_point']}")
        logging.info(f"TOTAL DURATION FOUND: {total_data['time'] / 3600} hours")
        logging.info(f"PRUNED: {pruned_data['data_point']} / {total_data['data_point']}")
        logging.info(f"PRUNED: {pruned_data['time']} / {total_data['time']}")
        logging.info(f"FILTERED: {filtered_data['data_point']} / {total_data['data_point']}")
        logging.info(f"FILTERED: {filtered_data['time']} / {total_data['time']}")

    def _filter_for_sliced_ds(self, item):
        return self.min_duration <= item["duration"]
    
    def _filter_ds(self, item):
        return self.min_duration <= item["duration"] <= self.max_duration
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        audio_path = item["audio_path"]
        audio_signal = AudioSignal(audio_path, sample_rate=self.sample_rate)
        if self.slice_audio and audio_signal.num_frames > self.max_frames:
            audio_signal = audio_signal.rand_slice_segment(self.max_frames)
        
        audio_signal = audio_signal.downmix_mono()
        audio_len = torch.tensor(audio_signal.num_frames).long()
        return {
            "audio_signal": audio_signal,
            "audio_len": audio_len,
        }
    
    def collate_fn(self, batch):
        audio_signals = [item["audio_signal"] for item in batch]
        audio_lens = [item["audio_len"] for item in batch]
        audio_signals = AudioSignal.from_list(audio_signals, self.sample_rate)
        audio_lens = torch.stack(audio_lens)
        return {
            "audio_signals": audio_signals,
            "audio_lens": audio_lens,
        }


