from typing import List, Union, Optional

import torch
import json
import logging
import numpy as np
import torchaudio as ta

from pathlib import Path
from torch.utils.data import Dataset

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
    manifest_fpaths: Union[str, Path, List[str], List[Path]]
    sample_rate: int
    min_duration: float = 0.58
    max_duration: float = 5.0
    slice_audio: bool = True
    f_min: float = 0
    f_max: Optional[float] = None

    def __post_init__(self):
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