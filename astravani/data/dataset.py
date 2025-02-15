import logging
import os
import random
from collections import defaultdict

# import numpy as np
# import torchaudio as ta
from pathlib import Path
from typing import List, Literal, Optional, Union

import torch
from dataclasses import dataclass
from torch.utils.data import Dataset

from astravani.core import AudioSignal
from astravani.data.tokenizers import Tokenizer
from astravani.utils.helpers import (
    SUP_DATA_TYPES_SET,
    read_manifest,
    stack_tensors,
    update_tracker,
)

def has_nan(tensor):
    return torch.isnan(tensor).any().item()


@dataclass(kw_only=True)
class AudioDataset(Dataset):
    """
    A PyTorch Dataset for handling audio data.

    Args:
        manifest_fpaths (Union[str, Path, List[str], List[Path]]): Path(s) to the manifest file(s) containing the audio data.
        sample_rate (int): The desired sample rate for the audio data.
        min_duration (float, optional): The minimum duration of audio data to include. Defaults to 0.58.
        max_duration (float, optional): The maximum duration of audio data to include. Defaults to 5.0.
        slice_audio (bool, optional): Whether to slice audio data that exceeds the maximum duration. Defaults to True.
    """

    manifest_fpaths: Union[str, Path, List[str], List[Path]]
    sample_rate: int = 24_000
    min_duration: float = 0.58
    max_duration: float = 5.0
    slice_audio: bool = True
    ### Spectrogram configs
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 128
    window: str = "hann"
    center: bool = False
    normalized: bool = True
    ### Supplementary data params
    sup_data_types: Optional[List[str]] = None
    sup_data_path: Optional[str] = None

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
        logging.info(f"TOTAL DURATION FOUND: {total_data['time'] / 3600:.2f} hours")
        logging.info(
            f"PRUNED: {pruned_data['data_point']} / {total_data['data_point']}"
        )
        logging.info(f"PRUNED: {pruned_data['time'] / 3600} / {total_data['time'] / 3600:.2f} hours")
        logging.info(
            f"FILTERED: {filtered_data['data_point']} / {total_data['data_point']}"
        )
        logging.info(f"FILTERED: {filtered_data['time'] / 3600} / {total_data['time'] / 3600:.2f} hours")

        for sup_data_type in self.sup_data_types:
            if sup_data_type not in SUP_DATA_TYPES_SET:
                raise ValueError(
                    f"Invalid supplementary data type ({sup_data_type}). Choose from: "
                    + "\n".join(SUP_DATA_TYPES_SET)
                )

        if self.sup_data_path is None:
            assert len(self.sup_data_types) == 0, "No supplementary data path provided."

        if self.sup_data_path is not None and 0 < len(self.sup_data_types):
            assert self.sup_data_path, "To extract/use suplementary data, please provide path to store/load the data from."
            os.makedirs(self.sup_data_path, exist_ok=True)
            if "energy" in self.sup_data_types:
                os.makedirs(os.path.join(self.sup_data_path, "energy"), exist_ok=True)

            if "reference_audio" in self.sup_data_types:
                assert (
                    "speaker_id" in self.sup_data_types
                ), "Reference audio requires speaker_id to be provided."
                self.speaker_id_to_idx_map = defaultdict(set)
                for i, d in enumerate(self.data):
                    self.speaker_id_to_idx_map[d["speaker_id"]].add(i)
            
            if "mel_spec" in self.sup_data_types:
                os.makedirs(os.path.join(self.sup_data_path, "mel_spec"), exist_ok=True)

    def __sample_reference_audio(self, speaker_id: int):
        pool = self.speaker_id_to_idx_map[speaker_id]
        idx = random.sample(pool, 1)[0]
        return self.data[idx]["audio_filepath"]

    def _filter_for_sliced_ds(self, item):
        return self.min_duration <= item["duration"]

    def _filter_ds(self, item):
        return self.min_duration <= item["duration"] <= self.max_duration

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            audio_path = item["audio_filepath"]
            audio_signal = AudioSignal(audio_path_or_array=audio_path, sample_rate=self.sample_rate)
            if self.slice_audio and audio_signal.num_frames > self.max_frames:
                audio_signal = audio_signal.rand_slice_segment(self.max_frames)

            audio_signal.downmix_mono()
            
            audio_len = torch.tensor(audio_signal.num_frames).long()
            speaker_id = None
            if "speaker_id" in self.sup_data_types:
                speaker_id = torch.tensor(item["speaker_id"]).long()

            energy = None
            energy_len = None
            if "energy" in self.sup_data_types:
                f_path = os.path.basename(audio_path).replace(".wav", "")
                energy_path = os.path.join(self.sup_data_path, f"energy/{f_path}.pt")
                if os.path.exists(energy_path) and not self.slice_audio:
                    energy = torch.load(energy_path)
                else:
                    energy = audio_signal.get_energy(
                        self.n_fft,
                        self.hop_length,
                        self.win_length,
                        self.window,
                        self.center,
                        self.normalized,
                    )
                    torch.save(energy, energy_path)
                energy = energy.squeeze(0)
                if has_nan(energy):
                    raise ValueError(f"Energy tensor of {item['filepath']} has nan values")
                energy_len = torch.tensor(energy.shape[-1]).long()
            
            mel_spec = None
            mel_spec_len = None
            if "mel_spec" in self.sup_data_types:
                f_path = os.path.basename(audio_path).replace(".wav", "")
                mel_spec_path = os.path.join(self.sup_data_path, f"mel_spec/{f_path}.pt")
                if os.path.exists(mel_spec_path) and not self.slice_audio:
                    mel_spec = torch.load(mel_spec_path)
                else:
                    mel_spec = audio_signal.get_mel_spec(
                        self.n_fft,
                        self.hop_length,
                        self.win_length,
                        self.n_mels,
                        self.window,
                        self.center,
                        self.normalized,
                    )
                    torch.save(mel_spec, mel_spec_path)
                mel_spec = mel_spec.squeeze(0)
                if has_nan(mel_spec):
                    raise ValueError(f"Spectrogram tensor of {item['filepath']} has nan values")
                mel_spec_len = torch.tensor(mel_spec.shape[-1]).long()

            pitch = None
            pitch_len = None
            if "pitch" in self.sup_data_types:
                f_path = os.path.basename(audio_path).replace(".wav", "")
                pitch_path = os.path.join(self.sup_data_path, f"pitch/{f_path}.pt")
                if os.path.exists(pitch_path) and not self.slice_audio:
                    pitch = torch.load(pitch_path)
                else:
                    raise logging.error(
                        f"Pitch extraction is not implemented yet. Manually extract pitch and save in {pitch_path}"
                    )
                pitch_len = torch.tensor(pitch.shape[-1]).long()

            reference_audio = None
            reference_audio_len = None
            if "reference_audio" in self.sup_data_types:
                reference_audio = AudioSignal(
                    audio_path_or_array=self.__sample_reference_audio(speaker_id),
                    sample_rate=self.sample_rate,
                )
                reference_audio_len = torch.tensor(reference_audio.num_frames).long()

            return {
                "audio_signal": audio_signal,
                "audio_len": audio_len,
                "speaker_id": speaker_id,
                "energy": energy,
                "energy_len": energy_len,
                "pitch": pitch,
                "pitch_len": pitch_len,
                "mel_spec": mel_spec,
                "mel_spec_len": mel_spec_len,
                "reference_audio": reference_audio,
                "referene_audio_len": reference_audio_len,
                "idx": idx,
            }
        except Exception as e:
            logging.warn(f"Skipping {idx} because of {e}")
            return self.__getitem__((idx + 1) % len(self))

    def collate_fn(self, batch):
        batch = {key: [item[key] for item in batch] for key in batch[0]}
        (
            audio_lens,
            energy_lens,
            pitch_lens,
            mel_spec_lens,
            reference_audio_lens,
        ) = (
            batch["audio_len"],
            batch["energy_len"],
            batch["pitch_len"],
            batch["mel_spec_len"],
            batch["referene_audio_len"],
        )
        
        max_energy_len = (
            max(energy_lens).item() if "energy" in self.sup_data_types else None
        )
        max_pitch_len = (
            max(pitch_lens).item() if "pitch" in self.sup_data_types else None
        )
        max_mel_spec_len = (
            max(mel_spec_lens).item() if "mel_spec" in self.sup_data_types else None
        )

        audio_signal = AudioSignal.from_list(
            audios=batch["audio_signal"], sample_rate=self.sample_rate
        )

        energies = (
            stack_tensors(batch["energy"], [max_energy_len])
            if "energy" in self.sup_data_types
            else None
        )
        pitches = (
            stack_tensors(batch["pitch"], [max_pitch_len])
            if "pitch" in self.sup_data_types
            else None
        )
        mel_specs = (
            stack_tensors(batch["mel_spec"], [max_mel_spec_len])
            if "mel_spec" in self.sup_data_types
            else None
        )
        reference_audios = (
            AudioSignal.from_list(
                audios=batch["reference_audio"], sample_rate=self.sample_rate
            )
            if "reference_audio" in self.sup_data_types
            else None
        )
        speaker_ids = (
            torch.stack(batch["speaker_id"])
            if "speaker_id" in self.sup_data_types
            else None
        )

        return {
            "audio_signal": audio_signal,
            "audio_len": torch.stack(audio_lens),
            "speaker_id": speaker_ids,
            "energy": energies,
            "energy_len": torch.stack(energy_lens) if "energy" in self.sup_data_types else None,
            "pitch": pitches,
            "pitch_len": torch.stack(pitch_lens) if "pitch" in self.sup_data_types else None,
            "mel_spec": mel_specs,
            "mel_spec_len": torch.stack(mel_spec_lens) if "mel_spec" in self.sup_data_types else None,
            "reference_audio": reference_audios,
            "referene_audio_len": torch.stack(reference_audio_lens) if "reference_audio" in self.sup_data_types else None,
        }


@dataclass(kw_only=True)
class TTSDataset(AudioDataset):
    tokenizer: Tokenizer
    slice_audio: bool = False # override AudioDataset
    ### curriculum learning params
    sort_batch_by: Optional[Literal["audio", "text"]] = False

    def __post_init__(self):
        if self.slice_audio:
            logging.warning(
                "slice_audio is not intended to be used with TTSDataset. If this was not intentional, please set slice_audio to False before proceeding."
            )
            # self.slice_audio = False

        super().__post_init__()

        if self.sup_data_types is None:
            self.sup_data_types = []

        for i, item in enumerate(self.data):
            text = item.get("text")
            text_tokens = self.tokenizer.encode(text)
            item["text_tokens"] = text_tokens
            if text is None:
                raise ValueError(f"No text data found for item at index {i}")

    def __getitem__(self, idx):
        try:
            audio_ds_item = super().__getitem__(idx)
            item = self.data[audio_ds_item["idx"]]
            text_tokens = torch.tensor(item["text_tokens"]).long()
            token_len = torch.tensor(len(text_tokens)).long()
            return {
                **audio_ds_item,
                "text": text_tokens,
                "text_len": token_len,
            }
        except Exception as e:
            logging.warn(f"Skipping {idx} because of {e}")
            return self.__getitem__((idx + 1) % len(self))

    def collate_fn(self, batch):
        batch_n = {key: [item[key] for item in batch] for key in batch[0]}
        audio_batch = super().collate_fn(batch)

        token_lens = batch_n["text_len"]
        max_token_len = max(token_lens).item()
        text_tokens = stack_tensors(batch_n["text"], [max_token_len], self.tokenizer.pad_id)

        batch = {
            **audio_batch,
            "text": text_tokens,
            "text_len": torch.stack(token_lens),
        }
        if self.sort_batch_by == "audio":
            batch = sorted(batch, key=lambda x: x["audio_len"])
        elif self.sort_batch_by == "text":
            batch = sorted(batch, key=lambda x: x["text_len"])
        return batch
