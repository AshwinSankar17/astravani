import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torchaudio as ta
from pydantic.dataclasses import dataclass

from astravani.utils.helpers import WINDOW_FN_SUPPORTED, stack_tensors


class Config:
    arbitrary_types_allowed = True

@dataclass(config=Config)
class AudioSignal:
    audio_path_or_array: Union[str, torch.Tensor, np.ndarray, Path]
    sample_rate: Optional[int] = None

    def __post_init__(self):
        if isinstance(self.audio_path_or_array, str) or isinstance(
            self.audio_path_or_array, Path
        ):
            self.load_from_path(self.audio_path_or_array)
        elif isinstance(self.audio_path_or_array, np.ndarray) or torch.is_tensor(
            self.audio_path_or_array
        ):
            assert (
                self.sample_rate is not None
            ), "Cannot process audio array without sample rate"
            self.load_from_array(self.audio_path_or_array)
        else:
            raise ValueError(
                "audio_path_or_array should either be Path, string, or a torch Tensor"
            )

    def load_from_path(self, path: Union[str, Path], sample_rate: Optional[int] = None):
        audio, sr = ta.load(path)
        assert audio.size(1) > 0, f"Empty audio file: {path}"

        if sample_rate is not None:
            audio = ta.functional.resample(audio, sr, sample_rate)
            self.sample_rate = sample_rate
        elif self.sample_rate is not None:
            audio = ta.functional.resample(audio, sr, self.sample_rate)
        else:
            self.sample_rate = sr

        self.signal = audio.unsqueeze(0)

    def load_from_array(
        self, array: Union[torch.Tensor, np.ndarray], sample_rate: Optional[int] = None
    ):
        if isinstance(array, np.ndarray):
            audio = torch.from_numpy(array)
        else:
            audio = array

        assert audio.size(-1) > 0, "Empty audio array"

        self.sample_rate = sample_rate
        # if sample_rate is not None and sample_rate != self.sample_rate:
        #     audio = ta.functional.resample(audio, sample_rate, self.sample_rate)

        # elif self.sample_rate is not None and sr != self.sample_rate:
        #     audio = ta.functional.resample(audio, sr, self.sample_rate)

        if audio.dim() < 2:
            audio = audio.unsqueeze(0)
        if audio.dim() < 3:
            audio = audio.unsqueeze(0)

        self.signal = audio

    @property
    def num_frames(self):
        return self.signal.size(-1)

    @property
    def duration(self):
        return self.signal.size(-1) / self.sample_rate

    @property
    def device(self):
        return self.signal.device

    @device.setter
    def device(self, device: Union[str, torch.device]):
        self.signal = self.signal.to(device)

    def downmix_mono(self):
        if self.signal.size(-2) > 1:
            self.signal = torch.mean(self.signal, dim=-2, keepdims=True).to(self.signal)

    def upmix_stereo(self):
        self.signal = torch.cat([self.signal] * 2, dim=-2).to(self.signal)

    def excerpt(self, start_idx: Optional[int] = None, end_idx: Optional[int] = None):
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = self.num_frames
        assert (
            end_idx <= self.num_frames
        ), f"Index out of bounds. Num frames in signal is {self.num_frames}."
        return AudioSignal(self.signal[..., start_idx:end_idx], self.sample_rate)

    def rand_slice_segment(self, segment_size: int):
        assert (
            segment_size <= self.num_frames
        ), "Segment size is larger than the number of frames in the signal."
        start_idx = torch.randint(0, self.num_frames - segment_size, (1,)).item()
        return self.excerpt(start_idx, start_idx + segment_size)

    def resample(self, to_sample_rate: int):
        self.signal = ta.functional.resample(
            self.signal, self.sample_rate, to_sample_rate
        )
        self.sample_rate = to_sample_rate

    def to(self, *args, **kwargs):
        self.signal = self.signal.to(*args, **kwargs)

    def cuda(self, *args, **kwargs):
        self.signal = self.signal.cuda(*args, **kwargs)

    def cpu(self, *args, **kwargs):
        self.signal = self.signal.cpu(*args, **kwargs)

    @classmethod
    def from_list(cls, audios: List["AudioSignal"], sample_rate: int):
        signals = [item.signal for item in audios]
        sample_rates = [item.sample_rate for item in audios]

        for ix, sr in enumerate(sample_rates):
            if sr != sample_rate:
                signals[ix].resample(sample_rate)

        signal_lenths = [item.num_frames for item in audios]

        max_signal_len = max(signal_lenths)

        batch_signal = stack_tensors(signals, [max_signal_len])

        return AudioSignal(batch_signal, sample_rate)

    def clone(self):
        return AudioSignal(self.signal.clone(), self.sample_rate, self.device)

    @torch.cuda.amp.autocast(enabled=False)
    def stft(
        self,
        n_fft: int,
        hop_length: int,
        win_length: int,
        window: Optional[str] = "hann",
        center: Optional[bool] = False,
        normalized: Optional[bool] = False,
    ):
        window_fn = WINDOW_FN_SUPPORTED[window]
        if self.signal.size(1) == 2:
            logging.warning(f"{type(self).__name__} does not support stereo stft computation. Downmixing to mono")
        self.downmix_mono()
        return torch.stft(
            self.signal.squeeze(1),
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window_fn(win_length, periodic=False).to(
                dtype=torch.float, device=self.device
            )
            if window_fn
            else None,
            center=center,
            normalized=normalized,
            return_complex=True,
        )

    @torch.cuda.amp.autocast(enabled=False)
    def get_spec(
        self,
        n_fft: int,
        hop_length: int,
        win_length: int,
        window: str = "hann",
        center: bool = False,
        normalized: bool = False,
        pwr: float = 2.0,
        eps: float = 1e-9,
    ):
        stft = self.stft(n_fft, hop_length, win_length, window, center, normalized)
        spec = stft.abs().pow(pwr) + eps
        return spec.to(device=self.device)

    @torch.cuda.amp.autocast(enabled=False)
    def get_energy(
        self,
        n_fft: int,
        hop_length: int,
        win_length: int,
        window: str = "hann",
        center: bool = False,
        normalized: bool = False,
        pwr: float = 2.0,
        eps: float = 1e-9,
    ):
        spec = self.get_spec(
            n_fft, hop_length, win_length, window, center, normalized, pwr, eps
        )
        energy = torch.linalg.norm(spec, axis=1).float()
        return energy.to(device=self.device)

    @torch.cuda.amp.autocast(enabled=False)
    def get_mel(
        self,
        n_fft: int,
        hop_length: int,
        win_length: int,
        n_mels: int,
        window: str = "hann",
        center: bool = False,
        normalized: bool = True,
        mel_scale: str = "htk",
        pwr: float = 2.0,
        eps: float = 1e-9,
    ):
        spec = self.get_spec(
            n_fft, hop_length, win_length, window, center, normalized, pwr
        )
        mel_filters = ta.functional.melscale_fbanks(
            int(n_fft // 2 + 1),
            n_mels=n_mels,
            sample_rate=self.sample_rate,
            f_min=0,
            f_max=int(self.sample_rate // 2),
            norm=None,
            mel_scale=mel_scale
        ).to(spec.device)
        mel_spec = torch.matmul(spec.transpose(-1, -2), mel_filters).transpose(-1, -2)
        return mel_spec.to(device=self.device)

    def detach(self):
        self.signal = self.signal.detach()

    def write(self, path: str, format: Optional[str] = "wav"):
        assert self.signal.size(0) == 1, "Only support writing single audio signal."
        ta.save(path, self.signal.squeeze(0), self.sample_rate, format=format)

    @property
    def size(self):
        return self.signal.size()

    @property
    def loudness(self):
        return ta.functional.loudness(self.signal, self.sample_rate)

    def gain(self, gain_db: float = 1.0):
        self.signal = ta.functional.gain(self.signal, gain_db=gain_db)
