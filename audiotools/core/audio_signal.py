from typing import List, Union, Optional

import torch
import torchaudio as ta
import numpy as np
from pathlib import Path

from pydantic.dataclasses import dataclass

from audiotools.utils.helpers import stack_tensors, WINDOW_FN_SUPPORTED

EPSILON = 1e-9


@dataclass
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
        return self.signal[..., start_idx:end_idx]

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
        return torch.stft(
            self.signal,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window_fn(win_length, periodic=False).to(dtype=torch.float, device=self.device)
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
        window: Optional[str] = "hann",
        center: Optional[bool] = False,
        normalized: Optional[bool] = False,
        pwr: Optional[float] = 2.0,
    ):
        stft = self.stft(n_fft, hop_length, win_length, window, center, normalized)
        if stft.dtype in [torch.cfloat, torch.cdouble]:
            stft = stft.view_as_real(stft)
        spec = torch.sqrt(stft.pow(pwr).sum(-1) + EPSILON)
        return spec.to(device=self.device)

    @torch.cuda.amp.autocast(enabled=False)
    def get_log_mel(
        self,
        n_fft: int,
        hop_length: int,
        win_length: int,
        n_mels: int,
        window: Optional[str] = "hann",
        center: Optional[bool] = False,
        normalized: Optional[bool] = False,
        pwr: Optional[float] = 2.0,
    ):
        spec = self.get_spec(
            n_fft, hop_length, win_length, window, center, normalized, pwr
        )
        mel_filters = ta.functional.melscale_fbanks(
            int(n_fft // 2 + 1),
            n_mels=n_mels,
            sample_rate=self.sample_rate,
            f_min=0,
            f_max=self.sample_rate / 2.0,
            norm="slaney",
        ).to(spec.device)
        mel_spec = torch.matmul(mel_filters, spec)
        log_mel_spec = torch.log(mel_spec + EPSILON)
        return log_mel_spec

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
