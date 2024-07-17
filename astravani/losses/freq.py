from typing import Dict, List, Optional, Tuple, Union

import torch
import torchaudio as ta

from astravani.core import AudioSignal
from astravani.utils.helpers import apply_reduction


class SpectralConvergenceLoss(torch.nn.Module):
    """
    Adapted from auraloss.freq.SpectralConvergenceLoss.

    This class implements the Spectral Convergence Loss module. It computes the loss between the magnitudes
    of two signals in the frequency domain. The loss is defined as the Frobenius norm of the difference between
    the magnitudes of the two signals, normalized by the Frobenius norm of the magnitudes of the first signal.
    See [Arik et al., 2018](https://arxiv.org/abs/1808.06719).
    """

    def __init__(self):
        super(SpectralConvergenceLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")


class STFTMagnitudeLoss(torch.nn.Module):
    """
    This class implements a loss function for comparing the magnitudes of Short-Time Fourier Transforms (STFT)
    of two signals. The loss can be computed using either L1 or L2 distance.

    Args:
        log (bool, optional): If True, the logarithm of the magnitudes is used before computing the distance.
            Default is True.
        log_eps (float, optional): A small value added to the magnitudes before taking the logarithm to avoid
            numerical instability. Default is 0.0.
        log_fac (float, optional): A factor by which the magnitudes are multiplied before taking the logarithm.
            Default is 1.0.
        distance (str, optional): The distance metric to use. Can be either "L1" for L1 loss or "L2" for MSE loss.
            Default is "L1".
        reduction (str, optional): The reduction to apply to the loss. Can be either "none", "mean", or "sum".
            Default is "mean".

    Attributes:
        log (bool): Whether to use the logarithm of the magnitudes.
        log_eps (float): The small value added to the magnitudes before taking the logarithm.
        log_fac (float): The factor by which the magnitudes are multiplied before taking the logarithm.
        distance (torch.nn.Module): The distance metric to use.

    Methods:
        forward(x_mag, y_mag): Computes the loss between the magnitudes of two signals.
    """

    def __init__(
        self, log=True, log_eps=0.0, log_fac=1.0, distance="L1", reduction="mean"
    ):
        super(STFTMagnitudeLoss, self).__init__()

        self.log = log
        self.log_eps = log_eps
        self.log_fac = log_fac

        if distance == "L1":
            self.distance = torch.nn.L1Loss(reduction=reduction)
        elif distance == "L2":
            self.distance = torch.nn.MSELoss(reduction=reduction)
        else:
            raise ValueError(f"Invalid distance: '{distance}'.")

    def forward(self, x_mag, y_mag):
        if self.log:
            x_mag = torch.log(self.log_fac * x_mag + self.log_eps)
            y_mag = torch.log(self.log_fac * y_mag + self.log_eps)
        return self.distance(x_mag, y_mag)


class STFTLoss(torch.nn.Module):
    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        window: str = "hann",
        lambda_sc: float = 1.0,
        lambda_log_mag: float = 1.0,
        lambda_lin_mag: float = 0.0,
        lambda_phase: float = 0.0,
        sample_rate: float = None,
        n_bins: Optional[int] = None,
        scale: Optional[str] = None,
        scale_invariance: bool = False,
        eps: float = 1e-8,
        reduction: str = "mean",
        mag_distance: str = "L1",
        device: Optional[Union[torch.device, str]] = None,
        **kwargs,
    ):
        super(STFTLoss, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.lambda_sc = lambda_sc
        self.lambda_log_mag = lambda_log_mag
        self.lambda_lin_mag = lambda_lin_mag
        self.lambda_phase = lambda_phase
        self.sample_rate = sample_rate
        self.n_bins = n_bins
        self.scale = scale
        self.scale_invariance = scale_invariance
        self.eps = eps
        self.reduction = reduction
        self.mag_distance = mag_distance
        self.device = device

        self.use_phase = bool(self.lambda_phase)

        self.spectral_conv = SpectralConvergenceLoss()
        self.log_stft = STFTMagnitudeLoss(
            log=True, distance=self.mag_distance, reduction=self.reduction, **kwargs
        )
        self.lin_stft = STFTMagnitudeLoss(
            log=False, distance=self.mag_distance, reduction=self.reduction, **kwargs
        )

        if scale is not None:
            if scale == "mel":
                fb = ta.functional.melscale_fbanks(
                    int(n_fft // 2 + 1),
                    n_mels=n_bins,
                    sample_rate=self.sample_rate,
                    f_min=0,
                    f_max=self.sample_rate / 2.0,
                    norm="slaney",
                ).to(self.device)
            else:
                raise NotImplementedError(f"scale={scale} is not supported.")
            self.register_buffer("fb", fb)

    def stft(self, audio_signal: AudioSignal) -> Tuple[torch.Tensor, torch.Tensor]:
        x_stft = audio_signal.stft(
            self.n_fft, self.hop_length, self.win_length, self.window
        )

        x_mag = torch.sqrt(torch.clamp(x_stft.real**2 + x_stft.imag**2, min=self.eps))

        if self.use_phase:
            x_phs = torch.angle(x_stft)
        else:
            x_phs = None

        return x_mag, x_phs

    def forward(
        self, x: AudioSignal, y: AudioSignal
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        x_mag, x_phs = self.stft(x)
        y_mag, y_phs = self.stft(y)

        if self.scale is not None:
            self.fb = self.fb.to(x_mag.device)
            x_mag = torch.matmul(self.fb, x_mag)
            y_mag = torch.matmul(self.fb, y_mag)

        if self.scale_invariance:
            alpha = (x_mag * y_mag).sum([-2, -1]) / ((y_mag**2).sum([-2, -1]))
            y_mag = y_mag * alpha.unsqueeze(-1)

        sc_mag_loss = self.spectral_conv(x_mag, y_mag) if self.w_sc else 0.0
        log_mag_loss = self.logstft(x_mag, y_mag) if self.w_log_mag else 0.0
        lin_mag_loss = self.linstft(x_mag, y_mag) if self.w_lin_mag else 0.0
        phs_loss = torch.nn.functional.mse_loss(x_phs, y_phs) if self.phs_used else 0.0

        # combine loss terms
        loss = (
            (self.w_sc * sc_mag_loss)
            + (self.w_log_mag * log_mag_loss)
            + (self.w_lin_mag * lin_mag_loss)
            + (self.w_phs * phs_loss)
        )

        loss = apply_reduction(loss, reduction=self.reduction)

        return loss, {
            "sc_mag_loss": sc_mag_loss,
            "log_mag_loss": log_mag_loss,
            "lin_mag_loss": lin_mag_loss,
            "phs_loss": phs_loss,
        }


class MelSTFTLoss(STFTLoss):
    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        window: str = "hann",
        lambda_sc: float = 1.0,
        lambda_log_mag: float = 1.0,
        lambda_lin_mag: float = 0.0,
        lambda_phase: float = 0.0,
        sample_rate: float = None,
        n_bins: Optional[int] = None,
        scale_invariance: bool = False,
        eps: float = 1e-8,
        reduction: str = "mean",
        mag_distance: str = "L1",
        device: Optional[Union[torch.device, str]] = None,
        **kwargs,
    ):
        super(MelSTFTLoss, self).__init__(
            n_fft,
            hop_length,
            win_length,
            window,
            lambda_sc,
            lambda_log_mag,
            lambda_lin_mag,
            lambda_phase,
            sample_rate,
            n_bins,
            "mel",
            scale_invariance,
            eps,
            reduction,
            mag_distance,
            device,
            **kwargs,
        )


class MultiResolutionSTFTLoss(torch.nn.Module):
    def __init__(
        self,
        n_ffts: List[int] = [1024, 2048, 512],
        hop_lengths: List[int] = [120, 240, 512],
        win_lengths: List[int] = [600, 1200, 240],
        window: str = "hann",
        lambda_sc: float = 1.0,
        lambda_log_mag: float = 1.0,
        lambda_lin_mag: float = 0.0,
        lambda_phase: float = 0.0,
        sample_rate: float = None,
        n_bins: Optional[int] = None,
        scale_invariance: bool = False,
        eps: float = 1e-8,
        reduction: str = "mean",
        mag_distance: str = "L1",
        device: Optional[Union[torch.device, str]] = None,
        **kwargs,
    ):
        super(MultiResolutionSTFTLoss, self).__init__()
        assert (
            len(n_ffts) == len(hop_lengths) == len(win_lengths)
        ), "Length of n_ffts, hop_lengths, and win_lengths must be the same."
        self.n_ffts = n_ffts
        self.hop_lengths = hop_lengths
        self.win_lengths = win_lengths

        self.stft_losses = torch.nn.ModuleList([])
        for n_fft, hop_length, win_length in zip(n_ffts, hop_lengths, win_lengths):
            self.stft_losses.append(
                STFTLoss(
                    n_fft,
                    hop_length,
                    win_length,
                    window,
                    lambda_sc,
                    lambda_log_mag,
                    lambda_lin_mag,
                    lambda_phase,
                    sample_rate,
                    n_bins,
                    None,  # no mel scale
                    scale_invariance,
                    eps,
                    reduction,
                    mag_distance,
                    device,
                    **kwargs,
                )
            )

    def forward(
        self, x: AudioSignal, y: AudioSignal
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        total_loss = []

        # Initialize a dictionary to store individual losses for each resolution
        losses = {
            "sc_mag_loss": [],
            "log_mag_loss": [],
            "lin_mag_loss": [],
            "phs_loss": [],
        }

        # Compute loss for each resolution and accumulate the total loss
        for stft_loss in self.stft_losses:
            loss, individual_losses = stft_loss(x, y)
            for key, value in individual_losses.items():
                losses[key].append(value)
            total_loss.append(loss)

        # Average the losses over the number of resolutions
        for key in losses.keys():
            losses[key] = apply_reduction(
                torch.stack(losses[key]), reduction=self.reduction
            )

        total_loss = apply_reduction(torch.stack(total_loss), reduction=self.reduction)

        return total_loss, losses


class MultiResolutionMelSTFTLoss(torch.nn.Module):
    def __init__(
        self,
        n_ffts: List[int] = [1024, 2048, 512],
        hop_lengths: List[int] = [120, 240, 512],
        win_lengths: List[int] = [600, 1200, 240],
        window: str = "hann",
        lambda_sc: float = 1.0,
        lambda_log_mag: float = 1.0,
        lambda_lin_mag: float = 0.0,
        lambda_phase: float = 0.0,
        sample_rate: float = None,
        n_bins: Optional[int] = 80,
        scale_invariance: bool = False,
        eps: float = 1e-8,
        reduction: str = "mean",
        mag_distance: str = "L1",
        device: Optional[Union[torch.device, str]] = None,
        **kwargs,
    ):
        super(MultiResolutionMelSTFTLoss, self).__init__()
        assert (
            len(n_ffts) == len(hop_lengths) == len(win_lengths)
        ), "Length of n_ffts, hop_lengths, and win_lengths must be the same."
        self.n_ffts = n_ffts
        self.hop_lengths = hop_lengths
        self.win_lengths = win_lengths

        self.stft_losses = torch.nn.ModuleList([])
        for n_fft, hop_length, win_length in zip(n_ffts, hop_lengths, win_lengths):
            self.stft_losses.append(
                STFTLoss(
                    n_fft,
                    hop_length,
                    win_length,
                    window,
                    lambda_sc,
                    lambda_log_mag,
                    lambda_lin_mag,
                    lambda_phase,
                    sample_rate,
                    n_bins,
                    "mel",  # no mel scale
                    scale_invariance,
                    eps,
                    reduction,
                    mag_distance,
                    device,
                    **kwargs,
                )
            )

    def forward(
        self, x: AudioSignal, y: AudioSignal
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        total_loss = []

        # Initialize a dictionary to store individual losses for each resolution
        losses = {
            "sc_mag_loss": [],
            "log_mag_loss": [],
            "lin_mag_loss": [],
            "phs_loss": [],
        }

        # Compute loss for each resolution and accumulate the total loss
        for stft_loss in self.stft_losses:
            loss, individual_losses = stft_loss(x, y)
            for key, value in individual_losses.items():
                losses[key].append(value)
            total_loss.append(loss)

        # Average the losses over the number of resolutions
        for key in losses.keys():
            losses[key] = apply_reduction(
                torch.stack(losses[key]), reduction=self.reduction
            )

        total_loss = apply_reduction(torch.stack(total_loss), reduction=self.reduction)

        return total_loss, losses
