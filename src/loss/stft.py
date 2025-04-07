# Code is based on https://github.com/facebookresearch/denoiser/blob/main/denoiser/stft_loss.py
# Adapted for the purpose of creating audio effects from MIT code under the original license
# Original copyright 2019 Tomoki Hayashi

import torch
import typing as tp

from src.utils.audio_utils import reconstruct_audio


def stft_helper(x: torch.Tensor, fft_size: int, hop_length: int, win_length: int,
                window: tp.Optional[torch.Tensor], normalized: bool) -> torch.Tensor:
    B, C, T = x.shape
    x_stft = torch.stft(
        x.view(-1, T), fft_size, hop_length, win_length, window,
        normalized=normalized, return_complex=True,
    )
    x_stft = x_stft.view(B, C, *x_stft.shape[1:])
    real = x_stft.real
    imag = x_stft.imag

    return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1)


class SpectralConvergenceLoss(torch.nn.Module):
    """Spectral convergence loss"""

    def __init__(self, eps: float = torch.finfo(torch.float32).eps):
        super().__init__()
        self.eps = eps

    def forward(self, x_mag: torch.Tensor, y_mag: torch.Tensor):
        return torch.norm(y_mag - x_mag, p="fro") / (torch.norm(y_mag, p="fro") + self.eps)


class LogSTFTMagnitudeLoss(torch.nn.Module):
    """Log STFT magnitude loss"""

    def __init__(self, eps: float = torch.finfo(torch.float32).eps):
        super().__init__()
        self.eps = eps

    def forward(self, x_mag: torch.Tensor, y_mag: torch.Tensor):

        return torch.nn.functional.l1_loss(torch.log(self.eps + y_mag), torch.log(self.eps + x_mag))


class STFTLosses(torch.nn.Module):
    """STFT losses"""

    def __init__(self, n_fft: int = 1024, hop_length: int = 120, win_length: int = 600,
                 window: str = "hann_window", normalized: bool = False,
                 eps: float = torch.finfo(torch.float32).eps):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.normalized = normalized
        self.register_buffer("window", getattr(torch, window)(win_length))
        self.spectral_convergenge_loss = SpectralConvergenceLoss(eps)
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss(eps)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        x_mag = stft_helper(x, self.n_fft, self.hop_length,
                            self.win_length, self.window, self.normalized)
        y_mag = stft_helper(y, self.n_fft, self.hop_length,
                            self.win_length, self.window, self.normalized)
        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)

        return sc_loss, mag_loss


class STFTLoss(torch.nn.Module):
    def __init__(self, n_fft: int = 1024, hop_length: int = 120, win_length: int = 600,
                 window: str = "hann_window", normalized: bool = False,
                 factor_sc: float = 0.1, factor_mag: float = 0.1,
                 eps: float = torch.finfo(torch.float32).eps):
        super().__init__()
        self.loss = STFTLosses(n_fft, hop_length, win_length, window, normalized, eps)
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tp.Tuple[torch.Tensor, torch.Tensor]:

        sc_loss, mag_loss = self.loss(x, y)
        return self.factor_sc * sc_loss + self.factor_mag * mag_loss


class MRSTFTLoss(torch.nn.Module):
    """Multi-resolution STFT loss"""

    def __init__(self, n_ffts: tp.Sequence[int] = [1024, 2048, 512], hop_lengths: tp.Sequence[int] = [120, 240, 50],
                 win_lengths: tp.Sequence[int] = [600, 1200, 240], window: str = "hann_window",
                 factor_sc: float = 0.1, factor_mag: float = 0.1,
                 normalized: bool = False, eps: float = torch.finfo(torch.float32).eps):
        super().__init__()
        assert len(n_ffts) == len(hop_lengths) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(n_ffts, hop_lengths, win_lengths):
            self.stft_losses += [STFTLosses(fs, ss, wl, window, normalized, eps)]
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, output_audio: torch.Tensor, predicted_audio: torch.Tensor, **batch) -> torch.Tensor:
        x = reconstruct_audio(output_audio).unsqueeze(0)
        y = reconstruct_audio(predicted_audio).unsqueeze(0)

        sc_loss = torch.zeros(1, device=self.device)
        mag_loss = torch.zeros(1, device=self.device)
        for f in self.stft_losses:
            sc_l, mag_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l

        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return {"loss": self.factor_sc * sc_loss + self.factor_mag * mag_loss}
