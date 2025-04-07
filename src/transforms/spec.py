import torch
from torchaudio.transforms import Spectrogram


class GetSpectrogram(torch.nn.Module):
    """STFT-spectrogram transform, returns the magnitude**2 of the STFT transform"""

    def __init__(self):
        super().__init__()

        self.spec_transform = Spectrogram(n_fft=254)
        self.eps = 1e-7
        self.expected_frames = 1024

    def forward(self, input_audio):
        spectrogram = self.spec_transform(input_audio)
        spectrogram = spectrogram.transpose(-1, -2)
        spectrogram = spectrogram.squeeze(0)
        current_frames = spectrogram.shape[0]
        if current_frames > self.expected_frames:
            spectrogram = spectrogram[:self.expected_frames, :]
        elif current_frames < self.expected_frames:
            spectrogram = torch.nn.functional.pad(
                spectrogram, (0, 0, 0, self.expected_frames - current_frames))
        spectrogram = (spectrogram + self.eps).log()
        spectrogram = spectrogram.unsqueeze(0)
        return spectrogram


class GetSTFT(torch.nn.Module):
    """STFT-transform"""

    def __init__(self):
        super().__init__()

        self.n_fft = 254
        self.expected_frames = 1024
        self.window = torch.hann_window(self.n_fft)

    def forward(self, input_audio):
        stft = torch.stft(input_audio,
                          n_fft=self.n_fft,
                          hop_length=self.n_fft // 2,
                          win_length=self.n_fft,
                          window=self.window,
                          normalized=False,
                          return_complex=True
                          )

        stft = stft.transpose(-1, -2)
        stft = stft.squeeze(0)
        current_frames = stft.shape[0]
        if current_frames > self.expected_frames:
            stft = stft[:self.expected_frames, :]
        elif current_frames < self.expected_frames:
            stft = torch.nn.functional.pad(stft, (0, 0, 0, self.expected_frames - current_frames))
        stft = stft.unsqueeze(0)
        return stft
