import torch
from torchaudio.compliance import kaldi


class GetMelSpectrogram(torch.nn.Module):
    "Mel-spectrogram transform with parametrs from  the original Audio-MAE paper: https://arxiv.org/pdf/2207.06405"

    def __init__(self):
        super().__init__()

        self.mean = -4.2677393
        self.std = 4.5689974
        self.expected_frames = 1024

    def forward(self, input_audio):
        """Input audio must has sr=16000 and will be tranformet to match the 1024 x 128 shape"""

        mel_spectrogram = kaldi.fbank(
            input_audio,
            num_mel_bins=128,
            htk_compat=True,
            use_energy=False,
            sample_frequency=16000,
            window_type='hanning'
        )

        curr_frames = mel_spectrogram.shape[0]
        if curr_frames > self.expected_frames:
            mel_spectrogram = mel_spectrogram[:self.expected_frames, :]
        elif curr_frames < self.expected_frames:
            mel_spectrogram = torch.nn.functional.pad(
                mel_spectrogram, (0, 0, 0, self.expected_frames - curr_frames))

        mel_spectrogram = mel_spectrogram.unsqueeze(0)
        return mel_spectrogram
