import torch
from src.utils.audio_utils import frame_audio


class FrameInput(torch.nn.Module):
    """
    Frame input audio to have shape (batch_size, frames_n, channels_n, frame_size)
    """

    def __init__(self, frame_size=4096, hop_size=2048, frames_lookahead=4):
        super().__init__()

        self.frame_size = frame_size
        self.hop_size = hop_size
        self.frames_lookahead = frames_lookahead
        self.frames_n = self.frames_lookahead * 2 + 1

    def forward(self, input):
        if input.dim() == 1:
            input = input.unsqueeze(0)
    
        data = frame_audio(data=input, frame_size=self.frame_size, hop_size=self.hop_size)
        data = torch.nn.functional.pad(data, (0, 0, self.frames_lookahead, self.frames_lookahead))
        data = data.unfold(dimension=1, size=self.frames_n, step=1)
        data = data.squeeze(0)
        data = data.transpose(-1, -2)
        data = data.unsqueeze(2)
        
        return data


class FrameOutput(torch.nn.Module):
    """
    Frame output audio to have shape (batch_size, channels_n, frame_size)
    """

    def __init__(self, frame_size=4096, hop_size=2048):
        super().__init__()

        self.frame_size = frame_size
        self.hop_size = hop_size

    def forward(self, output):
        if output.dim() == 1:
            output = output.unsqueeze(0)

        data = frame_audio(output, frame_size=self.frame_size, hop_size=self.hop_size)
        data = data.squeeze(0)
        data = data.unsqueeze(1)

        return data
