import torch
from src.utils.audio_utils import reconstruct_audio

class MAELoss(torch.nn.Module):
    """
    Mean absolute error loss
    """

    def __init__(self):
        super().__init__()
        self.loss = torch.nn.L1Loss()

    def forward(self, output_audio: torch.Tensor, predicted_audio, **batch):
        output_audio = reconstruct_audio(output_audio)
        predicted_audio = reconstruct_audio(predicted_audio)
        return {"loss": self.loss(output_audio, predicted_audio)}
