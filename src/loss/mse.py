import torch
from src.utils.audio_utils import reconstruct_audio

class MSELoss(torch.nn.Module):
    """MSE loss"""

    def __init__(self):
        super().__init__()
        self.loss = torch.nn.MSELoss()   

    def forward(self, output_spec: torch.Tensor, predicted_spec: torch.Tensor, **batch):
        return {"loss": self.loss(output_spec, predicted_spec)}
