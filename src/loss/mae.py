import torch


class MaeLoss(torch.nn.Module):
    """
    Mean absolute error loss
    """

    def __init__(self):
        super().__init__()
        self.loss = torch.nn.L1Loss()

    def forward(self, input_audio: torch.Tensor, output_audio: torch.Tensor, **batch):
        return {"loss": self.loss(input_audio, output_audio)}
