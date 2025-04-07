import torch


class PretrainModel(torch.nn.Module):
    """
    Encoder and Decoder blocks, pre-training unsupervised stage
    """

    def __init__(self, channels_n=1, filters_n=32, kernel_size=63, frame_size=4096):
        super().__init__()

        self.channels_n = channels_n
        self.filters_n = filters_n
        self.kernel_size = kernel_size
        self.frame_size = frame_size

        self.padding = self.kernel_size // 2
        self.pooling_factor = self.frame_size // 64

        # convolutional encoder
        self.conv = torch.nn.Conv1d(in_channels=self.channels_n, out_channels=self.filters_n,
                                    kernel_size=self.kernel_size, padding=self.padding)

        self.conv_local = torch.nn.Conv1d(in_channels=self.filters_n, out_channels=self.filters_n,
                                          kernel_size=self.kernel_size * 2 + 1, padding='same', groups=self.filters_n)

        self.softplus = torch.nn.Softplus()
        self.maxpool = torch.nn.MaxPool1d(
            kernel_size=self.pooling_factor, return_indices=True, ceil_mode=True)

        # decoding
        self.unpool = torch.nn.MaxUnpool1d(kernel_size=self.pooling_factor)

    def forward(self, input_audio, **batch):
        output = self.conv(input_audio)
        residual_1 = output

        output = torch.abs(output)

        output = self.conv_local(output)
        output = self.softplus(output)

        output, indices = self.maxpool(output)
        output = self.unpool(output, indices)

        output = residual_1 * output
        output = torch.nn.functional.conv_transpose1d(
            output, self.conv.weight, padding=self.padding)

        return {"predicted_audio": output}

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
