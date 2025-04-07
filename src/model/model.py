import torch
from src.model import SE
from src.model import SAAFc2


class Model(torch.nn.Module):
    """The lightweight-baseline model largely based on https://arxiv.org/pdf/1905.06148"""

    def __init__(self, channels_n=1, filters_n=32, kernel_size=63, frame_size=4096, frames_lookahead=4):
        super().__init__()

        self.channels_n = channels_n
        self.filters_n = filters_n
        self.kernel_size = kernel_size
        self.frame_size = frame_size
        self.frames_lookahead = frames_lookahead

        self.padding = self.kernel_size // 2
        self.frames_n = 2 * frames_lookahead + 1
        self.pooling_factor = self.frame_size // 64

        # convolutional encoder, adaptive front-end
        self.conv = torch.nn.Conv1d(self.channels_n, self.filters_n,
                                    self.kernel_size, padding=self.padding)
        self.conv_local = torch.nn.Conv1d(
            self.filters_n, self.filters_n, self.kernel_size * 2 + 1, padding='same', groups=self.filters_n)
        self.softplus = torch.nn.Softplus()
        self.maxpool = torch.nn.MaxPool1d(self.pooling_factor, ceil_mode=True)

        # Bi-LSTM, learning temporal dependencies, sizes from paper
        self.lstm_hidden_size_1 = 64
        self.lstm_hidden_size_2 = 32
        self.lstm_hidden_size_3 = 16
        self.lstm_dropout = 0.1

        self.bi_lstm_1 = torch.nn.LSTM(self.filters_n * self.frames_n,
                                       self.lstm_hidden_size_1, bidirectional=True, batch_first=True)
        self.bi_lstm_2 = torch.nn.LSTM(2 * self.lstm_hidden_size_1,
                                       self.lstm_hidden_size_2, bidirectional=True, batch_first=True)
        self.bi_lstm_3 = torch.nn.LSTM(2 * self.lstm_hidden_size_2,
                                       self.lstm_hidden_size_3, bidirectional=True, batch_first=True)
        self.dropout = torch.nn.Dropout(self.lstm_dropout)

        self.dnn_latent = torch.nn.Linear(2 * self.lstm_hidden_size_3, 2 * self.lstm_hidden_size_3)

        # SAAF
        self.breakpoints_n = 25
        self.breakpoints = [-1.0 + 2.0 / self.breakpoints_n
                            * k for k in range(self.breakpoints_n + 1)]
        self.saaf = SAAFc2(self.breakpoints, self.filters_n)

        # decoder, synthesis back-end
        self.unpool = torch.nn.Upsample(scale_factor=self.pooling_factor)

        # DNN-SAAF-SE
        self.dnn_hidden_size_1 = self.filters_n
        self.dnn_hidden_size_2 = self.filters_n // 2
        self.dnn_hidden_size_3 = self.filters_n // 2
        self.dnn_hidden_size_4 = self.filters_n
        self.dnn_saaf = torch.nn.Sequential(
            torch.nn.Linear(self.filters_n, self.dnn_hidden_size_1),
            torch.nn.Tanh(),
            torch.nn.Linear(self.dnn_hidden_size_1, self.dnn_hidden_size_2),
            torch.nn.Tanh(),
            torch.nn.Linear(self.dnn_hidden_size_2, self.dnn_hidden_size_3),
            torch.nn.Tanh(),
            torch.nn.Linear(self.dnn_hidden_size_3, self.dnn_hidden_size_4),
            SAAFc2(self.breakpoints, self.filters_n),
        )

        # SE block
        self.se_hidden_size_1 = 512
        self.se_hidden_size_2 = self.filters_n
        self.se_block = SE(self.filters_n, self.se_hidden_size_1, self.se_hidden_size_2)

    def forward(self, input_audio, **batch):
        """
        Input must has shape (batch_size, frames_n, channels_n, frame_size).
        Makes the prediction of the intermediate frame so output has shape (batch_size, channels_n, frame_size)
        """
        batch_size, _, _, _ = input_audio.shape
        input_audio = input_audio.reshape(
            batch_size * self.frames_n, self.channels_n, self.frame_size)

        output = self.conv(input_audio)

        output = output.reshape(batch_size, self.frames_n, self.filters_n, self.frame_size)
        residual_1 = output[:, self.frames_lookahead, :, :]

        output = torch.abs(output)

        output = output.reshape(batch_size * self.frames_n, self.filters_n, self.frame_size)
        output = self.conv_local(output)
        output = self.softplus(output)

        output = self.maxpool(output)

        output = output.reshape(batch_size, self.frames_n, self.filters_n, -1)
        output = output.reshape(batch_size, self.frames_n * self.filters_n, -1)
        output = output.transpose(-1, -2)

        output, _ = self.bi_lstm_1(output)
        output = self.dropout(output)

        output, _ = self.bi_lstm_2(output)
        output = self.dropout(output)

        output, _ = self.bi_lstm_3(output)

        output = self.dnn_latent(output)
        output = self.saaf(output)

        output = output.transpose(-1, -2)

        output = self.unpool(output)

        output = output * residual_1
        residual_2 = output

        output = output.transpose(-1, -2)
        output = self.dnn_saaf(output)
        output = output.transpose(-1, -2)

        output = self.se_block(output)

        output = output + residual_2
        output = torch.nn.functional.conv_transpose1d(
            output, self.conv.weight, padding=self.padding)

        return {"predicted_audio": output}

    def load_pretrained(self, state_dict):
        """Loading weights from the pre-training stage"""
        for name, parameters in state_dict.items():
            self.state_dict()[name].copy_(parameters)

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
