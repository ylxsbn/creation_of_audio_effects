import torch
from src.model import SE
from src.model import SAAFc2


class Model(torch.nn.Module):
    def __init__(self, channels_n, filters_n=32, kernel_size=63, frame_size=4096, frames_lookahead=0):
        super().__init__()

        self.filters_n = filters_n
        self.padding = kernel_size // 2
        self.frames_n = 2 * frames_lookahead + 1
        self.frames_lookahead = frames_lookahead
        self.pooling_factor = frame_size // 64

        # convolution encoder, adaptive front-end
        self.conv = torch.nn.Conv1d(channels_n, filters_n, kernel_size, padding=self.padding)
        self.conv_local = torch.nn.Conv1d(filters_n, filters_n, kernel_size * 2 + 1, padding='same', groups=filters_n)
        self.softplus = torch.nn.Softplus()
        self.maxpool = torch.nn.MaxPool1d(self.pooling_factor, return_indices=True, ceil_mode=True)

        # Bi-LSTM, learning temporal dependencies
        self.lstm_hidden_size_1 = 64
        self.lstm_hidden_size_2 = 32
        self.lstm_hidden_size_3 = 16
        self.lstm_dropout = 0.1

        self.bi_lstm_1 = torch.nn.LSTM(self.filters_n, self.lstm_hidden_size_1, bidirectional=True)
        self.bi_lstm_2 = torch.nn.LSTM(2 * self.lstm_hidden_size_1, self.lstm_hidden_size_2, bidirectional=True)
        self.bi_lstm_3 = torch.nn.LSTM(2 * self.lstm_hidden_size_2, self.lstm_hidden_size_3, bidirectional=True)
        self.dropout = torch.nn.Dropout(self.lstm_dropout)

        # SAAF
        self.breakpoints_n = 25
        self.breakpoints = [-1.0 + 2.0 / self.breakpoints_n * k for k in range(self.breakpoints_n + 1)]
        self.saaf = SAAFc2(self.breakpoints)

        # decoder, synthesis back-end
        self.unpool = torch.nn.MaxUnpool1d(self.pooling_factor)

        # SE block
        self.se_hidden_size_1 = 512
        self.se_hidden_size_2 = 32
        self.se_block = SE(filters_n, self.se_hidden_size_1, self.se_hidden_size_2)

        # DNN-SAAF-SE
        self.dnn_hidden_size_1 = 32
        self.dnn_hidden_size_2 = 16
        self.dnn_hidden_size_3 = 16
        self.dnn_hidden_size_4 = 32
        self.dnn_saaf = torch.nn.Sequential(
            torch.nn.Linear(filters_n, self.dnn_hidden_size_1),
            torch.nn.Tanh(),
            torch.nn.Linear(self.dnn_hidden_size_1, self.dnn_hidden_size_2),
            torch.nn.Tanh(),
            torch.nn.Linear(self.dnn_hidden_size_2, self.dnn_hidden_size_3),
            torch.nn.Tanh(),
            torch.nn.Linear(self.dnn_hidden_size_3, self.dnn_hidden_size_4),
            self.saaf,
        )


    def forward(self, x):
        output = self.conv(x)
        residual_1 = output[self.frames_lookahead, :, :]
        output = torch.abs(output)
        output = self.softplus(self.conv_local(output))
        output, indices = self.maxpool(output)
        indices = indices[self.frames_lookahead, :, :]

        output, _ = self.bi_lstm_1(output.transpose(1, 2))
        output = self.dropout(output)
        output, _ = self.bi_lstm_2(output)
        output = self.dropout(output)
        output, _ = self.bi_lstm_3(output)
        output = output[self.frames_lookahead, :, :].transpose(0, 1)
        output = self.saaf(output)

        output = self.unpool(output, indices)
        residual_2 = residual_1 * output
        output = self.dnn_saaf(residual_2.transpose(0, 1))
        output = torch.abs(output.transpose(0, 1))
        output = self.se_block(output)
        output = (residual_2.transpose(0, 1) * output).transpose(0, 1)
        output = output + residual_2
        output = torch.nn.functional.conv_transpose1d(output, self.conv.weight, padding=self.padding)

        return output

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