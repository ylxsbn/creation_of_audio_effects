import torch
from src.model import SE
from src.model import SAAFc2



class Model(torch.nn.Module):
    def __init__(self, channels_n=1, frame_size=4096, frames_lookahead=4):
        super().__init__()

        self.channels_n = channels_n
        self.filters_n = 32
        self.kernel_size = 63
        self.frame_size = frame_size
        self.frames_lookahead = frames_lookahead

        self.padding = self.kernel_size // 2
        self.frames_n = 2 * frames_lookahead + 1
        self.pooling_factor = self.frame_size // 64

        # convolution encoder, adaptive front-end
        self.conv = torch.nn.Conv1d(self.channels_n, self.filters_n, self.kernel_size, padding=self.padding)
        self.conv_local = torch.nn.Conv1d(self.filters_n, self.filters_n, self.kernel_size * 2 + 1, padding='same', groups=self.filters_n)
        self.softplus = torch.nn.Softplus()
        self.maxpool = torch.nn.MaxPool1d(self.pooling_factor, return_indices=True, ceil_mode=True)

        # Bi-LSTM, learning temporal dependencies
        self.lstm_hidden_size_1 = 64
        self.lstm_hidden_size_2 = 32
        self.lstm_hidden_size_3 = 16
        self.lstm_dropout = 0.1

        self.bi_lstm_1 = torch.nn.LSTM(self.filters_n, self.lstm_hidden_size_1, bidirectional=True, batch_first=True)
        self.bi_lstm_2 = torch.nn.LSTM(2 * self.lstm_hidden_size_1, self.lstm_hidden_size_2, bidirectional=True, batch_first=True)
        self.bi_lstm_3 = torch.nn.LSTM(2 * self.lstm_hidden_size_2, self.lstm_hidden_size_3, bidirectional=True, batch_first=True)
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
        self.se_block = SE(self.filters_n, self.se_hidden_size_1, self.se_hidden_size_2)

        # DNN-SAAF-SE
        self.dnn_hidden_size_1 = 32
        self.dnn_hidden_size_2 = 16
        self.dnn_hidden_size_3 = 16
        self.dnn_hidden_size_4 = 32
        self.dnn_saaf = torch.nn.Sequential(
            torch.nn.Linear(self.filters_n, self.dnn_hidden_size_1),
            torch.nn.Tanh(),
            torch.nn.Linear(self.dnn_hidden_size_1, self.dnn_hidden_size_2),
            torch.nn.Tanh(),
            torch.nn.Linear(self.dnn_hidden_size_2, self.dnn_hidden_size_3),
            torch.nn.Tanh(),
            torch.nn.Linear(self.dnn_hidden_size_3, self.dnn_hidden_size_4),
            self.saaf,
        )


    def forward(self, input_audio, **batch):
        """
        input must has shape (batch_size, frames_n, channels_n, frame_size)
        makes the prediction of the intermediate frame so output has shape (batch_size, channels_n, frame_size)
        """

        batch_size, _, _, _ = input_audio.shape
        input_audio = input_audio.reshape(batch_size * self.frames_n, self.channels_n, self.frame_size)

        output = self.conv(input_audio)

        output = output.reshape(batch_size, self.frames_n, -1, self.frame_size)
        residual_1 = output[:, self.frames_lookahead, :, :]
        output = output.reshape(batch_size * self.frames_n, -1, self.frame_size)

        output = torch.abs(output)
        output = self.softplus(self.conv_local(output))
        output, indices = self.maxpool(output)
        output = output.reshape(batch_size, self.frames_n, output.size(-2), output.size(-1))

        indices = indices.reshape(batch_size, self.frames_n, indices.size(-2), indices.size(-1))
        indices = indices[:, self.frames_lookahead, :, :]

        output = output.permute(0, 3, 1, 2)
        output = output.reshape(output.size(0) * output.size(1), output.size(2), output.size(3))
        
        output, _ = self.bi_lstm_1(output)
        output = self.dropout(output)

        output, _ = self.bi_lstm_2(output)
        output = self.dropout(output)

        output, _ = self.bi_lstm_3(output)
        output = output[:, self.frames_lookahead, :]

        output = self.saaf(output)

        output = output.reshape(batch_size, -1, output.size(-1))
        output = output.transpose(1, 2)

        output = self.unpool(output, indices)

        residual_2 = residual_1 * output
        output = residual_2

        output = output.transpose(1, 2)
        output = output.reshape(batch_size * self.frame_size, -1)

        output = self.dnn_saaf(output)

        output = torch.abs(output)

        output = output.reshape(batch_size, self.frame_size, -1)
        output = output.transpose(-1, -2)

        output = self.se_block(output)

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