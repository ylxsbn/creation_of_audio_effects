import torch


class SE(torch.nn.Module):
    """
    Squeeze and excitation block: https://arxiv.org/pdf/1709.01507
    """

    def __init__(self, input_size, hidden_size_1, hidden_size_2):
        super().__init__()

        self.global_avg_pool = torch.nn.AdaptiveAvgPool1d(output_size=1)
        self.linear_1 = torch.nn.Linear(input_size, hidden_size_1)
        self.relu_act = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(hidden_size_1, hidden_size_2)
        self.sig_act = torch.nn.Sigmoid()

    def forward(self, x):
        input = torch.abs(x)
        output = self.global_avg_pool(input)
        output = output.transpose(-1, -2)
        output = self.linear_1(output)
        output = self.relu_act(output)
        output = self.linear_2(output)
        output = self.sig_act(output)
        output = output.transpose(-1, -2)
        output = x * output

        return output
