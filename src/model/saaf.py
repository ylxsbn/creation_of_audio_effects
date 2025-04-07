import torch


class SAAFc2(torch.nn.Module):
    """
    Smooth adaptive activation function with c=2: https://arxiv.org/pdf/1608.06557
    """

    def __init__(self, breakpoints: list, channels_n):
        super().__init__()

        self.n = len(breakpoints) - 1
        self.breakpoints = breakpoints

        self.v0 = torch.nn.Parameter(torch.randn((1, channels_n)))
        self.v1 = torch.nn.Parameter(torch.randn((1, channels_n)))
        self.w = torch.nn.Parameter(torch.randn(self.n, channels_n))

    def forward(self, x):
        res = self.v1 * x
        for k in range(self.n):
            a = self.breakpoints[k]
            b = self.breakpoints[k + 1]
            res += self.w[k, :] * torch.where(x < a, 0, torch.where(x
                                              >= b, (b - a)**2 / 2 + (b - a) * (x - b), (x - a)**2 / 2))

        return res
