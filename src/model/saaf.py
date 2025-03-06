import torch


class SAAFc2(torch.nn.Module):
    """
    Smooth adaptive activation function with c=2
    """
    def __init__(self, breakpoints: list):
        super().__init__()

        self.n = len(breakpoints) - 1
        self.breakpoints = breakpoints

        self.v0 = torch.nn.Parameter(torch.randn(()))
        self.v1 = torch.nn.Parameter(torch.randn(()))
        self.w = torch.nn.Parameter(torch.randn(self.n))
        
    
    def forward(self, x):
        res = self.v0 + self.v1 * x
        for k in range(self.n):
            a = self.breakpoints[k]
            b = self.breakpoints[k + 1]
            res += self.w[k] * torch.where(x < a, 0, torch.where(x >= b, (b**2 - a**2) / 2 + (x - b), (x**2 - a**2) / 2))
        return res
