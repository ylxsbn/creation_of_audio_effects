import torch


def calc_energy_normalized_mae(pred, target):
    pred = pred / pred.abs().mean()

    target_energy = target.abs().mean()
    target = target / target_energy

    return torch.nn.functional.l1_loss(pred, target) * target_energy


def calc_spectral_centroid(x):
    spectrum = torch.fft.rfft(x).abs()
    norm_spectrum = spectrum / spectrum.sum()
    norm_freq = torch.linspace(0, 1, spectrum.size(-1))
    spectral_centroid = torch.sum(norm_freq * norm_spectrum)
    return spectral_centroid


def calc_crest_factor(x, amp=20, eps=1e-8):
    peak, _ = x.abs().max(dim=-1)
    rms = torch.sqrt((x**2).mean(dim=-1))
    return amp * torch.log(peak / rms.clamp(eps))


def calc_rms_energy(x, amp=20, eps=1e-8):
    rms = torch.sqrt((x**2).mean(dim=-1))
    return amp * torch.log(rms.clamp(eps))
