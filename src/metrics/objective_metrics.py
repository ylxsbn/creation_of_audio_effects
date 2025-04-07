# Metrics are based on: https://github.com/adobe-research/DeepAFx-ST/blob/main/deepafx_st/metrics.py
# And: https://github.com/ytsrt66589/pyneuralfx/blob/main/src/pyneuralfx/eval_metrics/eval_metrics.py

import torch
from src.metrics.base_metric import BaseMetric

from src.metrics.metric_helpers import calc_spectral_centroid, calc_crest_factor, calc_rms_energy, calc_energy_normalized_mae


class EnergyNormalizedMAE(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, predicted_audio: torch.Tensor, output_audio: torch.Tensor, **kwargs):
        return calc_energy_normalized_mae(predicted_audio, output_audio)


class SpectralCentroid(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, predicted_audio: torch.Tensor, output_audio: torch.Tensor, **kwargs):
        return torch.nn.functional.l1_loss(calc_spectral_centroid(predicted_audio), calc_spectral_centroid(output_audio))


class CrestFactor(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, predicted_audio: torch.Tensor, output_audio: torch.Tensor, **kwargs):
        return torch.nn.functional.l1_loss(calc_crest_factor(predicted_audio), calc_crest_factor(output_audio))


class RMSEnergy(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, predicted_audio: torch.Tensor, output_audio: torch.Tensor, **kwargs):
        return torch.nn.functional.l1_loss(calc_rms_energy(predicted_audio), calc_rms_energy(output_audio))
