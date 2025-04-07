from src.metrics.base_metric import BaseMetric


class Plug(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, **batch):
        return 0.0