"""WS3 SoG/ReBeL training kernel (toy through full).

- model: toy CVPN (policy + value)
- data: teacher-labelled replay collection
- train_loop: behavior cloning / distillation
- pipeline: one collect->train->eval->gate round
"""

from .data import Sample, collect_samples, sample_stats
from .model import CVPN, CVPNConfig, collate, config_for_encoder
from .train_loop import TrainConfig, TrainMetrics, train_bc

__all__ = [
    "Sample",
    "collect_samples",
    "sample_stats",
    "CVPN",
    "CVPNConfig",
    "collate",
    "config_for_encoder",
    "TrainConfig",
    "TrainMetrics",
    "train_bc",
]
