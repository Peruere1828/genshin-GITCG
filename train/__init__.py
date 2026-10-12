"""WS3 SoG/ReBeL training kernel (toy through full).

- model: toy CVPN (policy + value + optional S-C belief head)
- data: teacher-labelled replay collection (BC + search-teacher distillation)
- train_loop: behavior cloning
- distill: search-distillation (policy CE + value regression + belief scoring)
- pipeline: one collect->train->eval->gate round
"""

from .data import Sample, collect_samples, collect_search_samples, sample_stats
from .distill import DistillConfig, DistillMetrics, train_distill
from .model import CVPN, CVPNConfig, collate, config_for_encoder
from .train_loop import TrainConfig, TrainMetrics, train_bc

__all__ = [
    "Sample",
    "collect_samples",
    "collect_search_samples",
    "sample_stats",
    "CVPN",
    "CVPNConfig",
    "collate",
    "config_for_encoder",
    "TrainConfig",
    "TrainMetrics",
    "train_bc",
    "DistillConfig",
    "DistillMetrics",
    "train_distill",
]
