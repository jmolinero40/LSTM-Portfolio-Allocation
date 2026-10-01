"""Experiment configuration loaded from YAML.

Every number that shapes a result lives in a file under ``configs/``, never in
the source. The thesis code hard-coded parameters inside functions (``tau=6.0``
inside ``sharpe_like_model``, ``k = 5`` in the backtest script, ``span=20`` in
the baseline) and they drifted between scripts and the written thesis. A single
typed config, copied next to every run, makes that impossible.

Unknown keys raise an error instead of being silently ignored, so a typo in a
YAML file cannot quietly fall back to a default.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "DataConfig",
    "FeatureConfig",
    "WalkForwardConfig",
    "NetConfig",
    "PortfolioConfig",
    "EvaluationConfig",
    "OutputConfig",
    "ExperimentConfig",
    "load_config",
]


def _build(cls, payload: dict[str, Any] | None):
    payload = payload or {}
    known = {f.name for f in fields(cls)}
    unknown = set(payload) - known
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}. Valid: {sorted(known)}")
    return cls(**payload)


@dataclass
class DataConfig:
    snapshot: str = "data/snapshot/prices_adjclose.csv"
    sha256: str | None = None  # expected hash of the snapshot; null skips the check
    start: str = "2013-01-01"
    end: str = "2025-12-31"  # inclusive; fixed so that a rerun sees the same sample


@dataclass
class FeatureConfig:
    winsor_lower: float = 0.005
    winsor_upper: float = 0.995
    ema_span: int = 20  # baseline subtracted from the return target (thesis App. G)


@dataclass
class WalkForwardConfig:
    test_years: list[int] = field(default_factory=lambda: list(range(2017, 2026)))
    val_fraction: float = 0.15  # tail of the pre-test sample used for early stopping
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])


@dataclass
class NetConfig:
    lookback: int = 60
    units: list[int] = field(default_factory=lambda: [64, 32])
    dropout: float = 0.2
    l2: float = 1e-5
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 200
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 7
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-5
    shuffle: bool = False  # the thesis trained without shuffling; kept for fidelity
    huber_delta: float = 1.0  # return network only
    init_output_bias: bool = False  # risk network: start at the training variance


@dataclass
class PortfolioConfig:
    k: int = 5
    tau: float = 6.0
    w_max: float = 0.3
    target_annual_vol: float = 0.15
    corr_window: int = 60
    corr_lambda: float = 0.97
    corr_shrinkage: float = 0.3
    ewma_sigma_lambda: float = 0.97  # for the EWMA volatility baseline


@dataclass
class EvaluationConfig:
    cost_bps: list[float] = field(default_factory=lambda: [0.0, 5.0, 10.0])
    var_alpha: float = 0.01
    bootstrap_block: int = 20
    bootstrap_samples: int = 5000
    bootstrap_seed: int = 0


@dataclass
class OutputConfig:
    runs_dir: str = "runs/walkforward"  # per-fold checkpoints, not tracked
    results_dir: str = "results/walkforward"  # tracked: predictions, metrics, tables


@dataclass
class ExperimentConfig:
    name: str = "walkforward"
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    walkforward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    return_model: NetConfig = field(default_factory=NetConfig)
    risk_model: NetConfig = field(default_factory=lambda: NetConfig(lookback=240))
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperimentConfig:
        d = dict(d)
        sections = {
            "data": DataConfig,
            "features": FeatureConfig,
            "walkforward": WalkForwardConfig,
            "return_model": NetConfig,
            "risk_model": NetConfig,
            "portfolio": PortfolioConfig,
            "evaluation": EvaluationConfig,
            "output": OutputConfig,
        }
        unknown = set(d) - set(sections) - {"name"}
        if unknown:
            raise ValueError(f"Unknown top-level keys: {sorted(unknown)}")
        kwargs: dict[str, Any] = {"name": d.get("name", "walkforward")}
        for key, cls_ in sections.items():
            kwargs[key] = _build(cls_, d.get(key))
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path, encoding="utf-8") as f:
        return ExperimentConfig.from_dict(yaml.safe_load(f) or {})
