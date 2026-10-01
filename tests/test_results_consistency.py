"""The numbers in the README are exactly what the code produces.

Stage 2 (forecasts -> portfolios -> tables) is deterministic. This test reruns
it from the committed forecasts and the committed price snapshot, and checks
that (1) the metrics match the committed ``results/walkforward/metrics.csv``
and (2) every result table in README.md is, character for character, the one
the code writes. Editing a number in the README by hand, or changing the code
without regenerating the results, fails CI.
"""

from __future__ import annotations

import re
import shutil

import pandas as pd
import pytest

from lstm_portfolio.config import load_config
from lstm_portfolio.report import run_report


@pytest.fixture(scope="module")
def regenerated(tmp_path_factory):
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "walkforward.yaml")
    committed = root / cfg.output.results_dir
    if not (committed / "predictions.parquet").exists():
        pytest.skip("no committed walk-forward forecasts")
    out = tmp_path_factory.mktemp("results")
    shutil.copy(committed / "predictions.parquet", out / "predictions.parquet")
    cfg.data.snapshot = str(root / cfg.data.snapshot)
    cfg.output.results_dir = str(out)
    tables = run_report(cfg, figures=False)
    return root, committed, out, tables


@pytest.mark.slow
def test_metrics_match_committed(regenerated):
    _, committed, out, _ = regenerated
    a = pd.read_csv(committed / "metrics.csv")
    b = pd.read_csv(out / "metrics.csv")
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-8, atol=1e-12)


@pytest.mark.slow
def test_readme_tables_are_generated(regenerated):
    root, committed, _, tables = regenerated
    readme = (root / "README.md").read_text(encoding="utf-8").replace("\r\n", "\n")
    for name in ("headline", "main", "significance", "forecasts"):
        m = re.search(rf"<!-- results:{name}:start -->\n(.*?)<!-- results:{name}:end -->", readme, re.S)
        assert m, f"README block '{name}' missing"
        assert m.group(1) == tables[name], f"README table '{name}' differs from the code's output"
        assert (committed / "tables" / f"{name}.md").read_text(encoding="utf-8").replace(
            "\r\n", "\n"
        ) == tables[name]
