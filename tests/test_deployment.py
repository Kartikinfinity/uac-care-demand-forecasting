"""
tests/test_deployment.py -- Part 11 Deployment Tests, and the Part 11
Application Tests that can be asserted without a browser.

Two jobs:

  1. The repository must be deployable. No absolute paths, no machine-specific
     dependencies, every path resolving from the repository root, and the
     artifacts the deployed app needs actually present in the commit.

  2. Every control combination must produce a usable page. The roadmap requires
     "model selector changes the displayed chart/metrics correctly, for every
     option" and "edge cases explicitly checked: shortest horizon, longest
     horizon, every model option, every page". Those are exercised here against
     the real artifacts -- 48 model/horizon combinations per target -- because a
     manual click-through cannot realistically cover them all and would stop
     being repeated anyway.
"""
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import (
    FORECAST_HORIZONS,
    FORECAST_PROVENANCE_PATH,
    STATIC_PROVENANCE_PATH,
    TARGET_1,
    TARGET_2,
)

APP = project_root / "app"
REQUIRED_FOR_DEPLOY = [
    "forecasts/forward_forecasts.csv",
    "forecasts/interval_coverage.csv",
    "forecasts/holdout_evaluation.csv",
    "forecasts/imbalance_forecast.csv",
    "forecasts/early_warning_backtest.csv",
    "forecasts/early_warning_sensitivity.csv",
    "forecasts/kpi_summary.csv",
    "forecasts/champion_selection.csv",
    "forecasts/full_model_comparison.csv",
    "forecasts/provenance.json",
    "data/interim/master_series.parquet",
    "data/interim/provenance.json",
]


def _tracked_by_git(relative: str) -> bool:
    """True when git would include the file in a clone."""
    result = subprocess.run(
        ["git", "check-ignore", relative.replace("/", "\\")],
        cwd=project_root, capture_output=True, text=True,
    )
    return result.returncode != 0


# ======================================================================
# DEPLOYABILITY
# ======================================================================
def test_no_absolute_local_paths_anywhere_in_the_codebase():
    """
    A path that resolves on the development machine and nowhere else is the
    classic reason a working app dies on deploy.
    """
    pattern = re.compile(r"""["'](?:[A-Za-z]:[\\/]|/home/|/Users/)""")
    offenders = []
    for path in list((project_root / "src").rglob("*.py")) + \
                list(APP.rglob("*.py")) + list((project_root / "scripts").rglob("*.py")):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line) and "example" not in line.lower():
                offenders.append("%s:%d" % (path.relative_to(project_root), i))
    assert not offenders, "absolute local paths found: %s" % offenders


def test_every_path_resolves_from_the_repository_root():
    """
    Paths must be built from `PROJECT_ROOT`, never from the process working
    directory -- Streamlit Cloud does not run the app from the repo root.
    """
    config = (project_root / "src" / "config.py").read_text(encoding="utf-8")
    assert "PROJECT_ROOT = Path(__file__).resolve().parent.parent" in config
    for name in ("FORECASTS_DIR", "DATA_INTERIM_DIR", "MODELS_DIR", "DOCS_DIR"):
        assert name in config


def test_the_app_entrypoint_does_not_depend_on_the_working_directory():
    """`streamlit run app/Home.py` must work from any cwd."""
    home = (APP / "Home.py").read_text(encoding="utf-8")
    assert "Path(__file__).resolve().parent.parent" in home
    assert "sys.path.insert" in home


def test_every_artifact_the_deployed_app_needs_is_committed():
    """
    The dashboard reads pre-generated files and Streamlit Cloud has no build
    step that could produce them. If these are gitignored, a fresh clone
    deploys an app with nothing to serve.
    """
    missing, ignored = [], []
    for relative in REQUIRED_FOR_DEPLOY:
        path = project_root / relative
        if not path.exists():
            missing.append(relative)
        elif not _tracked_by_git(relative):
            ignored.append(relative)
    assert not missing, "artifacts absent: %s" % missing
    assert not ignored, "artifacts exist but are gitignored: %s" % ignored


def test_large_intermediates_stay_out_of_the_repository():
    """
    The flip side: multi-megabyte files the dashboard never reads must NOT be
    committed just because the allow-list was written carelessly.
    """
    for relative in ("forecasts/ml_predictions.csv", "forecasts/oos_residuals.csv",
                     "forecasts/statistical_predictions.csv",
                     "forecasts/baseline_predictions.csv",
                     "data/processed/features_target1.parquet"):
        if (project_root / relative).exists():
            assert not _tracked_by_git(relative), (
                "%s is a regenerable intermediate and should not be committed" % relative
            )


def test_requirements_cover_every_third_party_import():
    """
    Every package imported by shipped code must be declared. This project has
    already been bitten three times -- holidays, joblib and tornado were each
    imported but undeclared, and each produced a bare ModuleNotFoundError.
    """
    declared = set()
    for line in (project_root / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            declared.add(re.split(r"[><=\[;#\s]", line)[0].lower().replace("-", "_"))

    stdlib = set(sys.stdlib_module_names)
    local = {"src", "app", "scripts", "tests"}
    aliases = {"sklearn": "scikit_learn", "dateutil": "python_dateutil",
               "yaml": "pyyaml", "PIL": "pillow"}

    undeclared = set()
    for path in list((project_root / "src").rglob("*.py")) + list(APP.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            for name in names:
                root = name.split(".")[0]
                if root in stdlib or root in local:
                    continue
                key = aliases.get(root, root).lower().replace("-", "_")
                if key not in declared:
                    undeclared.add(root)
    assert not undeclared, "imported but not in requirements.txt: %s" % sorted(undeclared)


def test_streamlit_config_enables_static_serving_for_the_smoke_test():
    config = (project_root / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    assert "enableStaticServing = true" in config
    assert "gatherUsageStats = false" in config


# ======================================================================
# PROVENANCE -- the addendum's Day-12 extension
# ======================================================================
def test_committed_artifacts_describe_the_committed_data():
    """
    A stale committed artifact would be served as current without erroring.
    Its recorded hashes must match the data actually in the repository.
    """
    from src.data.validate import read_provenance

    data = read_provenance()
    forecast = json.loads(FORECAST_PROVENANCE_PATH.read_text(encoding="utf-8"))
    assert forecast["raw_csv_sha256"] == data["raw_csv_sha256"]
    assert forecast["master_series_sha256"] == data["master_series_sha256"]
    assert forecast["data_as_of"] == data["data_as_of"]


def test_the_static_sidecar_is_a_faithful_copy():
    """What a deployed app serves must equal what the pipeline wrote."""
    assert STATIC_PROVENANCE_PATH.exists(), "run `python -m src.forecast.generate`"
    assert (STATIC_PROVENANCE_PATH.read_text(encoding="utf-8")
            == FORECAST_PROVENANCE_PATH.read_text(encoding="utf-8"))


def test_the_provenance_check_actually_has_teeth(tmp_path, monkeypatch):
    """
    A smoke test that cannot fail is theatre. Tamper with the hash and confirm
    the comparison rejects it.
    """
    import scripts.smoke_test as smoke

    local = json.loads(FORECAST_PROVENANCE_PATH.read_text(encoding="utf-8"))
    tampered = dict(local, raw_csv_sha256="0" * 64)

    def fake_get(url, timeout=20):
        return 200, json.dumps(tampered).encode("utf-8")

    monkeypatch.setattr(smoke, "_get", fake_get)
    result = smoke.Result()
    smoke.check_provenance_match(result, "http://example.invalid")
    failures = [name for name, ok, _ in result.checks if not ok]
    assert any("raw_csv_sha256" in name for name in failures), (
        "a tampered hash was accepted"
    )


def test_smoke_test_local_checks_pass_on_this_repository():
    import scripts.smoke_test as smoke

    result = smoke.Result()
    smoke.check_local_consistency(result)
    assert not result.failed, [c[0] for c in result.failed]


# ======================================================================
# APPLICATION TESTS -- every control combination, not a sample
# ======================================================================
@pytest.fixture(scope="module")
def artifacts():
    from app.lib.artifacts import load_coverage, load_forward, load_history

    if not (project_root / "forecasts" / "forward_forecasts.csv").exists():
        pytest.skip("run `python -m src.forecast.generate` first")
    return (pd.read_parquet(project_root / "data/interim/master_series.parquet"),
            pd.read_csv(project_root / "forecasts/forward_forecasts.csv"),
            pd.read_csv(project_root / "forecasts/interval_coverage.csv"))


def test_every_model_and_horizon_combination_renders_usable_numbers(artifacts):
    """
    Part 11: "model selector changes the displayed chart/metrics correctly, for
    every option" and "edge cases explicitly checked: shortest horizon, longest
    horizon, every model option". All 48 combinations, not a spot check.
    """
    from app.lib.artifacts import available_models, forecast_row

    history, forward, _ = artifacts
    checked = 0
    for target in (TARGET_1, TARGET_2):
        for horizon in FORECAST_HORIZONS:
            for model in available_models(forward, target, horizon):
                row = forecast_row(forward, target, model, horizon)
                assert row is not None, "%s/%s/h%d missing" % (target, model, horizon)
                assert np.isfinite(row["point_forecast"]), (
                    "%s/%s/h%d has no point forecast" % (target, model, horizon))
                assert row["point_forecast"] >= 0, "negative count forecast"
                if bool(row["interval_sufficient"]):
                    assert row["lower"] <= row["point_forecast"] <= row["upper"]
                    assert row["lower"] >= 0
                checked += 1
    assert checked >= 48, "expected at least 48 combinations, checked %d" % checked


def test_changing_the_model_actually_changes_the_output(artifacts):
    """A selector that renders the same numbers for every option is broken."""
    from app.lib.artifacts import available_models, forecast_row

    _, forward, _ = artifacts
    for target in (TARGET_1, TARGET_2):
        for horizon in FORECAST_HORIZONS:
            values = {
                round(float(forecast_row(forward, target, m, horizon)["point_forecast"]), 6)
                for m in available_models(forward, target, horizon)
            }
            assert len(values) > 1, (
                "every model returns an identical forecast for %s h=%d" % (target, horizon))


def test_changing_the_horizon_actually_changes_the_output(artifacts):
    """Interval width must grow with horizon; a flat band signals a wiring bug."""
    from app.lib.artifacts import champion_for, forecast_row

    _, forward, _ = artifacts
    for target in (TARGET_1, TARGET_2):
        widths = []
        for horizon in sorted(FORECAST_HORIZONS):
            row = forecast_row(forward, target, champion_for(forward, target, horizon), horizon)
            if bool(row["interval_sufficient"]):
                widths.append(float(row["interval_width"]))
        assert len(widths) >= 2
        assert widths[-1] > widths[0], "interval does not widen with horizon for %s" % target


def test_every_forecast_date_lands_on_a_reporting_day(artifacts):
    """No control combination may place a forecast on a Friday or Saturday."""
    from app.lib.forecast_page import _forecast_date

    history, forward, _ = artifacts
    history["parsed_date"] = pd.to_datetime(history["parsed_date"])
    origin = int(forward["origin_pos"].iloc[0])
    for horizon in FORECAST_HORIZONS:
        assert _forecast_date(history, origin, horizon).dayofweek not in (4, 5)


def test_no_page_can_render_an_empty_chart(artifacts):
    """
    Part 11: "all charts render with valid, non-empty data -- no blank chart
    states." Each chart's underlying series must be non-empty and finite.
    """
    history, forward, _ = artifacts
    for column in (TARGET_1, TARGET_2):
        series = history[column].astype(float)
        assert len(series) > 0 and np.isfinite(series).any()
    assert len(forward) > 0
    assert forward.groupby(["target", "horizon"]).size().min() >= 8


def test_nullable_boolean_is_used_where_a_column_can_be_blank():
    """
    REGRESSION, found by the clean-environment deployment test. The Model
    Comparison page blanks the "beats both baselines" cell for the baselines
    themselves. pandas 2 silently upcast a plain bool column when pd.NA was
    assigned; pandas 3 raises "TypeError: Invalid value 'nan' for dtype 'bool'"
    and the page died on load. Development ran pandas 2.3; a fresh install
    resolves pandas 3.0, which is what a deployment gets.
    """
    source = (APP / "pages" / "5_Model_Comparison.py").read_text(encoding="utf-8")
    assert 'dtype="boolean"' in source, (
        "a column that receives pd.NA must be declared nullable boolean"
    )


def test_assigning_na_into_the_comparison_column_works_on_this_pandas():
    """The behaviour itself, not just the source pattern."""
    frame = pd.DataFrame({"model": ["naive", "sarima"], "MAE": [10.0, 8.0]})
    frame["beats"] = pd.array(frame["MAE"] < 9.0, dtype="boolean")
    frame.loc[frame["model"] == "naive", "beats"] = pd.NA
    assert pd.isna(frame.loc[0, "beats"])
    assert bool(frame.loc[1, "beats"]) is True


def test_a_documentation_placeholder_url_is_named_as_such():
    """
    Pasting the README's `https://<your-app>.streamlit.app` verbatim produced a
    bare "getaddrinfo failed", which reads like the deployment is broken rather
    than like the URL was never filled in. Reported for real.
    """
    import scripts.smoke_test as smoke

    assert smoke.looks_like_a_placeholder("https://<your-app>.streamlit.app")
    assert smoke.looks_like_a_placeholder("https://your-app.streamlit.app")
    assert not smoke.looks_like_a_placeholder("https://uac-forecasting.streamlit.app")
    assert not smoke.looks_like_a_placeholder("http://localhost:8501")

    result = smoke.Result()
    ok = smoke.check_reachable(result, "https://<your-app>.streamlit.app")
    assert ok is False
    detail = result.checks[-1][2]
    assert "placeholder" in detail, detail


def test_placeholder_detection_does_not_make_a_network_call(monkeypatch):
    """It must fail fast on the URL itself, not wait for a DNS timeout."""
    import scripts.smoke_test as smoke

    def explode(*args, **kwargs):
        raise AssertionError("network call attempted for a placeholder URL")

    monkeypatch.setattr(smoke, "_get", explode)
    result = smoke.Result()
    smoke.check_reachable(result, "https://<your-app>.streamlit.app")
    assert result.failed
