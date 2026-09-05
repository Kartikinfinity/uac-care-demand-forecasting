"""
tests/test_app.py -- The dashboard's data layer and page contracts.

The dashboard is a pure consumer of pre-generated artifacts. These tests pin
that boundary: the app layer must not import a model, must not fit anything,
and must surface the calibration caveats rather than showing a bare interval.

Streamlit's own rendering is not unit-tested here -- that was verified by
driving the running app in a browser (Day-10 validation checkpoint). What IS
tested is every pure function the pages depend on, because those are where a
silent mismatch between artifact and display would hide.
"""
import ast
import sys
from pathlib import Path

import pandas as pd
import pytest

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import (
    FORECAST_HORIZONS,
    FORWARD_FORECASTS_PATH,
    INTERVAL_COVERAGE_PATH,
    TARGET_1,
    TARGET_2,
)

APP = project_root / "app"


# ======================================================================
# THE ARCHITECTURAL BOUNDARY
# ======================================================================
def _imported_modules(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_no_app_file_imports_a_model_or_training_module():
    """
    Roadmap Part 9: the app "never trains or refits a model as part of serving a
    page". The cheapest way for that to be violated later is an innocent-looking
    import, so the import graph itself is asserted.
    """
    forbidden_prefixes = ("src.models", "src.forecast", "src.evaluation.run_",
                          "sklearn", "statsmodels")
    offenders = {}
    for path in APP.rglob("*.py"):
        bad = [m for m in _imported_modules(path)
               if any(m.startswith(p) for p in forbidden_prefixes)]
        if bad:
            offenders[path.name] = bad
    assert not offenders, "app imports training code: %s" % offenders


def test_no_app_file_calls_fit():
    """A blunt but effective guard against a model being fitted in a page."""
    for path in APP.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in ("fit", "predict"), (
                    "%s calls .%s() -- the dashboard must read artifacts, not "
                    "compute forecasts" % (path.name, node.func.attr)
                )


def test_both_mandated_core_module_pages_exist():
    """Part 8 pages 3 and 4 are documented Core Modules, not optional."""
    assert (APP / "pages" / "2_Care_Load_Forecast.py").exists()
    assert (APP / "pages" / "3_Discharge_Demand_Forecast.py").exists()
    assert (APP / "Home.py").exists()


def test_the_two_forecast_pages_share_one_renderer():
    """
    Two hand-maintained copies of the same chart drift. A reader comparing care
    load against discharge demand must be seeing a difference in the data, not
    in two slightly different renderers.
    """
    for name in ("2_Care_Load_Forecast.py", "3_Discharge_Demand_Forecast.py"):
        source = (APP / "pages" / name).read_text(encoding="utf-8")
        assert "from app.lib.forecast_page import render" in source
        assert "plotly" not in source, "%s builds its own chart" % name


# ======================================================================
# DERIVED VIEWS THE PAGES DEPEND ON
# ======================================================================
@pytest.fixture(scope="module")
def forward():
    if not FORWARD_FORECASTS_PATH.exists():
        pytest.skip("run `python -m src.forecast.generate` first")
    return pd.read_csv(FORWARD_FORECASTS_PATH)


@pytest.fixture(scope="module")
def coverage():
    if not INTERVAL_COVERAGE_PATH.exists():
        pytest.skip("run `python -m src.forecast.generate` first")
    return pd.read_csv(INTERVAL_COVERAGE_PATH)


def test_horizon_labels_always_carry_the_calendar_equivalent():
    """
    Addendum Day 10-11: horizon labels show calendar-day equivalents alongside
    period counts. The programme reports Sun-Thu, so 7 periods is ~9 days -- a
    reader planning staffing in days would be misled by the period count alone.
    """
    from app.lib.artifacts import horizon_label

    for horizon in FORECAST_HORIZONS:
        label = horizon_label(horizon)
        assert str(horizon) in label
        assert "day" in label, "no calendar equivalent in %r" % label
    assert "9 days" in horizon_label(7), "7 periods must not read as 7 days"
    assert "20 days" in horizon_label(14)


def test_the_model_toggle_offers_every_model_champion_first(forward):
    from app.lib.artifacts import available_models, champion_for

    for target in (TARGET_1, TARGET_2):
        for horizon in FORECAST_HORIZONS:
            models = available_models(forward, target, horizon)
            assert len(models) >= 8, "toggle is missing models"
            assert models[0] == champion_for(forward, target, horizon), (
                "champion is not offered first"
            )
            assert len(models) == len(set(models))


def test_every_offered_model_actually_has_a_forecast_row(forward):
    """The toggle must never offer an option that renders an error."""
    from app.lib.artifacts import available_models, forecast_row

    for target in (TARGET_1, TARGET_2):
        for horizon in FORECAST_HORIZONS:
            for model in available_models(forward, target, horizon):
                assert forecast_row(forward, target, model, horizon) is not None


def test_a_calibration_note_is_produced_for_every_champion_cell(forward, coverage):
    """An interval must never be shown without its measured calibration."""
    from app.lib.artifacts import champion_for, coverage_note

    for target in (TARGET_1, TARGET_2):
        for horizon in FORECAST_HORIZONS:
            champion = champion_for(forward, target, horizon)
            calibrated, message = coverage_note(coverage, target, champion, horizon)
            assert calibrated in (True, False), "no calibration evidence for a champion"
            assert "coverage" in message.lower()
            assert "n=" in message


def test_a_failed_calibration_is_stated_plainly_not_softened(forward, coverage):
    """
    Two of six cells genuinely miss their binomial band. The reader must be told
    the interval is too narrow, in those words -- not shown a bare band.
    """
    from app.lib.artifacts import coverage_note

    failures = coverage[~coverage["covers_nominal"]]
    assert len(failures) > 0, "fixture no longer exercises a calibration failure"
    for _, row in failures.iterrows():
        calibrated, message = coverage_note(
            coverage, row["target"], row["champion"], row["horizon"]
        )
        assert calibrated is False
        assert "too narrow" in message
        assert "optimistic" in message


def test_a_non_champion_model_is_marked_as_lacking_calibration_evidence(forward, coverage):
    from app.lib.artifacts import champion_for, coverage_note

    champion = champion_for(forward, TARGET_1, 1)
    other = next(m for m in forward[forward["target"] == TARGET_1]["model"].unique()
                 if m != champion)
    calibrated, message = coverage_note(coverage, TARGET_1, other, 1)
    assert calibrated is None
    assert "without a calibration check" in message


def test_provenance_caption_states_data_date_and_manual_refresh():
    """The addendum requires this readout wherever forecasts appear."""
    from app.lib.artifacts import provenance_caption

    caption = provenance_caption({
        "data_as_of": "2025-12-21", "generated_at_utc": "2026-01-01T00:00:00+00:00",
    })
    assert "2025-12-21" in caption
    assert "manual refresh" in caption.lower()


def test_missing_artifacts_raise_an_actionable_error(tmp_path, monkeypatch):
    """
    A bare FileNotFoundError from deep inside a page is unhelpful. The loader
    must name the command that produces the file.
    """
    import app.lib.artifacts as artifacts

    with pytest.raises(FileNotFoundError) as exc:
        artifacts._require(tmp_path / "nope.csv", "python -m src.forecast.generate")
    assert "python -m src.forecast.generate" in str(exc.value)
    assert "never trains" in str(exc.value)


def test_forecast_dates_land_on_reporting_days_only():
    """
    Forward dates are projected on the Sun-Thu cadence, not by adding calendar
    days -- adding days would place a forecast on a Friday or Saturday, which
    the programme never reports on.
    """
    from app.lib.forecast_page import _forecast_date
    from src.config import MASTER_SERIES_PATH

    history = pd.read_parquet(MASTER_SERIES_PATH)
    history["parsed_date"] = pd.to_datetime(history["parsed_date"])
    last_pos = len(history) - 1
    for horizon in FORECAST_HORIZONS:
        date = _forecast_date(history, last_pos, horizon)
        assert date.dayofweek not in (4, 5), (
            "h=%d projected onto %s, a non-reporting day" % (horizon, date.date())
        )
        assert date > history["parsed_date"].iloc[last_pos]
