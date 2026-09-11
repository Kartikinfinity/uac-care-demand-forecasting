"""
test_day13.py -- The research paper (Day 13).

The centrepiece is `test_every_numeral_in_the_paper_was_issued_by_the_ledger`.
The Day 13 validation checkpoint asks that "every number in the paper is
cross-checked against an actual output file -- zero figures written from memory
or assumption", and that is not something a human read-through can establish: a
figure typed from memory is indistinguishable, on the page, from one read out of
a CSV.

So the paper is generated, every numeral passes through `Evidence.cite`, and this
module scans the finished document and fails if it finds a number the ledger never
issued. A hand-typed figure does not survive that scan.

The other tests here check the addendum's four Day 13 content requirements and
verify that the ledger's claims still match the artifacts they name.
"""
from __future__ import annotations

import json
import re

import pandas as pd
import pytest

from src.config import (
    COL_DISCHARGED, COL_HHS_CARE, FEATURE_IMPORTANCE_PATH, PAPER_EVIDENCE_PATH,
    PROJECT_ROOT, RESEARCH_PAPER_PATH, SELECTION_SCOPE, SELECTION_WINDOW_RULE,
    TARGET_1, TARGET_2,
)
from src.reporting.build_paper import MODEL_LABELS, SECTIONS, build
from src.reporting.comparison_matrix import COMPLETE_COMPARISON_PATH


# ----------------------------------------------------------------------
# FIXTURES -- build once, inspect many times
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def built():
    build()
    return (RESEARCH_PAPER_PATH.read_text(encoding="utf-8"),
            json.loads(PAPER_EVIDENCE_PATH.read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def paper(built):
    return built[0]


@pytest.fixture(scope="module")
def ledger(built):
    return built[1]


def flat(text: str) -> str:
    """
    Collapse whitespace before matching prose.

    The paper is hard-wrapped markdown, so a sentence the generator wrote as one
    string arrives split across lines. Asserting on the raw text would make these
    checks fail on rewrapping rather than on meaning.

    Blockquote markers are dropped first: a wrapped line inside a `>` block would
    otherwise collapse to "were > scored", splitting a sentence that is continuous
    when read.
    """
    text = re.sub(r"^\s*>\s?", " ", text, flags=re.MULTILINE)
    return re.sub(r"\s+", " ", text)


# ----------------------------------------------------------------------
# THE CHECKPOINT
# ----------------------------------------------------------------------

# Structural, non-result numerals, removed before scanning. Each entry is a
# pattern whose numbers are navigational or bibliographic rather than findings,
# and each is listed individually so the exemption stays visible and small.
STRUCTURAL_PATTERNS = [
    (r"^#{1,6} .*$", "heading lines carry section numbers (## 14. Results)"),
    (r"```.*?```", "fenced code blocks are commands and file paths, not results"),
    (r"`[^`]*`", "inline code: identifiers, filenames, hashes, config names"),
    (r"§\s*\d+(?:\.\d+)?", "cross-references (§17.1)"),
    (r"Section\s+\d+(?:\.\d+)?", "cross-references (Section 17.1)"),
    (r"\b[LO]\d+\b", "limitation and objective labels (L2, O6)"),
    (r"\bh=\d+", "horizon labels -- the horizon values are ledgered separately"),
]


def _strip_structural(text: str) -> str:
    for pattern, _reason in STRUCTURAL_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.MULTILINE | re.DOTALL)
    return text


# Matched before plain numbers so an ISO date or timestamp stays one token
# instead of fragmenting into 2023 / 01 / 12.
COMPOUND = re.compile(r"\d{4}-\d{2}-\d{2}(?:T[\d:]+(?:[+-]\d{2}:\d{2})?)?")
NUMBER = re.compile(r"(?<![\w.\-])\d[\d,]*(?:\.\d+)?")


def _numerals(text: str) -> list[str]:
    found = COMPOUND.findall(text)
    found += NUMBER.findall(COMPOUND.sub(" ", text))
    return found


def _issued(ledger: dict) -> set[str]:
    """Every string the ledger rendered, plus the numerals inside each."""
    out: set[str] = set()
    for citation in ledger["citations"]:
        rendered = str(citation["rendered"])
        out.add(rendered)
        out.update(_numerals(rendered))
    return out


def test_every_numeral_in_the_paper_was_issued_by_the_ledger(paper, ledger):
    """
    THE Day 13 checkpoint, mechanised.

    Fails if the paper contains a number that did not come out of an artifact.
    """
    issued = _issued(ledger)
    body = _strip_structural(paper)

    unaccounted: dict[str, str] = {}
    for line in body.split("\n"):
        for token in _numerals(line):
            if token in issued:
                continue
            unaccounted.setdefault(token, line.strip()[:120])

    assert not unaccounted, (
        "These numerals appear in the paper but were never issued by "
        "Evidence.cite, so they cannot be traced to an artifact:\n"
        + "\n".join("  %-24s in: %s" % (tok, ctx)
                    for tok, ctx in sorted(unaccounted.items()))
    )


def test_the_references_section_is_the_only_bibliographic_exemption(paper):
    """
    Section 24 carries publication years and volume numbers, which are citations
    rather than results. The scan above covers them anyway -- this test exists to
    assert that fact, so nobody later "fixes" the scan by exempting the section.
    """
    assert "## 24. References" in paper
    assert "STRUCTURAL_PATTERNS" not in paper
    references = paper.split("## 24. References")[1].split("## 25.")[0]
    assert "1927" in references, "expected the Wilson citation year to survive"


def test_ledger_claims_still_match_their_artifacts(ledger):
    """
    Re-derive a sample of ledgered values straight from the CSVs, independently
    of the generator. Catches a paper built against artifacts that have since
    been regenerated with different numbers.
    """
    champions = pd.read_csv(PROJECT_ROOT / "forecasts/champion_selection.csv")
    checked = 0
    for citation in ledger["citations"]:
        if citation["source"] != "forecasts/champion_selection.csv":
            continue
        match = re.match(r"target=(.+) horizon=(\d+) -> champion_mae$",
                         citation["locator"])
        if not match:
            continue
        row = champions[(champions["target"] == match.group(1))
                        & (champions["horizon"] == int(match.group(2)))].iloc[0]
        # Compare the RAW value, not the rendering: the same figure is quoted at
        # two precisions in different parts of the paper, and what must match the
        # artifact is the number, not the number of decimal places.
        assert float(citation["raw_value"]) == pytest.approx(row["champion_mae"])
        checked += 1
    assert checked >= 6, "expected every champion MAE to be re-checkable"


def test_no_placeholder_text_survived(paper):
    for marker in ["TODO", "TBD", "FIXME", "XXX", "lorem ipsum", "<insert", "N/A%"]:
        assert marker.lower() not in paper.lower(), "placeholder left in paper: " + marker


# ----------------------------------------------------------------------
# ADDENDUM DAY 13 CONTENT REQUIREMENTS
# ----------------------------------------------------------------------

def test_horizon_versus_decision_timescale_gap_is_prominent(paper):
    """
    Addendum: "states the horizon-vs-decision-timescale gap plainly".

    Prominent means the executive summary and the head of the early-warning
    section, not a footnote -- so assert placement, not just presence.
    """
    summary = flat(paper.split("## 2. Executive Summary")[1].split("## 3.")[0])
    assert "lead time" in summary.lower()
    assert "cannot support the decisions its framing invites" in summary

    early_warning = flat(paper.split("## 17. Early-Warning System")[1].split("## 18.")[0])
    assert "horizon-versus-decision-timescale gap" in early_warning[:1200].lower(), \
        "the gap must open the section, not trail it"
    assert "does not provide enough notice to change a capacity decision" in early_warning
    assert "17.1" in flat(paper.split("## 20. Limitations")[1][:5000]), \
        "L1 should point back at the early-warning section"


def test_full_comparison_matrix_is_reported_not_just_the_winning_path(paper):
    """
    Addendum: "reports the FULL comparison matrix, not just the winning path".

    All eight models must appear in every one of the six cells, including the
    ones that beat the champion numerically.
    """
    section = flat(paper.split("## 15. Model Comparison")[1].split("## 16.")[0])
    for target in (TARGET_1, TARGET_2):
        for horizon in (1, 7, 14):
            heading = "#### %s — h=%d" % (target, horizon)
            assert heading in section, "missing comparison block: " + heading

    for model_label in MODEL_LABELS.values():
        assert model_label in section, "model absent from the matrix: " + model_label

    assert "including the runs that beat the selected champion numerically" in section


def test_feature_importances_carry_the_collinearity_caveat_above_the_table(paper):
    """
    Addendum: "caveats any RF/GBR feature importances for collinearity".

    The caveat must precede the numbers; a caveat after the table is read after
    the reader has already formed a conclusion.
    """
    section = flat(paper.split("## 10. Feature Engineering")[1].split("## 11.")[0])
    if "impurity importance" not in section.lower():
        pytest.skip("no importances extracted; nothing to caveat")

    caveat_at = section.find("Read the caveat before the table")
    table_at = section.find("| Target | Horizon |")
    assert caveat_at != -1, "collinearity caveat missing"
    assert table_at != -1, "importance table missing"
    assert caveat_at < table_at, "the caveat must appear above the table"
    assert "correlated predictors" in section
    assert "not evidence of a causal driver" in section


def test_omitted_variable_limitation_is_prominent_and_unhedged(paper):
    """
    Addendum: the omitted-variable / reconciliation limitation gets "prominent,
    unhedged placement".

    Prominent: executive summary, its own top-level limitation, and the
    conclusion. Unhedged: no softening qualifier around the core claim.
    """
    assert "### L2 — The measured flows do not explain the measured stock" in paper

    limitation = flat(paper.split("### L2 —")[1].split("### L3")[0])
    assert "at least one material intake channel" in limitation.lower()
    assert "untestable with this dataset" in limitation

    for hedge in ["may possibly", "might perhaps", "arguably", "it could be argued",
                  "some might say", "relatively minor"]:
        assert hedge not in limitation.lower(), "hedged language in L2: " + hedge

    summary = flat(paper.split("## 2. Executive Summary")[1].split("## 3.")[0])
    assert "do not explain the measured stock" in summary

    conclusion = flat(paper.split("## 23. Conclusion")[1].split("## 24.")[0])
    assert "open system" in conclusion.lower()


# ----------------------------------------------------------------------
# STRUCTURE AND CONSISTENCY
# ----------------------------------------------------------------------

def test_all_twentyfive_part12_sections_present(paper):
    expected = [
        (2, "Executive Summary"), (3, "Background"), (4, "Problem Statement"),
        (5, "Objectives"), (6, "Dataset"), (7, "Data Preparation"),
        (8, "Exploratory Analysis"), (9, "Methodology"),
        (10, "Feature Engineering"), (11, "Forecasting Models"),
        (12, "Experimental Design"), (13, "Validation Strategy"), (14, "Results"),
        (15, "Model Comparison"), (16, "Forecast Analysis"),
        (17, "Early-Warning System"), (18, "Dashboard"), (19, "Deployment"),
        (20, "Limitations"), (21, "Recommendations"), (22, "Future Work"),
        (23, "Conclusion"), (24, "References"), (25, "Appendix"),
    ]
    for number, title in expected:
        assert "## %d. %s" % (number, title) in paper, "missing section %d" % number
    assert paper.lstrip().startswith("# "), "section 1 is the title block"
    assert len(SECTIONS) == 25


def test_model_labels_match_the_dashboard(paper):
    """
    The reporting module duplicates MODEL_LABELS to avoid importing streamlit.
    Assert the copy has not drifted from the app's.
    """
    from app.lib.artifacts import MODEL_LABELS as APP_LABELS
    assert MODEL_LABELS == APP_LABELS


def test_capacity_proxy_is_never_presented_as_official(paper):
    early_warning = flat(paper.split("## 17. Early-Warning System")[1].split("## 18.")[0])
    assert "proxy" in early_warning.lower()
    flat_paper = flat(paper)
    assert "No official capacity threshold exists" in flat_paper
    assert "does not indicate that capacity has been or will be breached" in flat_paper


def test_paper_reports_the_baseline_result_rather_than_burying_it(paper):
    """
    The headline finding is that simple baselines win. A paper that led with the
    complex models would be misrepresenting its own evidence.
    """
    summary = flat(paper.split("## 2. Executive Summary")[1].split("## 3.")[0])
    assert "principal finding is negative" in summary
    comparison = flat(paper.split("### 15.1")[1].split("## 16.")[0])
    assert "real result, not a placeholder" in comparison


def test_champion_table_matches_the_registry(paper):
    """Every champion named in Section 14 is the one the frozen registry holds."""
    registry = json.loads((PROJECT_ROOT / "models/model_registry.json")
                          .read_text(encoding="utf-8"))
    results = flat(paper.split("### 14.1 Selected champions")[1].split("### 14.2")[0])
    for entry in registry["entries"]:
        assert MODEL_LABELS[entry["champion"]] in results, (
            "registry champion %s missing from the results table for %s h=%d"
            % (entry["champion"], entry["target"], entry["horizon"]))


def test_provenance_hashes_appear_in_the_paper(paper):
    data_provenance = json.loads(
        (PROJECT_ROOT / "data/interim/provenance.json").read_text(encoding="utf-8"))
    assert data_provenance["raw_csv_sha256"] in paper
    assert data_provenance["master_series_sha256"] in paper


def test_evidence_ledger_is_substantial_and_multi_sourced(ledger):
    assert ledger["n_citations"] > 300, "suspiciously few cited numbers"
    assert ledger["n_distinct_sources"] >= 8
    for required in ["forecasts/champion_selection.csv",
                     "forecasts/comparison_matrix.csv",
                     "forecasts/kpi_summary.csv",
                     "forecasts/interval_coverage.csv",
                     "data/interim/provenance.json",
                     "src/config.py"]:
        assert required in ledger["sources"], "ledger never cited " + required


def test_both_mandated_targets_are_covered(paper):
    for target in (COL_HHS_CARE, COL_DISCHARGED):
        assert target in paper


def test_comparison_uses_the_governing_scope(paper):
    section = flat(paper.split("## 15. Model Comparison")[1].split("## 16.")[0])
    assert SELECTION_SCOPE in section
    assert SELECTION_WINDOW_RULE in section


# ----------------------------------------------------------------------
# FEATURE IMPORTANCE EXTRACTION
# ----------------------------------------------------------------------

def test_feature_importance_artifact_is_well_formed():
    if not FEATURE_IMPORTANCE_PATH.exists():
        pytest.skip("importances not extracted")
    frame = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    assert set(frame.columns) >= {"target", "family", "horizon", "feature",
                                  "importance"}
    assert (frame["importance"] >= 0).all()
    assert set(frame["target"]) == {TARGET_1, TARGET_2}
    # Impurity importances are a distribution over features, per fitted model.
    for _, group in frame.groupby(["target", "family", "horizon"]):
        assert group["importance"].sum() == pytest.approx(1.0, abs=1e-6)


def test_gradient_boosting_is_omitted_rather_than_substituted():
    """
    HistGradientBoostingRegressor exposes no impurity importances. Reporting a
    different measure in the same column would produce numbers that are not
    comparable with the Random Forest ones.
    """
    if not FEATURE_IMPORTANCE_PATH.exists():
        pytest.skip("importances not extracted")
    frame = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    assert "gradient_boosting" not in set(frame["family"])


# ----------------------------------------------------------------------
# COMPLETED COMPARISON MATRIX
# ----------------------------------------------------------------------

def test_completed_matrix_contains_all_eight_models():
    matrix = pd.read_csv(COMPLETE_COMPARISON_PATH)
    assert set(matrix["model"]) == set(MODEL_LABELS), (
        "the matrix must carry every evaluated candidate")


def test_appending_the_ensemble_changed_no_existing_number():
    """
    The guard that makes the extension safe.

    Every row from the seven-model artifact must survive byte-identically. If a
    future change recomputes common support across eight models instead, every
    one of these numbers shifts and this test fails -- which is the intended
    outcome, because the paper would then disagree with the dashboard and the
    registry.
    """
    original = pd.read_csv(PROJECT_ROOT / "forecasts/full_model_comparison.csv")
    complete = pd.read_csv(COMPLETE_COMPARISON_PATH)
    carried = complete[complete["model"] != "ensemble"].reset_index(drop=True)
    pd.testing.assert_frame_equal(original, carried, check_like=False)


def test_ensemble_scored_on_the_same_points_as_the_others():
    """n_scored must match the incumbent models cell for cell."""
    matrix = pd.read_csv(COMPLETE_COMPARISON_PATH)
    governing = matrix[(matrix["fold_scope"] == SELECTION_SCOPE)
                       & (matrix["window_rule"] == SELECTION_WINDOW_RULE)]
    for (target, horizon), cell in governing.groupby(["target", "horizon"]):
        counts = set(cell["n_scored"])
        assert len(counts) == 1, (
            "models in %s h=%d were scored on different numbers of points: %s"
            % (target, horizon, sorted(counts)))


def test_ensemble_mae_reproduces_the_frozen_registry():
    """
    Independent confirmation: the registry recorded an ensemble MAE at Day 8 via
    a different code path. Scoring it here must land on the same number.
    """
    registry = json.loads((PROJECT_ROOT / "models/model_registry.json")
                          .read_text(encoding="utf-8"))
    matrix = pd.read_csv(COMPLETE_COMPARISON_PATH)
    governing = matrix[(matrix["fold_scope"] == SELECTION_SCOPE)
                       & (matrix["window_rule"] == SELECTION_WINDOW_RULE)
                       & (matrix["model"] == "ensemble")]
    checked = 0
    for entry in registry["entries"]:
        if entry.get("numerical_leader") != "ensemble":
            continue
        row = governing[(governing["target"] == entry["target"])
                        & (governing["horizon"] == entry["horizon"])].iloc[0]
        assert row["MAE"] == pytest.approx(entry["numerical_leader_mae"])
        checked += 1
    assert checked >= 1, "expected at least one cell where the ensemble led"


def test_paper_explains_the_matrix_assembly(paper):
    section = flat(paper.split("## 15. Model Comparison")[1].split("## 16.")[0])
    assert "A note on how this matrix was assembled" in section
    assert "the support set the other seven were scored on" in section.lower()
    assert "carried across unchanged" in section.lower()


def test_dashboard_and_paper_report_the_same_models():
    """
    The paper and the Model Comparison page must not disagree about how many
    candidates were evaluated. Both now read the completed matrix; this test is
    what stops them drifting apart again.
    """
    from app.lib.artifacts import load_comparison
    app_models = set(load_comparison()["model"])
    paper_models = set(pd.read_csv(COMPLETE_COMPARISON_PATH)["model"])
    assert app_models == paper_models == set(MODEL_LABELS)


def test_the_app_still_never_imports_the_evaluation_harness():
    """
    The app reads artifacts and never trains. Wiring it to the completed matrix
    must not have dragged the walk-forward harness in through an import.
    """
    import ast

    source = (PROJECT_ROOT / "app/lib/artifacts.py").read_text(encoding="utf-8")
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    # Checked against the import graph, not the raw text: `_require` builds
    # error messages that name `python -m src.evaluation.run_ml` as the command
    # to run, and that guidance is not an import.
    for forbidden in ("src.evaluation", "src.reporting", "sklearn", "statsmodels"):
        offenders = [m for m in imported if m == forbidden
                     or m.startswith(forbidden + ".")]
        assert not offenders, (
            "app/lib/artifacts.py must not import %s (found %s)"
            % (forbidden, offenders))
