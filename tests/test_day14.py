"""
test_day14.py -- Executive summary and documentation finalisation (Day 14).

Part 13 specifies the executive summary tightly enough to test: one page, eight
questions in a fixed order, non-technical, "no equations, no library names, no
code", and figures "filled in only from actual results". Those are asserted here
rather than eyeballed.

The addendum's Day 14 requirement -- "README/executive summary finalize seed,
versioning, refresh-policy, and hosting cold-start disclosures" -- gets one test
per disclosure, because a requirement of the form "make sure X is documented" is
otherwise the easiest kind to believe you have met.
"""
from __future__ import annotations

import json
import re

import pytest

from src.config import (
    EXEC_SUMMARY_EVIDENCE_PATH, EXECUTIVE_SUMMARY_PATH, PROJECT_ROOT,
    RANDOM_SEED, TARGET_1, TARGET_2,
)
from src.reporting.build_executive_summary import BANNED_TECHNICAL_TERMS, build

QUESTIONS = [
    (1, "problem"), (2, "built"), (3, "data"), (4, "achieved"),
    (5, "early"), (6, "value"), (7, "limitations"), (8, "used"),
]


@pytest.fixture(scope="module")
def summary():
    build()
    return EXECUTIVE_SUMMARY_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def ledger():
    return json.loads(EXEC_SUMMARY_EVIDENCE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def readme():
    return (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")


def flat(text: str) -> str:
    return re.sub(r"\s+", " ", text)


# ----------------------------------------------------------------------
# PART 13 SPEC
# ----------------------------------------------------------------------

def test_all_eight_questions_answered_in_the_specified_order(summary):
    headings = re.findall(r"^## (\d+)\. (.+)$", summary, re.MULTILINE)
    assert [int(n) for n, _ in headings] == [1, 2, 3, 4, 5, 6, 7, 8], (
        "Part 13 fixes both the set and the order of the eight questions")
    for (number, keyword), (_, title) in zip(QUESTIONS, headings):
        assert keyword in title.lower(), (
            "question %d should be about '%s', got '%s'" % (number, keyword, title))


def test_every_question_has_substantive_content(summary):
    """A heading with nothing under it would pass the order test."""
    sections = re.split(r"^## \d+\. .*$", summary, flags=re.MULTILINE)[1:]
    for number, body in enumerate(sections, 1):
        assert len(body.split()) >= 40, (
            "question %d has only %d words" % (number, len(body.split())))


def test_no_technical_jargon_library_names_or_code(summary):
    """
    Part 13: "No equations, no library names, no code."

    The audience is an operational stakeholder. A model name or a metric
    abbreviation makes the document unreadable for them, and the temptation to
    reach for one is exactly what this test exists to catch.
    """
    lowered = summary.lower()
    found = [term for term in BANNED_TECHNICAL_TERMS
             if re.search(r"\b" + re.escape(term) + r"\b", lowered)]
    assert not found, "technical terms in a non-technical summary: %s" % found

    assert "```" not in summary, "no code blocks"
    assert "=" not in summary.replace("--", ""), "no equations"


def test_raw_column_identifiers_never_appear(summary):
    """
    A stakeholder should not have to read a schema. The plain-language names are
    used instead of the dataset's column headers.
    """
    for column in (TARGET_1, TARGET_2):
        assert column not in summary, "raw column name leaked: " + column
    assert "children in HHS care" in summary


def test_it_is_about_one_page(summary):
    """
    Part 13 says one page. Eight mandated questions -- including a six-item
    limitations list and three decision questions -- do not compress below
    roughly a thousand words without dropping required substance, so the ceiling
    here is set at what the spec's own content demands and no higher.
    """
    words = len(summary.split())
    assert 700 <= words <= 1200, "%d words is not one page" % words


# ----------------------------------------------------------------------
# REAL NUMBERS ONLY
# ----------------------------------------------------------------------

def test_every_figure_is_ledgered_to_an_artifact(ledger):
    assert ledger["n_citations"] >= 15
    for source in ("forecasts/champion_selection.csv", "forecasts/kpi_summary.csv",
                   "data/interim/provenance.json"):
        assert source in ledger["sources"], "never cited " + source


def test_accuracy_figures_match_the_champion_artifact(ledger):
    """Re-derive the headline accuracy numbers straight from the CSV."""
    import pandas as pd

    champions = pd.read_csv(PROJECT_ROOT / "forecasts/champion_selection.csv")
    checked = 0
    for citation in ledger["citations"]:
        match = re.match(r"target=(.+) horizon=(\d+) -> champion_mae$",
                         citation["locator"])
        if not match:
            continue
        row = champions[(champions["target"] == match.group(1))
                        & (champions["horizon"] == int(match.group(2)))].iloc[0]
        assert float(citation["raw_value"]) == pytest.approx(row["champion_mae"])
        checked += 1
    assert checked >= 5


def test_lead_time_matches_the_kpi_artifact(ledger):
    import pandas as pd

    kpi = pd.read_csv(PROJECT_ROOT / "forecasts/kpi_summary.csv")
    found = False
    for citation in ledger["citations"]:
        if "median_surge_lead_time_periods" not in citation["locator"]:
            continue
        target = citation["locator"].split("target=")[1].split(" ->")[0]
        row = kpi[kpi["target"] == target].iloc[0]
        assert float(citation["raw_value"]) == pytest.approx(
            row["median_surge_lead_time_periods"])
        found = True
    assert found, "the surge lead time must be cited, not described"


# ----------------------------------------------------------------------
# THE HONEST ANSWERS
# ----------------------------------------------------------------------

def test_capacity_threshold_limitation_stated_explicitly(summary):
    """Part 16 checklist: 'States the capacity-threshold limitation explicitly'."""
    body = flat(summary)
    assert "not an official capacity figure" in body
    assert "no such figure exists" in body.lower()
    assert "high by recent standards" in body
    assert "There is no official capacity number" in body


def test_the_third_decision_question_is_answered_no(summary):
    """
    Part 1.2's third question is when to scale up ahead of a surge. The measured
    lead time does not support it, and the summary has to say so rather than
    letting the reader assume it is covered by the other two.
    """
    section = flat(summary.split("## 6.")[1].split("## 7.")[0])
    assert "scaled up ahead of a surge" in section
    assert "this system cannot answer it" in section
    assert "not capacity planning" in section


def test_the_baseline_result_is_not_dressed_up(summary):
    section = flat(summary.split("## 4.")[1].split("## 5.")[0])
    assert "should be stated rather than buried" in section
    assert "a finding about the data, not a shortcut" in section


def test_the_reconciliation_gap_reaches_the_stakeholder(summary):
    section = flat(summary.split("## 7.")[1].split("## 8.")[0])
    assert "at least one route into care is missing" in section.lower()


def test_used_as_input_not_as_an_automatic_trigger(summary):
    section = flat(summary.split("## 8.")[1])
    assert "not as an automatic trigger" in section
    assert "alongside existing judgement" in section


# ----------------------------------------------------------------------
# ADDENDUM DAY 14 DISCLOSURES -- one test each
# ----------------------------------------------------------------------

def test_disclosure_refresh_policy(summary, readme):
    assert "not live" in flat(summary).lower()
    assert "nothing updates or retrains on a schedule" in flat(summary)
    assert "Refresh Policy" in readme
    assert "not** a continuously live system" in readme or \
           "not a continuously live system" in readme


def test_disclosure_hosting_cold_start(summary, readme):
    """The free tier sleeps. A stakeholder meeting a 30-second blank page needs
    to know that is expected, not a failure."""
    assert "sleeps after disuse" in flat(summary)
    assert "half a minute" in flat(summary)
    body = flat(readme).lower()
    assert "sleep" in body and ("30 second" in body or "~30 second" in body)


def test_disclosure_seed(readme):
    assert str(RANDOM_SEED) in readme, "the random seed must be disclosed"
    assert "seed" in readme.lower()
    assert "reproduc" in readme.lower()


def test_disclosure_versioning(readme):
    """Which data vintage produced these artifacts, and how a reader checks."""
    body = readme.lower()
    assert "sha-256" in body or "sha256" in body
    assert "provenance" in body
    assert "data_as_of" in body or "data as of" in body


def test_readme_covers_setup_run_steps_architecture_and_limitations(readme):
    """Part 16: 'README covers setup, run steps, architecture, and every known
    limitation'."""
    for heading in ("## Setup", "## Project Structure",
                    "## Known Limitations", "## Deployment", "## Refresh Policy"):
        assert heading in readme, "README missing section: " + heading
    assert "pip install -r requirements.txt" in readme
    assert "streamlit run app/Home.py" in readme


def test_readme_links_resolve_to_files_that_exist(readme):
    """Every repository-relative path the README names must actually be there."""
    referenced = set(re.findall(r"`((?:src|app|tests|docs|reports|forecasts|data|scripts)/[\w./-]+)`",
                                readme))
    missing = [p for p in referenced
               if not (PROJECT_ROOT / p).exists() and "*" not in p]
    assert not missing, "README references paths that do not exist: %s" % sorted(missing)


def test_readme_names_the_live_deployment(readme):
    assert "streamlit.app" in readme
    assert "https://" in readme


# ----------------------------------------------------------------------
# REGENERATION IS DETERMINISTIC
# ----------------------------------------------------------------------

def test_regenerating_the_summary_is_byte_identical(summary):
    """
    The claim in section 8 -- "repeated runs on the same data give identical
    results by design" -- is itself testable.
    """
    build()
    assert EXECUTIVE_SUMMARY_PATH.read_text(encoding="utf-8") == summary


def test_readme_hashes_match_the_actual_provenance(readme):
    """
    The README quotes the current data digests. Quoting a value is the one thing
    this project does not otherwise allow, so it needs the same guard everything
    else gets: if the data is replaced and the README is not updated, the
    documented vintage silently stops describing the repository.
    """
    provenance = json.loads(
        (PROJECT_ROOT / "data/interim/provenance.json").read_text(encoding="utf-8"))
    for field in ("raw_csv_sha256", "master_series_sha256", "data_as_of"):
        assert provenance[field] in readme, (
            "README %s is stale: provenance says %s" % (field, provenance[field]))


def test_readme_links_the_two_deliverable_documents(readme):
    assert "reports/research_paper.md" in readme
    assert "reports/executive_summary.md" in readme


def test_the_summary_and_the_paper_agree_on_the_headline_numbers(summary):
    """
    Two generated documents, one pipeline. If they disagree, one of them was
    built against different artifacts.
    """
    paper = (PROJECT_ROOT / "reports/research_paper.md").read_text(encoding="utf-8")
    for shared in ("1,972", "11,516", "2,484", "720"):
        assert shared in summary, "summary missing " + shared
        assert shared in paper, "paper missing " + shared
