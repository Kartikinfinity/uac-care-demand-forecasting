"""
smoke_test.py -- Deployment smoke test (Day 12).

Runs the roadmap's Part 11 Deployment Tests against a RUNNING app, local or
deployed, and the addendum's extension: "confirm the deployed app's provenance
readout matches the local run".

That last check is the important one. A dashboard that reads pre-generated
artifacts has a specific failure mode -- it keeps serving happily while showing
a data version that no longer matches the repository. Nothing crashes; the
numbers are simply from a different vintage than the code implies. So the app
serves its provenance sidecar as a static file, and this script fetches it from
the live URL and compares the hashes byte for byte against the local artifact.

Usage
    python scripts/smoke_test.py                          # against localhost:8501
    python scripts/smoke_test.py --url https://<app>.streamlit.app

Exit code is 0 only if every check passes, so it can gate a deploy.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import (  # noqa: E402
    FORECAST_PROVENANCE_PATH,
    STATIC_PROVENANCE_PATH,
)
from src.data.validate import read_provenance  # noqa: E402

DEFAULT_URL = "http://localhost:8501"
TIMEOUT = 20

# Every page Part 8 requires, by the route Streamlit derives from its filename.
PAGES = [
    ("/", "Executive Overview (Home)"),
    ("/Historical_Trends", "Historical Trends"),
    ("/Care_Load_Forecast", "Care Load Forecast"),
    ("/Discharge_Demand_Forecast", "Discharge Demand Forecast"),
    ("/Intake_vs_Exit_Pressure", "Intake vs. Exit Pressure"),
    ("/Model_Comparison", "Model Comparison & Accuracy"),
    ("/Scenario_Comparison", "Scenario Comparison"),
    ("/Methodology", "Methodology & Data"),
]


class Result:
    def __init__(self):
        self.checks = []

    def record(self, name: str, ok: bool, detail: str = "") -> bool:
        self.checks.append((name, ok, detail))
        print("  %s  %s%s" % ("PASS" if ok else "FAIL", name,
                              ("  -- " + detail) if detail else ""))
        return ok

    @property
    def failed(self):
        return [c for c in self.checks if not c[1]]


def _get(url: str, timeout: int = TIMEOUT):
    request = urllib.request.Request(url, headers={"User-Agent": "attest-smoke-test"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, response.read()


PLACEHOLDER_MARKERS = ("<", ">", "your-app", "your_app", "YOUR-APP", "example.com")


def looks_like_a_placeholder(url: str) -> bool:
    """
    The README shows `https://<your-app>.streamlit.app`. Pasted verbatim, that
    fails with a bare DNS error that reads like the deployment is broken. Catch
    it here and say what actually happened.
    """
    return any(marker in url for marker in PLACEHOLDER_MARKERS)


def check_reachable(result: Result, base: str) -> bool:
    if looks_like_a_placeholder(base):
        return result.record(
            "app is reachable and healthy", False,
            "the URL is still the documentation placeholder -- replace it with "
            "your real app address, e.g. https://uac-forecasting.streamlit.app",
        )
    try:
        status, _ = _get(base.rstrip("/") + "/_stcore/health")
        return result.record("app is reachable and healthy", status == 200,
                             "HTTP %d" % status)
    except urllib.error.HTTPError as exc:
        # streamlit.app resolves by wildcard, so a name with no app behind it
        # does NOT fail DNS -- it redirects, and urllib gives up on the loop.
        # Measured against a deliberately non-existent app name.
        hint = ""
        if "streamlit.app" in base and exc.code in (301, 302, 303, 307, 308):
            hint = (" -- resolves, but redirects to sign-in. Either no app "
                    "exists at this address, or it is deployed PRIVATE. A "
                    "private app cannot be smoke-tested: set Sharing to "
                    "'anyone with the link' in the app settings.")
        return result.record("app is reachable and healthy", False,
                             "HTTP %d%s" % (exc.code, hint))
    except urllib.error.URLError as exc:
        reason = str(getattr(exc, "reason", exc))
        hint = ""
        if "getaddrinfo" in reason or "Name or service not known" in reason:
            hint = (" -- the hostname does not resolve. Is the app deployed yet, "
                    "and is the address spelled correctly?")
        elif "infinite loop" in reason or "redirect" in reason.lower():
            hint = (" -- resolves, but redirects to sign-in. Either no app "
                    "exists at this address, or it is deployed PRIVATE. A "
                    "private app cannot be smoke-tested: set Sharing to "
                    "'anyone with the link' in the app settings.")
        return result.record("app is reachable and healthy", False,
                             "%s%s" % (reason, hint))
    except Exception as exc:  # noqa: BLE001
        return result.record("app is reachable and healthy", False,
                             "%s: %s" % (type(exc).__name__, exc))


def check_pages(result: Result, base: str) -> None:
    """
    Every route responds. Streamlit renders client-side, so a 200 here proves
    the route exists and the server is serving it -- not that the page rendered
    without a Python exception. Rendering is covered by the browser QA pass and
    by the artifact tests; this catches missing or misnamed pages after a deploy.
    """
    for route, label in PAGES:
        try:
            status, body = _get(base.rstrip("/") + route)
            result.record("page responds: %s" % label,
                          status == 200 and len(body) > 0, "HTTP %d" % status)
        except Exception as exc:  # noqa: BLE001
            result.record("page responds: %s" % label, False,
                          "%s: %s" % (type(exc).__name__, exc))


def check_provenance_match(result: Result, base: str) -> None:
    """
    THE addendum check: the live app must be serving the same data version the
    repository holds.

    Compares the raw-CSV and master-series SHA-256 from the deployed static
    sidecar against the local one. A mismatch means the deployment is running a
    different vintage of the data -- silently, since nothing errors.
    """
    url = base.rstrip("/") + "/app/static/provenance.json"
    try:
        status, body = _get(url)
    except Exception as exc:  # noqa: BLE001
        result.record("deployed provenance is fetchable", False,
                      "%s: %s -- is enableStaticServing set?" % (type(exc).__name__, exc))
        return
    if status != 200:
        result.record("deployed provenance is fetchable", False, "HTTP %d" % status)
        return
    result.record("deployed provenance is fetchable", True, url)

    try:
        deployed = json.loads(body.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        result.record("deployed provenance parses", False, str(exc))
        return

    local = json.loads(FORECAST_PROVENANCE_PATH.read_text(encoding="utf-8"))
    for field in ("raw_csv_sha256", "master_series_sha256", "data_as_of"):
        result.record(
            "deployed %s matches local" % field,
            deployed.get(field) == local.get(field),
            "deployed=%s local=%s" % (str(deployed.get(field))[:16],
                                      str(local.get(field))[:16]),
        )

    result.record(
        "deployed app reports manual-refresh policy",
        str(deployed.get("refresh_policy", "")).startswith("manual only"),
        str(deployed.get("refresh_policy", ""))[:40],
    )
    result.record(
        "deployed app labels the capacity signal a proxy",
        bool(deployed.get("early_warning", {}).get("is_proxy")),
    )


def check_local_consistency(result: Result) -> None:
    """
    The artifacts committed to the repository must describe the data actually in
    the repository. This is what stops a stale committed artifact being deployed
    and served as current.
    """
    try:
        data = read_provenance()
    except Exception as exc:  # noqa: BLE001
        result.record("local data provenance readable", False, str(exc))
        return
    result.record("local data provenance readable", True, data["data_as_of"])

    forecast = json.loads(FORECAST_PROVENANCE_PATH.read_text(encoding="utf-8"))
    for field in ("raw_csv_sha256", "master_series_sha256"):
        result.record(
            "committed forecasts match the committed data (%s)" % field,
            forecast.get(field) == data.get(field),
        )

    if STATIC_PROVENANCE_PATH.exists():
        static = json.loads(STATIC_PROVENANCE_PATH.read_text(encoding="utf-8"))
        result.record("static sidecar matches the forecast sidecar",
                      static.get("raw_csv_sha256") == forecast.get("raw_csv_sha256"))
    else:
        result.record("static sidecar exists", False,
                      "run: python -m src.forecast.generate")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL,
                        help="Base URL of the running app (default: %s)" % DEFAULT_URL)
    parser.add_argument("--skip-live", action="store_true",
                        help="Run only the local consistency checks.")
    args = parser.parse_args()

    result = Result()
    print("Deployment smoke test")
    print("  target: %s\n" % args.url)

    print("Local artifact consistency")
    check_local_consistency(result)

    if not args.skip_live:
        print("\nLive application")
        if check_reachable(result, args.url):
            check_pages(result, args.url)
            check_provenance_match(result, args.url)
        else:
            print("  (skipping page and provenance checks -- app unreachable)")

    print("\n%d checks, %d failed" % (len(result.checks), len(result.failed)))
    if result.failed:
        print("\nFAILED:")
        for name, _, detail in result.failed:
            print("  - %s%s" % (name, ("  -- " + detail) if detail else ""))
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
