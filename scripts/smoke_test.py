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


# Streamlit Community Cloud sends the FIRST request from any client through
# share.streamlit.io/-/auth/app to plant a session cookie -- on public apps too.
# A browser follows the hop, gets the cookie and comes back; a cookieless client
# sees 303 forever. So a 303 here says nothing about whether the app is private.
#
# `/~/+/<path>` is the cloud's internal route that skips the auth hop, and it is
# what this script uses once the plain path redirects. Measured against the live
# deployment: `/~/+/_stcore/health` returns "ok" (2 bytes) and
# `/~/+/app/static/provenance.json` returns the real JSON, where a cookie-jar
# client is handed the 9.8 KB SPA shell for every path -- including the static
# file -- which would make the provenance comparison parse HTML as JSON.
AUTH_HOP = "share.streamlit.io/-/auth"
INTERNAL_PREFIX = "/~/+"


def _resolve_prefix(base: str) -> str | None:
    """
    Work out how to reach this deployment: "" if plain paths work, INTERNAL_PREFIX
    if the app sits behind the auth hop, or None if neither responds.
    """
    for prefix in ("", INTERNAL_PREFIX):
        try:
            status, _ = _get(base.rstrip("/") + prefix + "/_stcore/health")
            if status == 200:
                return prefix
        except urllib.error.HTTPError as exc:
            if exc.code not in (301, 302, 303, 307, 308):
                return None
        except urllib.error.URLError:
            return None
        except Exception:  # noqa: BLE001
            return None
    return None


PLACEHOLDER_MARKERS = ("<", ">", "your-app", "your_app", "YOUR-APP", "example.com")


def looks_like_a_placeholder(url: str) -> bool:
    """
    The README shows `https://<your-app>.streamlit.app`. Pasted verbatim, that
    fails with a bare DNS error that reads like the deployment is broken. Catch
    it here and say what actually happened.
    """
    return any(marker in url for marker in PLACEHOLDER_MARKERS)


def check_reachable(result: Result, base: str) -> str | None:
    """
    Returns the path prefix to use for every later request, or None if the app
    could not be reached at all.
    """
    if looks_like_a_placeholder(base):
        result.record(
            "app is reachable and healthy", False,
            "the URL is still the documentation placeholder -- replace it with "
            "your real app address, e.g. https://uac-forecasting.streamlit.app",
        )
        return None

    try:
        prefix = _resolve_prefix(base)
    except Exception as exc:  # noqa: BLE001
        result.record("app is reachable and healthy", False,
                      "%s: %s" % (type(exc).__name__, exc))
        return None

    if prefix is None:
        # Reached only after BOTH the plain and internal routes failed. An app
        # that is merely asleep wakes on the first request and answers on a
        # retry, so persistent failure here is a real problem: wrong address,
        # never deployed, or genuinely private.
        result.record(
            "app is reachable and healthy", False,
            "no healthy response on either the public or the internal route. "
            "Check the address is right and the app is deployed; if it is "
            "sleeping, open it in a browser once to wake it, then re-run.",
        )
        return None

    result.record("app is reachable and healthy", True,
                  "HTTP 200" + (" via the internal route (the public route "
                                "redirects through Streamlit's auth hop, which "
                                "is normal for a public app)" if prefix else ""))
    return prefix


def check_pages(result: Result, base: str, prefix: str = "") -> None:
    """
    Every route responds. Streamlit renders client-side, so a 200 here proves
    the route exists and the server is serving it -- not that the page rendered
    without a Python exception. Rendering is covered by the browser QA pass and
    by the artifact tests; this catches missing or misnamed pages after a deploy.
    """
    for route, label in PAGES:
        try:
            status, body = _get(base.rstrip("/") + prefix + route)
            result.record("page responds: %s" % label,
                          status == 200 and len(body) > 0, "HTTP %d" % status)
        except Exception as exc:  # noqa: BLE001
            result.record("page responds: %s" % label, False,
                          "%s: %s" % (type(exc).__name__, exc))


def check_provenance_match(result: Result, base: str, prefix: str = "") -> None:
    """
    THE addendum check: the live app must be serving the same data version the
    repository holds.

    Compares the raw-CSV and master-series SHA-256 from the deployed static
    sidecar against the local one. A mismatch means the deployment is running a
    different vintage of the data -- silently, since nothing errors.
    """
    url = base.rstrip("/") + prefix + "/app/static/provenance.json"
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
        prefix = check_reachable(result, args.url)
        if prefix is not None:
            check_pages(result, args.url, prefix)
            check_provenance_match(result, args.url, prefix)
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
