#!/usr/bin/env python3
"""Record the public HERMES multiplicity-database metadata and links.

The database page is reachable independently of the DESY archive host.  This
driver stores a local HTML provenance copy and a compact link/interpretation
manifest; it does not claim that the linked covariance archive was downloaded.
"""

from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
from html.parser import HTMLParser
from html import unescape
from pathlib import Path
import re

import requests


CAMPAIGN = Path(__file__).resolve().parents[1]
PAGE_URL = "https://hermesmults.appspot.com/"
DEFAULT_RAW = CAMPAIGN / "data/raw/hermes/hermesmults_index.html"
DEFAULT_OUTPUT = CAMPAIGN / "data/hermes_database_manifest.json"


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[dict[str, str]] = []
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        values = dict(attrs)
        self._href = values.get("href")
        self._text = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._href is not None:
            self.links.append({"text": " ".join("".join(self._text).split()), "href": self._href})
            self._href = None
            self._text = []


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page-url", default=PAGE_URL)
    parser.add_argument("--raw-page", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    response = requests.get(args.page_url, timeout=(20, 90), headers={"User-Agent": "bT-TMD-SIDIS-provenance/1.0"})
    response.raise_for_status()
    args.raw_page.parent.mkdir(parents=True, exist_ok=True)
    args.raw_page.write_bytes(response.content)
    text = response.content.decode("utf-8", errors="replace")
    parser_html = LinkParser()
    parser_html.feed(text)
    links = [item for item in parser_html.links if item.get("href")]
    relevant = [item for item in links if any(token in item["href"].lower() for token in ("multiplicities", "covmat", "mults"))]
    query_reports = []
    for target in ("proton", "deuteron"):
        query = {"what": "mults", "targ": target, "tag": "vmsub", "conf": "zpt-3D", "proj": "pt"}
        try:
            query_response = requests.post(args.page_url.rstrip("/") + "/results", data=query, timeout=(20, 90), headers={"User-Agent": "bT-TMD-SIDIS-provenance/1.0"})
            query_response.raise_for_status()
            query_parser = LinkParser()
            query_parser.feed(query_response.text)
            query_links = [item for item in query_parser.links if item.get("href") and item["href"].lower().endswith((".tar.gz", ".list.gz"))]
            query_reports.append({"query": query, "status": "links_recorded_no_archive_download", "links": sorted({item["href"] for item in query_links})})
        except requests.RequestException as exc:
            query_reports.append({"query": query, "status": "query_failed", "error": str(exc), "links": []})
    archive_links = sorted({item["href"] for item in relevant if item["href"].lower().endswith((".tar.gz", ".list.gz"))})
    archive_links.extend(link for item in query_reports for link in item["links"] if link not in archive_links)
    lower = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", unescape(text)).lower())
    observations = {
        "recommended_z_lower_cut": "z > 0.2 is recommended for general use",
        "high_z_warning": "z > 0.8 requires caution because exclusive contributions grow",
        "statistical_correlation": "unfolding induces statistical correlations; covariance matrices are provided",
        "systematic_treatment": "systematic uncertainties are described as point-to-point on the database page",
        "bin_integration": "numerator and denominator are integrated separately over each kinematic bin",
        "observations_found": {
            "z_cut": "z > 0.2" in lower,
            "high_z": "z > 0.8" in lower,
            "covariance": "covariance matrix" in lower,
            "bin_integrated": "integrated quantities" in lower,
        },
    }
    report = {
        "campaign": "sidis_global_analysis_2026",
        "status": "hermes_database_metadata_recorded_archive_pending",
        "source_url": args.page_url,
        "retrieved_date": date.today().isoformat(),
        "page_sha256": digest(args.raw_page),
        "download_all_url": next((item["href"] for item in links if "HERMES-multiplicities.tar.gz" in item.get("href", "")), None),
        "relevant_link_count": len(relevant),
        "selection_links": archive_links,
        "selection_queries": query_reports,
        "observations": observations,
        "archive_downloaded": False,
        "covariance_downloaded": False,
        "row_selection_authorized": False,
        "production_authorized": False,
        "note": "The DESY archive host was unreachable in this run; links are retained for a later retry.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "relevant_link_count": len(relevant), "archive_downloaded": False}, indent=2))


if __name__ == "__main__":
    main()
