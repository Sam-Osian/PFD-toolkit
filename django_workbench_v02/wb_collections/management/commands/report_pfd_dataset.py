from __future__ import annotations

import hashlib
import json
from collections import Counter
from importlib import metadata

import pandas as pd
from django.core.management.base import BaseCommand

from wb_collections.services import load_collections_dataset
from wb_workspaces.report_identity import REPORT_IDENTITY_COLUMN


def _package_version(package_name: str) -> str | None:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def _dataset_fingerprint(report_identities: list[str]) -> str:
    digest = hashlib.sha256()
    for report_identity in sorted(report_identities):
        digest.update(report_identity.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()[:20]


def _receiver_count(series: pd.Series) -> int:
    counter: Counter[str] = Counter()
    for value in series.fillna("").astype(str).tolist():
        for chunk in value.split(";"):
            receiver = chunk.strip()
            if not receiver:
                continue
            counter[receiver] += 1
    return len(counter)


class Command(BaseCommand):
    help = "Report live PFD dataset metadata (row count, date range, fingerprint, package version)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--refresh",
            action="store_true",
            help="Force upstream dataset refresh before reporting.",
        )
        parser.add_argument(
            "--json",
            action="store_true",
            help="Emit JSON instead of text lines.",
        )

    def handle(self, *args, **options):
        force_refresh = bool(options["refresh"])
        as_json = bool(options["json"])

        reports_df = load_collections_dataset(force_refresh=force_refresh)
        row_count = int(len(reports_df))
        column_count = int(len(reports_df.columns))

        date_series = pd.to_datetime(
            reports_df.get("date", pd.Series(dtype="object")),
            errors="coerce",
            dayfirst=True,
        ).dropna()
        min_date = date_series.min().date().isoformat() if not date_series.empty else None
        max_date = date_series.max().date().isoformat() if not date_series.empty else None

        report_identities = [
            str(value).strip()
            for value in reports_df.get(REPORT_IDENTITY_COLUMN, pd.Series(dtype="object")).tolist()
            if str(value).strip()
        ]
        fingerprint = _dataset_fingerprint(report_identities)

        area_unique = int(
            reports_df.get("area", pd.Series(dtype="object")).fillna("").astype(str).str.strip().replace("", pd.NA).dropna().nunique()
        )
        coroner_unique = int(
            reports_df.get("coroner", pd.Series(dtype="object")).fillna("").astype(str).str.strip().replace("", pd.NA).dropna().nunique()
        )
        receiver_unique = _receiver_count(reports_df.get("receiver", pd.Series(dtype="object")))

        theme_columns = sorted(
            [str(column) for column in reports_df.columns if str(column).startswith("theme_")]
        )

        payload = {
            "rows": row_count,
            "columns": column_count,
            "min_date": min_date,
            "max_date": max_date,
            "unique_areas": area_unique,
            "unique_coroners": coroner_unique,
            "unique_receivers": receiver_unique,
            "theme_column_count": len(theme_columns),
            "report_identity_fingerprint": fingerprint,
            "pfd_toolkit_version": _package_version("pfd_toolkit") or _package_version("pfd-toolkit"),
            "refreshed": force_refresh,
        }

        if as_json:
            self.stdout.write(json.dumps(payload, indent=2, sort_keys=True))
            return

        self.stdout.write(
            "\n".join(
                [
                    f"rows={payload['rows']}",
                    f"columns={payload['columns']}",
                    f"min_date={payload['min_date']}",
                    f"max_date={payload['max_date']}",
                    f"unique_areas={payload['unique_areas']}",
                    f"unique_coroners={payload['unique_coroners']}",
                    f"unique_receivers={payload['unique_receivers']}",
                    f"theme_column_count={payload['theme_column_count']}",
                    f"report_identity_fingerprint={payload['report_identity_fingerprint']}",
                    f"pfd_toolkit_version={payload['pfd_toolkit_version']}",
                    f"refreshed={payload['refreshed']}",
                ]
            )
        )
