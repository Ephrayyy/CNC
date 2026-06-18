from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.api.imdb_client import IMDbClient
from src.api.tmdb_client import TMDbClient


RAW_DIR = ROOT_DIR / "data" / "raw"


def enrich_with_omdb(records: List[Dict[str, object]], imdb_client: IMDbClient) -> List[Dict[str, object]]:
    enriched: List[Dict[str, object]] = []
    for record in records:
        imdb_id = record.get("imdb_id")
        merged = dict(record)
        if imdb_id and imdb_client.configured:
            try:
                omdb_payload = imdb_client.get_by_imdb_id(str(imdb_id))
                merged.update(imdb_client.normalize_record(omdb_payload))
            except Exception as exc:
                merged["omdb_error"] = str(exc)
        enriched.append(merged)
    return enriched


def write_jsonl(records: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(records: List[Dict[str, object]], path: Path) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame(records)
    dataframe.to_csv(path, index=False)
    return dataframe


def extract_movies(
    start_year: int,
    end_year: int,
    max_pages_per_year: int,
    original_language: Optional[str],
    include_omdb: bool,
) -> Dict[str, object]:
    tmdb_client = TMDbClient()
    imdb_client = IMDbClient()

    all_payloads: List[Dict[str, object]] = []
    for year in range(start_year, end_year + 1):
        payloads = tmdb_client.fetch_movies_by_year(
            year=year,
            max_pages=max_pages_per_year,
            original_language=original_language,
        )
        all_payloads.extend(payloads)

    normalized = tmdb_client.normalize_records(all_payloads)
    if include_omdb and imdb_client.configured:
        normalized = enrich_with_omdb(normalized, imdb_client)

    extracted_at = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    jsonl_path = RAW_DIR / f"tmdb_animation_movies_{extracted_at}.jsonl"
    csv_path = RAW_DIR / f"tmdb_animation_movies_{extracted_at}.csv"
    latest_jsonl_path = RAW_DIR / "tmdb_animation_movies_latest.jsonl"
    latest_csv_path = RAW_DIR / "tmdb_animation_movies_latest.csv"

    write_jsonl(all_payloads, jsonl_path)
    write_jsonl(all_payloads, latest_jsonl_path)
    dataframe = write_csv(normalized, csv_path)
    dataframe.to_csv(latest_csv_path, index=False)

    return {
        "rows": len(dataframe),
        "years": [start_year, end_year],
        "jsonl_path": str(jsonl_path),
        "csv_path": str(csv_path),
        "latest_csv_path": str(latest_csv_path),
        "include_omdb": include_omdb and imdb_client.configured,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract real animation movie data from TMDb and optionally OMDb.")
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=datetime.now(UTC).year)
    parser.add_argument("--max-pages-per-year", type=int, default=3)
    parser.add_argument("--original-language", type=str, default=None)
    parser.add_argument("--skip-omdb", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = extract_movies(
        start_year=args.start_year,
        end_year=args.end_year,
        max_pages_per_year=args.max_pages_per_year,
        original_language=args.original_language,
        include_omdb=not args.skip_omdb,
    )
    print(json.dumps(result, indent=2))
