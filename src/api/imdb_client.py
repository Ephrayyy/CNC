from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import requests


OMDB_BASE_URL = "https://www.omdbapi.com/"


@dataclass
class IMDbClient:
    api_key: Optional[str] = None
    timeout: int = 20
    max_retries: int = 3
    retry_sleep: float = 1.0
    session: requests.Session = field(default_factory=requests.Session)

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.getenv("OMDB_API_KEY")

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _get(self, **params: Any) -> Dict[str, Any]:
        if not self.api_key:
            raise ValueError("OMDB_API_KEY is missing. Add it to your environment to enable live enrichment.")

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(
                    OMDB_BASE_URL,
                    params={**params, "apikey": self.api_key},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                payload = response.json()
                if payload.get("Response") == "False":
                    raise ValueError(payload.get("Error", "Unknown OMDb error"))
                return payload
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                time.sleep(self.retry_sleep * attempt)

        raise RuntimeError(f"OMDb request failed: {last_error}")

    def get_title(self, title: str, year: Optional[int] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {"t": title, "plot": "short", "type": "movie"}
        if year is not None:
            params["y"] = year
        return self._get(**params)

    def get_by_imdb_id(self, imdb_id: str) -> Dict[str, Any]:
        return self._get(i=imdb_id, plot="short")

    @staticmethod
    def normalize_record(payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "imdb_id": payload.get("imdbID"),
            "omdb_title": payload.get("Title"),
            "omdb_year": payload.get("Year"),
            "omdb_rated": payload.get("Rated"),
            "omdb_released": payload.get("Released"),
            "omdb_runtime": payload.get("Runtime", "").replace(" min", "") or None,
            "omdb_genres": payload.get("Genre"),
            "omdb_director": payload.get("Director"),
            "omdb_writer": payload.get("Writer"),
            "omdb_actors": payload.get("Actors"),
            "omdb_plot": payload.get("Plot"),
            "omdb_language": payload.get("Language"),
            "omdb_country": payload.get("Country"),
            "omdb_awards": payload.get("Awards"),
            "omdb_metascore": payload.get("Metascore"),
            "omdb_imdb_rating": payload.get("imdbRating"),
            "omdb_imdb_votes": payload.get("imdbVotes", "").replace(",", "") or None,
            "omdb_box_office": payload.get("BoxOffice"),
            "omdb_production": payload.get("Production"),
        }
