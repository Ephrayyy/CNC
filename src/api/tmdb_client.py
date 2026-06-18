from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import requests


TMDB_BASE_URL = "https://api.themoviedb.org/3"


@dataclass
class TMDbClient:
    api_key: Optional[str] = None
    bearer_token: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    retry_sleep: float = 1.5
    session: requests.Session = field(default_factory=requests.Session)

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.getenv("TMDB_API_KEY")
        self.bearer_token = self.bearer_token or os.getenv("TMDB_BEARER_TOKEN")

    @property
    def configured(self) -> bool:
        return bool(self.api_key or self.bearer_token)

    def _headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        return headers

    def _auth_params(self) -> Dict[str, Any]:
        return {} if self.bearer_token else {"api_key": self.api_key}

    def _get(self, endpoint: str, **params: Any) -> Dict[str, Any]:
        if not self.configured:
            raise ValueError(
                "TMDb credentials missing. Set TMDB_BEARER_TOKEN or TMDB_API_KEY before running live extraction."
            )

        url = f"{TMDB_BASE_URL}/{endpoint.lstrip('/')}"
        merged_params = {**params, **self._auth_params()}

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(
                    url,
                    params=merged_params,
                    headers=self._headers(),
                    timeout=self.timeout,
                )
                if response.status_code == 429 and attempt < self.max_retries:
                    time.sleep(self.retry_sleep * attempt)
                    continue
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                time.sleep(self.retry_sleep * attempt)

        raise RuntimeError(f"TMDb request failed for endpoint '{endpoint}': {last_error}")

    def search_movie(self, query: str, page: int = 1, include_adult: bool = False) -> Dict[str, Any]:
        return self._get("search/movie", query=query, page=page, include_adult=str(include_adult).lower())

    def discover_movies(
        self,
        page: int = 1,
        primary_release_year: Optional[int] = None,
        with_genres: Optional[str] = None,
        sort_by: str = "popularity.desc",
        vote_count_gte: int = 50,
        original_language: Optional[str] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "page": page,
            "sort_by": sort_by,
            "vote_count.gte": vote_count_gte,
            "include_adult": "false",
            "include_video": "false",
        }
        if primary_release_year is not None:
            params["primary_release_year"] = primary_release_year
        if with_genres:
            params["with_genres"] = with_genres
        if original_language:
            params["with_original_language"] = original_language
        return self._get("discover/movie", **params)

    def discover_animation_movies(
        self,
        page: int = 1,
        year: Optional[int] = None,
        original_language: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.discover_movies(
            page=page,
            primary_release_year=year,
            with_genres="16",
            original_language=original_language,
        )

    def get_movie_details(self, movie_id: int) -> Dict[str, Any]:
        return self._get(
            f"movie/{movie_id}",
            append_to_response="credits,keywords,release_dates,external_ids",
        )

    def fetch_movies_by_year(
        self,
        year: int,
        max_pages: int = 5,
        original_language: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        detailed_records: List[Dict[str, Any]] = []

        for page in range(1, max_pages + 1):
            discovered = self.discover_animation_movies(
                page=page,
                year=year,
                original_language=original_language,
            )
            results = discovered.get("results", [])
            if not results:
                break

            for movie in results:
                movie_id = movie.get("id")
                if movie_id is None:
                    continue
                detailed_records.append(self.get_movie_details(int(movie_id)))

        return detailed_records

    @staticmethod
    def _extract_genres(payload: Dict[str, Any]) -> str:
        genres = payload.get("genres", [])
        return ", ".join(genre["name"] for genre in genres if genre.get("name"))

    @staticmethod
    def _extract_country(payload: Dict[str, Any]) -> Optional[str]:
        countries = payload.get("production_countries", [])
        if countries:
            return countries[0].get("iso_3166_1")
        return None

    @staticmethod
    def _extract_director(payload: Dict[str, Any]) -> Optional[str]:
        crew = payload.get("credits", {}).get("crew", [])
        directors = [person.get("name") for person in crew if person.get("job") == "Director" and person.get("name")]
        return ", ".join(directors) if directors else None

    @staticmethod
    def _extract_top_cast(payload: Dict[str, Any], limit: int = 5) -> str:
        cast = payload.get("credits", {}).get("cast", [])
        top_cast = [person.get("name") for person in cast[:limit] if person.get("name")]
        return ", ".join(top_cast)

    @staticmethod
    def _extract_keywords(payload: Dict[str, Any], limit: int = 10) -> str:
        keywords = payload.get("keywords", {}).get("keywords", [])
        return ", ".join(keyword.get("name") for keyword in keywords[:limit] if keyword.get("name"))

    @classmethod
    def normalize_record(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        external_ids = payload.get("external_ids", {})
        return {
            "tmdb_id": payload.get("id"),
            "imdb_id": payload.get("imdb_id") or external_ids.get("imdb_id"),
            "title": payload.get("title"),
            "original_title": payload.get("original_title"),
            "overview": payload.get("overview"),
            "release_date": payload.get("release_date"),
            "status": payload.get("status"),
            "genres": cls._extract_genres(payload),
            "keywords": cls._extract_keywords(payload),
            "director": cls._extract_director(payload),
            "top_cast": cls._extract_top_cast(payload),
            "budget": payload.get("budget"),
            "revenue": payload.get("revenue"),
            "runtime": payload.get("runtime"),
            "vote_average": payload.get("vote_average"),
            "vote_count": payload.get("vote_count"),
            "popularity": payload.get("popularity"),
            "original_language": payload.get("original_language"),
            "production_country": cls._extract_country(payload),
            "adult": payload.get("adult"),
            "video": payload.get("video"),
        }

    @classmethod
    def normalize_records(cls, payloads: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [cls.normalize_record(payload) for payload in payloads]
