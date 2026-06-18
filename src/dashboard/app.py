from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from src.models.train_model import METRICS_PATH, MODEL_PATH
from src.processing.cleaning import PROCESSED_DATA_PATH, build_clean_dataset
from src.processing.feature_engineering import GENRE_FLAGS, build_feature_table


st.set_page_config(page_title="CineScope AI", layout="wide")


PALETTE = ["#0f766e", "#f59e0b", "#dc2626", "#2563eb", "#7c3aed", "#111827"]


@st.cache_data
def load_dataset() -> pd.DataFrame:
    if PROCESSED_DATA_PATH.exists():
        df = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["release_date"])
    else:
        df = build_clean_dataset()

    required_feature_columns = {
        "release_year",
        "release_month",
        "release_quarter",
        "is_holiday_window",
        "budget_musd",
        "revenue_musd",
        "profit_musd",
        "vote_count_log",
        "popularity_log",
        "num_genres",
        "genre_animation",
        "genre_adventure",
        "genre_family",
        "genre_drama",
        "genre_fantasy",
        "genre_comedy",
        "genre_action",
        "success_score",
        "success_label",
    }

    if not required_feature_columns.issubset(df.columns):
        df = build_feature_table(df)
        df.to_csv(PROCESSED_DATA_PATH, index=False)

    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").fillna(0).astype(int)
    return df


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def format_money(value: float) -> str:
    return f"${value / 1_000_000_000:.2f}B" if value >= 1_000_000_000 else f"${value / 1_000_000:.0f}M"


def extract_genre_counts(df: pd.DataFrame) -> pd.DataFrame:
    counter = Counter()
    for value in df["genres"].dropna():
        for genre in [item.strip() for item in str(value).split(",") if item.strip()]:
            counter[genre] += 1
    genre_df = pd.DataFrame(counter.items(), columns=["genre", "count"]).sort_values("count", ascending=False)
    return genre_df


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filtres")

    year_min = int(df["release_year"].min())
    year_max = int(df["release_year"].max())
    selected_years = st.sidebar.slider("Période", min_value=year_min, max_value=year_max, value=(year_min, year_max))

    languages = sorted(df["original_language"].dropna().unique().tolist())
    selected_languages = st.sidebar.multiselect("Langues", languages, default=languages)

    countries = sorted(df["production_country"].dropna().unique().tolist())
    selected_countries = st.sidebar.multiselect("Pays", countries, default=countries)

    genre_options = sorted(extract_genre_counts(df)["genre"].tolist())
    selected_genres = st.sidebar.multiselect("Genres", genre_options, default=[])

    min_votes = st.sidebar.slider("Votes minimum", min_value=0, max_value=int(df["vote_count"].max()), value=50)
    only_released = st.sidebar.checkbox("Uniquement les titres sortis", value=True)

    filtered = df[df["release_year"].between(*selected_years)].copy()
    filtered = filtered[filtered["original_language"].isin(selected_languages)]
    filtered = filtered[filtered["production_country"].isin(selected_countries)]
    filtered = filtered[filtered["vote_count"] >= min_votes]

    if only_released and "status" in filtered.columns:
        filtered = filtered[filtered["status"].fillna("").eq("Released")]

    if selected_genres:
        pattern = "|".join(selected_genres)
        filtered = filtered[filtered["genres"].str.contains(pattern, case=False, na=False)]

    return filtered


def render_header(df: pd.DataFrame) -> None:
    st.title("CineScope AI")
    st.caption("Veille data sur l'animation mondiale : extraction API, analyse marché, modélisation et restitution")

    zero_budget_share = (df["budget"] == 0).mean() if len(df) else 0
    zero_revenue_share = (df["revenue"] == 0).mean() if len(df) else 0
    st.info(
        "Lecture recommandée : utiliser ce dashboard comme outil de veille marché. "
        f"La base filtrée contient {len(df)} titres, avec {zero_budget_share:.0%} de budgets nuls et "
        f"{zero_revenue_share:.0%} de revenus nuls."
    )


def render_overview(df: pd.DataFrame) -> None:
    total_titles = len(df)
    total_revenue = float(df["revenue"].sum())
    avg_rating = float(df["vote_average"].mean())
    avg_roi = float(df["roi"].mean())
    top_language = df["original_language"].mode().iloc[0] if not df["original_language"].mode().empty else "n/a"
    top_country = df["production_country"].mode().iloc[0] if not df["production_country"].mode().empty else "n/a"

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Titres", f"{total_titles}")
    col2.metric("Revenus cumulés", format_money(total_revenue))
    col3.metric("Note moyenne", f"{avg_rating:.2f}")
    col4.metric("ROI moyen", f"{avg_roi:.2f}")
    col5.metric("Langue dominante", top_language.upper())
    col6.metric("Pays dominant", top_country)

    yearly = df.groupby("release_year", as_index=False).agg(
        n_titles=("title", "count"),
        revenue=("revenue", "sum"),
        avg_rating=("vote_average", "mean"),
    )

    left, right = st.columns(2)
    with left:
        fig_titles = px.bar(
            yearly,
            x="release_year",
            y="n_titles",
            title="Volume de titres par année",
            color_discrete_sequence=[PALETTE[0]],
        )
        st.plotly_chart(fig_titles, use_container_width=True)

    with right:
        fig_revenue = px.line(
            yearly,
            x="release_year",
            y="revenue",
            markers=True,
            title="Revenus cumulés par année",
            color_discrete_sequence=[PALETTE[1]],
        )
        fig_revenue.update_yaxes(tickprefix="$")
        st.plotly_chart(fig_revenue, use_container_width=True)


def render_market_view(df: pd.DataFrame) -> None:
    st.subheader("Panorama Marché")

    language_df = df["original_language"].value_counts().head(10).reset_index()
    language_df.columns = ["original_language", "count"]

    country_df = df["production_country"].value_counts().head(10).reset_index()
    country_df.columns = ["production_country", "count"]

    genre_df = extract_genre_counts(df).head(12)

    left, right = st.columns(2)
    with left:
        fig_lang = px.bar(
            language_df,
            x="original_language",
            y="count",
            title="Top langues",
            color_discrete_sequence=[PALETTE[3]],
        )
        st.plotly_chart(fig_lang, use_container_width=True)

    with right:
        fig_country = px.bar(
            country_df,
            x="production_country",
            y="count",
            title="Top pays de production",
            color_discrete_sequence=[PALETTE[2]],
        )
        st.plotly_chart(fig_country, use_container_width=True)

    fig_genres = px.bar(
        genre_df,
        x="genre",
        y="count",
        title="Genres les plus fréquents",
        color_discrete_sequence=[PALETTE[4]],
    )
    st.plotly_chart(fig_genres, use_container_width=True)


def render_performance_view(df: pd.DataFrame) -> None:
    st.subheader("Performance")

    left, right = st.columns(2)
    with left:
        fig_budget_revenue = px.scatter(
            df,
            x="budget",
            y="revenue",
            color="original_language",
            hover_name="title",
            hover_data=["release_year", "production_country", "vote_average", "roi"],
            title="Budget vs revenu",
            color_discrete_sequence=PALETTE,
        )
        st.plotly_chart(fig_budget_revenue, use_container_width=True)

    with right:
        fig_popularity = px.scatter(
            df,
            x="popularity",
            y="vote_average",
            size="vote_count",
            color="production_country",
            hover_name="title",
            hover_data=["release_year", "revenue"],
            title="Popularité vs note",
            color_discrete_sequence=PALETTE,
        )
        st.plotly_chart(fig_popularity, use_container_width=True)

    ranking_metric = st.radio(
        "Classement principal",
        ["revenue", "roi", "vote_average", "popularity"],
        horizontal=True,
        format_func=lambda value: {
            "revenue": "Revenus",
            "roi": "ROI",
            "vote_average": "Note",
            "popularity": "Popularité",
        }[value],
    )

    ranking = (
        df[["title", "release_date", "genres", "production_country", "original_language", "budget", "revenue", "roi", "vote_average", "popularity"]]
        .sort_values(ranking_metric, ascending=False)
        .head(15)
        .copy()
    )
    ranking["release_date"] = pd.to_datetime(ranking["release_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    st.dataframe(ranking, use_container_width=True, hide_index=True)


def render_explorer(df: pd.DataFrame) -> None:
    st.subheader("Exploration Titre")

    options = df.sort_values(["release_year", "title"], ascending=[False, True])["title"].dropna().tolist()
    selected_title = st.selectbox("Choisir un titre", options)
    selected = df.loc[df["title"] == selected_title].iloc[0]

    left, right = st.columns([1, 1])
    with left:
        st.markdown(f"### {selected['title']}")
        st.write(f"**Date de sortie** : {pd.to_datetime(selected['release_date']).date() if pd.notna(selected['release_date']) else 'n/a'}")
        st.write(f"**Genres** : {selected.get('genres', 'n/a')}")
        st.write(f"**Réalisateur(s)** : {selected.get('director', 'n/a')}")
        st.write(f"**Casting principal** : {selected.get('top_cast', 'n/a')}")
        st.write(f"**Pays / Langue** : {selected.get('production_country', 'n/a')} / {str(selected.get('original_language', 'n/a')).upper()}")

    with right:
        st.metric("Budget", format_money(float(selected["budget"])))
        st.metric("Revenu", format_money(float(selected["revenue"])))
        st.metric("ROI", f"{float(selected['roi']):.2f}")
        st.metric("Note", f"{float(selected['vote_average']):.2f}")
        st.metric("Popularité", f"{float(selected['popularity']):.2f}")

    st.write("**Synopsis**")
    st.write(selected.get("overview") or "Aucun synopsis disponible.")

    st.write("**Mots-clés**")
    st.write(selected.get("keywords") or "Aucun mot-clé disponible.")


def render_model_view(df: pd.DataFrame, artifact) -> None:
    st.subheader("Modélisation")

    metrics = None
    if METRICS_PATH.exists():
        with METRICS_PATH.open("r", encoding="utf-8") as stream:
            metrics = json.load(stream)

    if metrics:
        top_left, top_right = st.columns([1, 1])
        with top_left:
            st.metric("Modèle retenu", metrics["selected_model"])
            st.metric("Accuracy", f"{metrics['metrics']['accuracy']:.3f}")
            st.metric("F1", f"{metrics['metrics']['f1']:.3f}")
        with top_right:
            st.write("**Lecture**")
            st.write(
                "La cible est un score de succès construit à partir du ROI, de la note, de la popularité "
                "et du volume de votes. Le modèle sert à illustrer une aide à la décision, pas une vérité métier absolue."
            )

    if artifact is None:
        st.warning("Artefact modèle absent. Lance `python3 src/models/train_model.py` avant d'utiliser le simulateur.")
        return

    st.markdown("#### Simulateur Projet")
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            title = st.text_input("Nom du projet", value="Projet animation CNC")
            budget_musd = st.slider("Budget (M$)", 1, 300, 60)
            runtime = st.slider("Durée", 60, 180, 95)
            release_month = st.selectbox("Mois de sortie", list(range(1, 13)), index=5)
        with col2:
            vote_average = st.slider("Note critique attendue", 1.0, 10.0, 7.0, 0.1)
            vote_count_log_seed = st.slider("Volume d'avis attendu (log)", 0.0, 10.0, 5.0, 0.1)
            popularity_log = st.slider("Popularité attendue (log)", 0.0, 6.0, 3.2, 0.1)
            num_genres = st.slider("Nombre de genres", 1, 5, 3)
        with col3:
            original_language = st.selectbox("Langue originale", sorted(df["original_language"].dropna().unique()))
            production_country = st.selectbox("Pays principal", sorted(df["production_country"].dropna().unique()))
            selected_genres = st.multiselect("Genres", GENRE_FLAGS, default=["Animation", "Adventure", "Family"])

        submitted = st.form_submit_button("Estimer le potentiel")

    if submitted:
        features = {
            "budget_musd": budget_musd,
            "runtime": runtime,
            "vote_average": vote_average,
            "vote_count_log": vote_count_log_seed,
            "popularity_log": popularity_log,
            "num_genres": num_genres,
            "release_year": 2026,
            "release_month": release_month,
            "release_quarter": ((release_month - 1) // 3) + 1,
            "is_holiday_window": int(release_month in [6, 7, 11, 12]),
            "is_animation": int("Animation" in selected_genres),
            "genre_animation": int("Animation" in selected_genres),
            "genre_adventure": int("Adventure" in selected_genres),
            "genre_family": int("Family" in selected_genres),
            "genre_drama": int("Drama" in selected_genres),
            "genre_fantasy": int("Fantasy" in selected_genres),
            "genre_comedy": int("Comedy" in selected_genres),
            "genre_action": int("Action" in selected_genres),
            "original_language": original_language,
            "production_country": production_country,
        }

        features_df = pd.DataFrame([features])[artifact["feature_columns"]]
        prediction = artifact["pipeline"].predict(features_df)[0]
        probability = artifact["pipeline"].predict_proba(features_df)[0][1]

        left, right = st.columns([1, 1])
        with left:
            st.success(
                f"{title}: {'succès probable' if prediction == 1 else 'potentiel plus incertain'} "
                f"avec une probabilité estimée de {probability:.1%}"
            )
        with right:
            benchmark = df[["title", "budget_musd", "vote_average", "popularity_log", "revenue_musd"]].copy()
            benchmark["distance"] = (
                (benchmark["budget_musd"] - budget_musd).abs()
                + (benchmark["vote_average"] - vote_average).abs() * 10
                + (benchmark["popularity_log"] - popularity_log).abs() * 10
            )
            similar = benchmark.sort_values("distance").head(5).drop(columns="distance")
            st.write("**Titres comparables dans la base**")
            st.dataframe(similar, use_container_width=True, hide_index=True)


def main() -> None:
    df = load_dataset()
    filtered_df = apply_filters(df)
    artifact = load_model()

    render_header(filtered_df)

    if filtered_df.empty:
        st.warning("Aucun titre ne correspond aux filtres sélectionnés.")
        return

    tab1, tab2, tab3, tab4 = st.tabs(["Vue d'ensemble", "Marché", "Performance", "Modélisation"])

    with tab1:
        render_overview(filtered_df)
        render_explorer(filtered_df)

    with tab2:
        render_market_view(filtered_df)

    with tab3:
        render_performance_view(filtered_df)

    with tab4:
        render_model_view(filtered_df, artifact)


if __name__ == "__main__":
    main()
