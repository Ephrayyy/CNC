# CineScope AI

Projet data science orienté **cinéma / audiovisuel / image animée**, pensé comme une vraie chaîne de collecte et d'analyse autour de **TMDb** et **OMDb**.

L'objectif est de construire une pipeline exploitable :

- extraction de vraies métadonnées films depuis des API externes ;
- stockage brut des réponses ;
- structuration tabulaire ;
- nettoyage et enrichissement ;
- modélisation ;
- restitution dans un dashboard.

## 1. Pourquoi ce projet colle à la fiche de poste

La fiche de poste parle de :

- extraction automatisée de données via API ;
- alimentation de bases internes ;
- analyse de liens entre notes, audience, exploitation et performance ;
- mise en place d'outils IA pour analyser des fichiers / données ;
- production d'études et d'indicateurs pour le cinéma, l'audiovisuel et l'image animée.

Ce projet répond à ces attentes avec une version compacte, démontrable et défendable :

- `src/api/` : collecte de données externes ;
- `src/processing/` : nettoyage et feature engineering ;
- `src/models/` : modèle de scoring / classification ;
- `src/dashboard/` : restitution visuelle et simulateur.

## 2. Structure du projet

```text
cine_scope_ai/
├── data/
│   ├── raw/sample_movies.csv
│   └── processed/
├── src/
│   ├── api/
│   ├── dashboard/
│   ├── models/
│   └── processing/
└── notebooks/
```

## 3. Sources réelles

### TMDb

Le projet extrait les films d'animation depuis TMDb via l'endpoint de découverte, puis appelle les détails de chaque film.

Selon la documentation officielle TMDb, l'authentification applicative se fait soit avec :

- un `api_key` en paramètre ;
- un `Bearer token` dans l'en-tête `Authorization`.

Source officielle :
- https://developer.themoviedb.org/docs/authentication-application

### OMDb

Le projet peut enrichir les films via leur `imdb_id` avec OMDb.

La documentation officielle OMDb indique l'usage de requêtes du type :

- `?apikey=...&i=tt....`
- ou `?apikey=...&t=...`

Source officielle :
- https://www.omdbapi.com/

Le fichier `data/raw/sample_movies.csv` reste présent uniquement comme secours hors ligne, mais la source prioritaire du projet est `data/raw/tmdb_animation_movies_latest.csv` quand une extraction réelle a déjà été faite.

## 4. Pipeline

### Extraction

- `TMDbClient` interroge réellement TMDb.
- `IMDbClient` enrichit réellement les titres via OMDb.
- `src/api/extract_movies.py` pilote l'extraction paginée par année, sauvegarde les réponses brutes en `jsonl` et la table structurée en `csv`.

Pourquoi OMDb :
IMDb ne fournit pas une API publique simple équivalente à TMDb. Pour récupérer des champs IMDb de façon pragmatique, OMDb est la passerelle la plus simple.

### Nettoyage

Dans `src/processing/cleaning.py` :

- normalisation des colonnes ;
- conversion des dates et variables numériques ;
- suppression des doublons ;
- création du `roi` ;
- flag `is_animation`.

### Feature engineering

Dans `src/processing/feature_engineering.py` :

- variables temporelles : année, mois, trimestre, fenêtre vacances ;
- variables business : budget, profit, logs de popularité et de volume d'avis ;
- encodage léger des genres ;
- construction d'un `success_score` puis d'un `success_label`.

### Modélisation

Dans `src/models/train_model.py` :

- comparaison entre `LogisticRegression` et `RandomForestClassifier` ;
- sélection du meilleur modèle selon le score F1 ;
- sauvegarde de l'artefact avec `joblib`.

Le but n'est pas de produire un modèle “parfait”, mais un pipeline lisible :

- explicable ;
- reproductible ;
- améliorable.

### Dashboard

Dans `src/dashboard/app.py` :

- KPIs ;
- visualisations budget / revenu ;
- classement des titres à potentiel ;
- simulateur de succès d'un projet.

## 5. Variables d'environnement

Avant toute extraction réelle, configure au moins TMDb :

```bash
export TMDB_BEARER_TOKEN="ton_token_tmdb"
```

ou

```bash
export TMDB_API_KEY="ta_cle_tmdb"
```

Pour l'enrichissement OMDb :

```bash
export OMDB_API_KEY="ta_cle_omdb"
```

## 6. Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 7. Exécution

### Extraire de vraies données

```bash
python3 src/api/extract_movies.py --start-year 2018 --end-year 2025 --max-pages-per-year 5
```

Options utiles :

```bash
python3 src/api/extract_movies.py --start-year 2020 --end-year 2025 --max-pages-per-year 3 --original-language fr
python3 src/api/extract_movies.py --start-year 2015 --end-year 2025 --max-pages-per-year 5 --skip-omdb
```

### Nettoyer les données extraites

```bash
python3 src/processing/cleaning.py
```

### Entraîner le modèle

```bash
python3 src/models/train_model.py
```

### Lancer le dashboard

```bash
streamlit run src/dashboard/app.py
```

## 8. Comment l'expliquer en entretien

Tu peux présenter le projet en 5 étapes :

1. **Problème métier** : aider à la veille et à l'analyse de la performance de films, surtout animation / audiovisuel.
2. **Collecte** : récupérer des données structurées via API.
3. **Transformation** : fiabiliser les données et créer des variables métier.
4. **Modèle** : estimer un potentiel de succès pour aider à prioriser l'analyse.
5. **Restitution** : rendre le résultat exploitable dans un dashboard.

## 9. Améliorations possibles

- enrichir avec données box office, festivals, aides publiques ou diffusion ;
- ajouter une analyse NLP sur synopsis ou critiques ;
- créer une segmentation par type d'oeuvre ;
- connecter une base SQL ou un entrepôt analytique.

## 10. Limites assumées

- dépendance aux quotas et aux clés API ;
- couverture limitée aux champs exposés par les APIs ;
- modèle initial encore dépendant de la qualité et du volume de l'extraction ;
- score de succès construit comme proxy métier ;
- certaines données streaming/séries n'ont pas de revenus box office, donc le signal est partiel.
