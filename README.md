# Movie Streaming Recommendation Project

This project implements topic 23: a movie streaming user behavior sorting and recommendation system.

## Data

- MovieLens small dataset: `data/ml-latest-small`
- Netflix Prize dataset: `data/netflix-prize/download`

The project is organized as isolated dataset pipelines. MovieLens is the default runnable pipeline because it includes movie titles, genres, ratings, and tags. Netflix Prize has its own placeholder pipeline and should be implemented separately so it does not affect MovieLens behavior.

Raw dataset files are intentionally not committed to GitHub because they can be large. Download or place the required dataset under `data/` before regenerating outputs.

## Dataset Architecture

- Shared code lives in `src/core/` and `src/algorithms/`.
- MovieLens-specific loading, profiling, search, recommendation, experiments, and frontend export live in `src/datasets/movielens/`.
- Netflix-specific work belongs in `src/datasets/netflix/`.
- Generated outputs are isolated by dataset, for example `output/movielens/`.

## Run

Create a local Python virtual environment first:

Windows:

```powershell
.\setup_venv.ps1
```

macOS / Linux:

```bash
chmod +x ./setup_venv.sh ./run.sh ./start_frontend.sh
./setup_venv.sh
```

Then run the command-line demos:

Windows:

```powershell
.\run.ps1 demo
.\run.ps1 --dataset movielens demo
.\run.ps1 top -n 10 --algorithm heap
.\run.ps1 top -n 10 --algorithm merge
.\run.ps1 search title "Toy Story"
.\run.ps1 search genre Comedy
.\run.ps1 search tag funny
.\run.ps1 recommend "Toy Story"
.\run.ps1 experiment
```

macOS / Linux:

```bash
./run.sh demo
./run.sh --dataset movielens demo
./run.sh top -n 10 --algorithm heap
./run.sh top -n 10 --algorithm merge
./run.sh search title "Toy Story"
./run.sh search genre Comedy
./run.sh search tag funny
./run.sh recommend "Toy Story"
./run.sh experiment
```

## Frontend Dashboard

Windows:

```powershell
.\start_frontend.ps1
.\start_frontend.ps1 -Dataset movielens
```

macOS / Linux:

```bash
chmod +x ./start_frontend.sh
./start_frontend.sh
./start_frontend.sh 8013 movielens
```

Then open:

```text
http://127.0.0.1:8013/
```

The startup script serves the frontend and a FastAPI backend from the same port. The dashboard uses:

- `GET /api/dashboard` for dataset summary and runtime CSV data.
- `GET /api/top?n=10&algorithm=heap` for Top-N heap recommendations.
- `GET /api/search?kind=title&query=Toy%20Story&n=20` for indexed search.
- `GET /api/recommend?title=Toy%20Story&n=10` for similar movie recommendations.
- `POST /api/events` for browser-session behavior tracking.
- `GET /api/for-you?session_id=...&n=10` for personalized For You recommendations.

FastAPI docs are available at:

```text
http://127.0.0.1:8013/docs
```

The dashboard shows Top-N recommendations, personalized For You recommendations, title/genre/tag search, similar movie recommendation, and runtime comparisons.

## Implemented Algorithms

- Merge sort for movie ranking.
- Heap sort for full ranking experiments.
- Top-N heap selection for movie recommendations without sorting the full movie list.
- Linear search for baseline movie lookup.
- Dictionary/inverted-index search for faster title, genre, and tag lookup.
- Similar movie recommendation based on shared genres, shared tags, and comprehensive score.
- Personalized For You recommendation based on backend behavior events, sparse content vectors, inverted-index cosine scoring, score-quality boosts, and a diverse high-score cold start fallback.

## Outputs

Generated MovieLens files are saved under `output/movielens/`:

- `movie_profiles.csv`
- `sorting_runtime.csv`
- `search_runtime.csv`
- `runtime_chart.svg`

Optional static export payloads can still be generated under `web/data/`:

- `movielens-dashboard-data.json`
- `dashboard-data.json` as the default active dashboard payload
