# Movie Streaming Recommendation Project

This project implements topic 23: a movie streaming user behavior sorting and recommendation system.

## Data

- MovieLens small dataset: `data/ml-latest-small`

The main system uses MovieLens because it includes movie titles, genres, ratings, and tags.

Raw dataset files are intentionally not committed to GitHub because they can be large. Download or place the MovieLens dataset under `data/` before regenerating outputs.

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
```

macOS / Linux:

```bash
chmod +x ./start_frontend.sh
./start_frontend.sh
```

Then open:

```text
http://127.0.0.1:8013/
```

The dashboard shows Top-N recommendations, title/genre/tag search, similar movie recommendation, and runtime comparisons.

## Implemented Algorithms

- Merge sort for movie ranking.
- Heap sort for movie ranking and Top-N recommendation.
- Linear search for baseline movie lookup.
- Dictionary/inverted-index search for faster title, genre, and tag lookup.
- Similar movie recommendation based on shared genres, shared tags, and comprehensive score.

## Outputs

Generated files are saved under `output/`:

- `movie_profiles.csv`
- `sorting_runtime.csv`
- `search_runtime.csv`
- `runtime_chart.svg`
