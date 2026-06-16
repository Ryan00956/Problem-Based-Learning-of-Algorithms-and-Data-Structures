# Report Outline

## 1. Background

This project builds a movie streaming recommendation system based on user rating behavior.
The main goal is to rank movies, search movie information efficiently, and recommend similar movies.

## 2. Dataset

Main dataset: MovieLens `ml-latest-small`.

- `movies.csv`: movie id, title, genres.
- `ratings.csv`: user id, movie id, rating, timestamp.
- `tags.csv`: user id, movie id, text tag, timestamp.

Extension dataset: Netflix Prize.

- `movie_titles.txt`: movie id, release year, title.
- `training_set`: one rating file per movie.

## 3. Data Preprocessing

For each movie, the system computes:

- Average rating.
- Rating count.
- Tag count.
- Genre list.
- Tag list.
- Comprehensive score.

Comprehensive score combines rating quality, popularity, and tag activity.

## 4. Sorting Algorithms

Implemented algorithms:

- Merge sort.
- Heap sort.

Sorting target:

- Sort movies by comprehensive score.
- Generate Top-N recommendation list.

Complexity:

- Merge sort: O(n log n).
- Heap sort: O(n log n).

## 5. Search Algorithms

Implemented search methods:

- Linear search.
- Dictionary and inverted-index search.

Search targets:

- Movie title.
- Genre.
- Tag.

Complexity:

- Linear search: O(n).
- Dictionary lookup: average O(1).
- Inverted-index lookup: close to O(1) plus result size.

## 6. Similar Movie Recommendation

The system recommends similar movies based on:

- Shared genres.
- Shared user tags.
- Comprehensive movie score.

The similarity score is used to rank candidate movies.

## 7. Experiments

Output files:

- `output/sorting_runtime.csv`
- `output/search_runtime.csv`
- `output/netflix_sorting_runtime.csv`
- `output/runtime_chart.svg`

Experiment comparisons:

- Merge sort vs heap sort.
- Linear search vs index search.
- MovieLens main dataset and Netflix sample dataset.

## 8. Conclusion

The project shows that algorithm choice affects system efficiency.
Sorting algorithms are useful for Top-N ranking, while indexes greatly improve query speed.
The combination of sorting, search, and simple similarity scoring forms a complete recommendation workflow.
