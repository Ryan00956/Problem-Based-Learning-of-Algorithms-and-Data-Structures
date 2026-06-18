# Report Outline

## 1. Background

This project builds a movie streaming recommendation system based on user rating behavior.
The main goal is to rank movies, search movie information efficiently, and recommend similar movies.

## 2. Dataset

Main runnable dataset: MovieLens `ml-latest-small`.

- `movies.csv`: movie id, title, genres.
- `ratings.csv`: user id, movie id, rating, timestamp.
- `tags.csv`: user id, movie id, text tag, timestamp.

Extension dataset boundary: Netflix Prize has a separate dataset pipeline placeholder. Its loader, preprocessing, and recommendation algorithms should be implemented under `src/datasets/netflix/` without changing the MovieLens pipeline.

## 3. Data Preprocessing

For each movie, the system computes:

- Average rating.
- Bayesian trusted rating.
- Rating count.
- Recent rating count.
- Tag count.
- Genre list.
- Tag list.
- Comprehensive score.

Comprehensive score combines Bayesian rating quality, popularity, tag activity, and recent activity.
The Bayesian rating pulls low-sample movies toward the global MovieLens average, so one 5-star rating does not receive the full rating-quality score.

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

- `output/movielens/sorting_runtime.csv`
- `output/movielens/search_runtime.csv`
- `output/movielens/runtime_chart.svg`

Experiment comparisons:

- Merge sort vs heap sort.
- Linear search vs index search.
- MovieLens dataset.

## 8. Conclusion

The project shows that algorithm choice affects system efficiency.
Sorting algorithms are useful for Top-N ranking, while indexes greatly improve query speed.
The combination of sorting, search, and simple similarity scoring forms a complete recommendation workflow.
