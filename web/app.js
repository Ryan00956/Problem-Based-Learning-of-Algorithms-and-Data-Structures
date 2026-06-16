const state = {
  data: null,
};

const numberFmt = new Intl.NumberFormat("en-US");

function normalize(value) {
  return String(value || "").trim().toLowerCase();
}

function scoreMovie(movie) {
  return Number(movie.comprehensive_score || 0);
}

function mergeSort(items, keyFn, reverse = true) {
  const values = [...items];
  if (values.length <= 1) return values;
  const mid = Math.floor(values.length / 2);
  const left = mergeSort(values.slice(0, mid), keyFn, reverse);
  const right = mergeSort(values.slice(mid), keyFn, reverse);
  const result = [];
  let i = 0;
  let j = 0;
  while (i < left.length && j < right.length) {
    const before = reverse ? keyFn(left[i]) >= keyFn(right[j]) : keyFn(left[i]) <= keyFn(right[j]);
    result.push(before ? left[i++] : right[j++]);
  }
  return result.concat(left.slice(i), right.slice(j));
}

function heapSort(items, keyFn, reverse = true) {
  const heap = [];
  const higherPriority = (a, b) => (reverse ? keyFn(a) > keyFn(b) : keyFn(a) < keyFn(b));
  const siftUp = (index) => {
    while (index > 0) {
      const parent = Math.floor((index - 1) / 2);
      if (!higherPriority(heap[index], heap[parent])) break;
      [heap[index], heap[parent]] = [heap[parent], heap[index]];
      index = parent;
    }
  };
  const siftDown = (index) => {
    while (true) {
      const left = index * 2 + 1;
      const right = left + 1;
      let best = index;
      if (left < heap.length && higherPriority(heap[left], heap[best])) best = left;
      if (right < heap.length && higherPriority(heap[right], heap[best])) best = right;
      if (best === index) break;
      [heap[index], heap[best]] = [heap[best], heap[index]];
      index = best;
    }
  };
  for (const item of items) {
    heap.push(item);
    siftUp(heap.length - 1);
  }
  const result = [];
  while (heap.length) {
    const root = heap[0];
    const last = heap.pop();
    if (heap.length) {
      heap[0] = last;
      siftDown(0);
    }
    result.push(root);
  }
  return result;
}

function renderStats() {
  const { summary } = state.data;
  document.querySelector("#movieCount").textContent = numberFmt.format(summary.movie_count);
  document.querySelector("#ratingCount").textContent = numberFmt.format(summary.rating_count);
  document.querySelector("#tagCount").textContent = numberFmt.format(summary.tag_count);
  document.querySelector("#userCount").textContent = numberFmt.format(summary.user_count);
  document.querySelector("#datasetStatus").textContent = "MovieLens loaded";
}

function renderTopMovies() {
  const limit = Number(document.querySelector("#topLimit").value);
  const algorithm = document.querySelector("#algorithmSelect").value;
  const sorted = algorithm === "merge"
    ? mergeSort(state.data.topMovies, scoreMovie, true)
    : heapSort(state.data.topMovies, scoreMovie, true);
  const rows = sorted.slice(0, limit).map((movie, index) => `
    <tr>
      <td>${index + 1}</td>
      <td><div class="movie-title">${movie.title}</div></td>
      <td>${genreTags(movie.genres)}</td>
      <td>${Number(movie.avg_rating).toFixed(2)}</td>
      <td>${numberFmt.format(movie.rating_count)}</td>
      <td class="score">${Number(movie.comprehensive_score).toFixed(2)}</td>
    </tr>
  `).join("");
  document.querySelector("#topMoviesBody").innerHTML = rows;
}

function genreTags(genres) {
  return (genres || []).slice(0, 4).map((genre) => `<span class="tag">${genre}</span>`).join("");
}

function resultItem(movie, extra = "") {
  return `
    <div class="result-item">
      <strong>${movie.title}</strong>
      <span>rating ${Number(movie.avg_rating).toFixed(2)} · count ${numberFmt.format(movie.rating_count)} · score ${Number(movie.comprehensive_score).toFixed(2)}${extra}</span>
      <div>${genreTags(movie.genres)}</div>
    </div>
  `;
}

function runSearch() {
  const kind = document.querySelector("#searchKind").value;
  const query = normalize(document.querySelector("#searchInput").value);
  const started = performance.now();
  let results = [];
  if (!query) {
    results = [];
  } else if (kind === "title") {
    results = state.data.movies.filter((movie) => normalize(movie.title).includes(query));
  } else if (kind === "genre") {
    results = state.data.movies.filter((movie) => (movie.genres || []).some((genre) => normalize(genre) === query));
  } else {
    results = state.data.movies.filter((movie) => (movie.tags || []).some((tag) => normalize(tag).includes(query)));
  }
  const elapsed = performance.now() - started;
  document.querySelector("#searchMeta").textContent = `${results.length} results · ${elapsed.toFixed(3)} ms in browser`;
  document.querySelector("#searchResults").innerHTML = results.slice(0, 20).map((movie) => resultItem(movie)).join("") || "<div class=\"result-item\"><strong>No result</strong><span>Try Comedy, funny, or Toy Story.</span></div>";
}

function recommendSimilar() {
  const query = normalize(document.querySelector("#similarInput").value);
  const target = state.data.movies.find((movie) => normalize(movie.title).includes(query));
  if (!target) {
    document.querySelector("#similarTarget").textContent = "No target movie found.";
    document.querySelector("#similarResults").innerHTML = "";
    return;
  }
  const targetGenres = new Set((target.genres || []).map(normalize));
  const targetTags = new Set((target.tags || []).map(normalize));
  const candidates = state.data.movies
    .filter((movie) => movie.movieId !== target.movieId)
    .map((movie) => {
      const genreOverlap = (movie.genres || []).filter((genre) => targetGenres.has(normalize(genre))).length;
      const tagOverlap = (movie.tags || []).filter((tag) => targetTags.has(normalize(tag))).length;
      return {
        ...movie,
        similarity_score: genreOverlap * 10 + tagOverlap * 15 + scoreMovie(movie) * 0.1,
        shared_genres: genreOverlap,
        shared_tags: tagOverlap,
      };
    })
    .filter((movie) => movie.shared_genres || movie.shared_tags);
  const ranked = heapSort(candidates, (movie) => movie.similarity_score, true).slice(0, 10);
  document.querySelector("#similarTarget").textContent = `Target: ${target.title}`;
  document.querySelector("#similarResults").innerHTML = ranked.map((movie) => {
    const extra = ` · shared genres ${movie.shared_genres} · shared tags ${movie.shared_tags}`;
    return resultItem(movie, extra);
  }).join("");
}

function renderCharts() {
  const sortRows = state.data.sortRuntime || [];
  const maxSort = Math.max(...sortRows.flatMap((row) => [Number(row.merge_sort_seconds), Number(row.heap_sort_seconds)]), 0.001);
  document.querySelector("#sortChart").innerHTML = sortRows.map((row) => `
    ${barRow(`${row.data_size}`, Number(row.merge_sort_seconds), maxSort, "merge", "merge")}
    ${barRow("", Number(row.heap_sort_seconds), maxSort, "heap", "heap")}
  `).join("");

  const searchRows = state.data.searchRuntime || [];
  const maxSearch = Math.max(...searchRows.flatMap((row) => [Number(row.linear_seconds), Number(row.index_seconds)]), 0.001);
  document.querySelector("#searchChart").innerHTML = searchRows.map((row) => `
    ${barRow(row.query_type, Number(row.linear_seconds), maxSearch, "linear", "linear")}
    ${barRow("", Number(row.index_seconds), maxSearch, "index", "index")}
  `).join("");
}

function barRow(label, value, max, kind, text) {
  const width = Math.max(2, (value / max) * 100);
  return `
    <div class="bar-row">
      <span>${label}</span>
      <div class="bar-track"><div class="bar ${kind}" style="width:${width}%"></div></div>
      <span>${text} ${value.toFixed(5)}s</span>
    </div>
  `;
}

async function init() {
  const response = await fetch("./data/dashboard-data.json");
  state.data = await response.json();
  renderStats();
  renderTopMovies();
  runSearch();
  recommendSimilar();
  renderCharts();
}

document.querySelector("#topLimit").addEventListener("change", renderTopMovies);
document.querySelector("#algorithmSelect").addEventListener("change", renderTopMovies);
document.querySelector("#searchButton").addEventListener("click", runSearch);
document.querySelector("#searchInput").addEventListener("keydown", (event) => {
  if (event.key === "Enter") runSearch();
});
document.querySelector("#similarButton").addEventListener("click", recommendSimilar);
document.querySelector("#similarInput").addEventListener("keydown", (event) => {
  if (event.key === "Enter") recommendSimilar();
});

init().catch((error) => {
  document.querySelector("#datasetStatus").textContent = "Failed to load data";
  console.error(error);
});
