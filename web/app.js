const state = {
  data: null,
  topRequest: 0,
  searchRequest: 0,
  similarRequest: 0,
};

const numberFmt = new Intl.NumberFormat("en-US");

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

async function apiGet(path, params = {}) {
  const url = new URL(path, window.location.origin);
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null) {
      url.searchParams.set(key, value);
    }
  });

  const response = await fetch(url);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `Request failed: ${response.status}`);
  }
  return payload;
}

function renderStats() {
  const { summary } = state.data;
  const datasetName = state.data.dataset?.display_name || "MovieLens";
  document.querySelector("#movieCount").textContent = numberFmt.format(summary.movie_count);
  document.querySelector("#ratingCount").textContent = numberFmt.format(summary.rating_count);
  document.querySelector("#tagCount").textContent = numberFmt.format(summary.tag_count);
  document.querySelector("#userCount").textContent = numberFmt.format(summary.user_count);
  document.querySelector("#datasetStatus").textContent = `${datasetName} loaded from backend`;
}

async function renderTopMovies() {
  const requestId = ++state.topRequest;
  const limit = Number(document.querySelector("#topLimit").value);
  const algorithm = document.querySelector("#algorithmSelect").value;
  const body = document.querySelector("#topMoviesBody");
  body.innerHTML = "<tr><td colspan=\"7\">Loading...</td></tr>";

  try {
    const payload = await apiGet("/api/top", { n: limit, algorithm });
    if (requestId !== state.topRequest) return;
    const rows = payload.items.map((movie, index) => `
      <tr>
        <td>${index + 1}</td>
        <td><div class="movie-title">${escapeHtml(movie.title)}</div></td>
        <td>${genreTags(movie.genres)}</td>
        <td>${Number(movie.avg_rating).toFixed(2)}</td>
        <td>${Number(movie.bayesian_rating ?? movie.avg_rating).toFixed(2)}</td>
        <td>${numberFmt.format(movie.rating_count)}</td>
        <td class="score">${Number(movie.comprehensive_score).toFixed(2)}</td>
      </tr>
    `).join("");
    body.innerHTML = rows;
  } catch (error) {
    body.innerHTML = `<tr><td colspan="7">${escapeHtml(error.message)}</td></tr>`;
  }
}

function genreTags(genres) {
  return (genres || []).slice(0, 4).map((genre) => `<span class="tag">${escapeHtml(genre)}</span>`).join("");
}

function resultItem(movie, extra = "") {
  return `
    <div class="result-item">
      <strong>${escapeHtml(movie.title)}</strong>
      <span>rating ${Number(movie.avg_rating).toFixed(2)} | count ${numberFmt.format(movie.rating_count)} | score ${Number(movie.comprehensive_score).toFixed(2)}${extra}</span>
      <div>${genreTags(movie.genres)}</div>
    </div>
  `;
}

async function runSearch() {
  const requestId = ++state.searchRequest;
  const kind = document.querySelector("#searchKind").value;
  const query = document.querySelector("#searchInput").value.trim();
  const meta = document.querySelector("#searchMeta");
  const results = document.querySelector("#searchResults");
  if (!query) {
    meta.textContent = "Enter a search query.";
    results.innerHTML = "";
    return;
  }

  meta.textContent = "Searching backend...";
  results.innerHTML = "";
  try {
    const payload = await apiGet("/api/search", { kind, query, n: 20 });
    if (requestId !== state.searchRequest) return;
    meta.textContent = `${payload.count} results | ${payload.elapsed_ms.toFixed(3)} ms via backend ${payload.engine}`;
    results.innerHTML = payload.items.map((movie) => resultItem(movie)).join("") || "<div class=\"result-item\"><strong>No result</strong><span>Try Comedy, funny, or Toy Story.</span></div>";
  } catch (error) {
    meta.textContent = error.message;
    results.innerHTML = "";
  }
}

async function recommendSimilar() {
  const requestId = ++state.similarRequest;
  const title = document.querySelector("#similarInput").value.trim();
  const targetEl = document.querySelector("#similarTarget");
  const resultsEl = document.querySelector("#similarResults");
  if (!title) {
    targetEl.textContent = "Enter a movie title.";
    resultsEl.innerHTML = "";
    return;
  }

  targetEl.textContent = "Recommending from backend...";
  resultsEl.innerHTML = "";
  try {
    const payload = await apiGet("/api/recommend", { title, n: 10 });
    if (requestId !== state.similarRequest) return;
    if (!payload.target) {
      targetEl.textContent = "No target movie found.";
      resultsEl.innerHTML = "";
      return;
    }
    targetEl.textContent = `Target: ${payload.target.title} | ${payload.elapsed_ms.toFixed(3)} ms via backend ${payload.engine}`;
    resultsEl.innerHTML = payload.items.map((movie) => {
      const extra = ` | shared genres ${movie.shared_genres} | shared tags ${movie.shared_tags}`;
      return resultItem(movie, extra);
    }).join("");
  } catch (error) {
    targetEl.textContent = error.message;
    resultsEl.innerHTML = "";
  }
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
      <span>${escapeHtml(label)}</span>
      <div class="bar-track"><div class="bar ${kind}" style="width:${width}%"></div></div>
      <span>${escapeHtml(text)} ${value.toFixed(5)}s</span>
    </div>
  `;
}

async function init() {
  state.data = await apiGet("/api/dashboard");
  renderStats();
  await renderTopMovies();
  await runSearch();
  await recommendSimilar();
  renderCharts();
}

document.querySelector("#topLimit").addEventListener("change", () => renderTopMovies());
document.querySelector("#algorithmSelect").addEventListener("change", () => renderTopMovies());
document.querySelector("#searchButton").addEventListener("click", () => runSearch());
document.querySelector("#searchInput").addEventListener("keydown", (event) => {
  if (event.key === "Enter") runSearch();
});
document.querySelector("#similarButton").addEventListener("click", () => recommendSimilar());
document.querySelector("#similarInput").addEventListener("keydown", (event) => {
  if (event.key === "Enter") recommendSimilar();
});

init().catch((error) => {
  document.querySelector("#datasetStatus").textContent = "Failed to load backend API";
  console.error(error);
});
