const state = {
  data: null,
  sessionId: getSessionId(),
  movieCache: new Map(),
  activeMovieId: null,
  activeRecommendationSource: null,
  topRequest: 0,
  forYouRequest: 0,
  searchRequest: 0,
  similarRequest: 0,
  tagAliasRequest: 0,
  tagSemanticRequest: 0,
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

function getSessionId() {
  const key = "movie_lab_session_id";
  const existing = window.localStorage.getItem(key);
  if (existing) return existing;
  const next = window.crypto?.randomUUID?.() || `session-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  window.localStorage.setItem(key, next);
  return next;
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
    throw new Error(payload.detail || payload.error || `Request failed: ${response.status}`);
  }
  return payload;
}

async function apiPost(path, body = {}) {
  const response = await fetch(new URL(path, window.location.origin), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || payload.error || `Request failed: ${response.status}`);
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

function cacheMovie(movie) {
  const movieId = movie?.movieId ?? movie?.movie_id;
  if (movieId !== undefined && movieId !== null) {
    const cacheKey = String(movieId);
    state.movieCache.set(cacheKey, { ...(state.movieCache.get(cacheKey) || {}), ...movie });
  }
  return movieId;
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
    const rows = payload.items.map((movie, index) => {
      const movieId = cacheMovie(movie);
      return `
        <tr class="movie-row" data-movie-id="${escapeHtml(movieId)}" data-recommendation-source="top" tabindex="0">
          <td>${index + 1}</td>
          <td><div class="movie-title">${escapeHtml(movie.title)}</div></td>
          <td>${genreTags(movie.genres)}</td>
          <td>${Number(movie.avg_rating).toFixed(2)}</td>
          <td>${Number(movie.bayesian_rating ?? movie.avg_rating).toFixed(2)}</td>
          <td>${numberFmt.format(movie.rating_count)}</td>
          <td class="score">${Number(movie.comprehensive_score).toFixed(2)}</td>
        </tr>
      `;
    }).join("");
    body.innerHTML = rows;
  } catch (error) {
    body.innerHTML = `<tr><td colspan="7">${escapeHtml(error.message)}</td></tr>`;
  }
}

function genreTags(genres) {
  return (genres || []).slice(0, 4).map((genre) => `<span class="tag">${escapeHtml(genre)}</span>`).join("");
}

function resultItem(movie, extra = "") {
  const movieId = cacheMovie(movie);
  const movieAttr = movieId === undefined || movieId === null ? "" : ` data-movie-id="${escapeHtml(movieId)}"`;
  const sourceAttr = movie.recommendation_bucket ? ` data-recommendation-source="${escapeHtml(movie.recommendation_bucket)}"` : "";
  return `
    <div class="result-item"${movieAttr}${sourceAttr} tabindex="0">
      <strong>${escapeHtml(movie.title)}</strong>
      <span>rating ${Number(movie.avg_rating).toFixed(2)} | count ${numberFmt.format(movie.rating_count)} | score ${Number(movie.comprehensive_score).toFixed(2)}${extra}</span>
      <div>${genreTags(movie.genres)}</div>
      ${recommendationMeta(movie)}
      <button class="details-button ghost-button" type="button">Details</button>
    </div>
  `;
}

function recommendationMeta(movie) {
  if (!movie.recommendation_bucket && !movie.recommendation_reason) return "";
  const bucket = movie.recommendation_bucket ? `
    <span class="recommendation-bucket ${bucketClass(movie.recommendation_bucket)}">${escapeHtml(bucketLabel(movie.recommendation_bucket))}</span>
  ` : "";
  const reason = movie.recommendation_reason ? `<span class="recommendation-reason">${escapeHtml(movie.recommendation_reason)}</span>` : "";
  return `<div class="recommendation-meta">${bucket}${reason}</div>`;
}

function bucketLabel(bucket) {
  return {
    interest: "Interest",
    collaborative: "Similar users",
    explore: "Explore",
  }[bucket] || bucket;
}

function bucketClass(bucket) {
  if (bucket === "collaborative") return "bucket-collaborative";
  return bucket === "explore" ? "bucket-explore" : "bucket-interest";
}

function openMovieDetails(movieId, source = null) {
  const movie = state.movieCache.get(String(movieId));
  if (!movie) return;

  state.activeMovieId = Number(movieId);
  state.activeRecommendationSource = source || movie.recommendation_bucket || null;
  document.querySelector("#movieModalTitle").textContent = movie.title;
  document.querySelector("#movieModalSubtitle").textContent = `${(movie.genres || []).join(" | ") || "No genre"} | movieId ${movie.movieId}`;
  document.querySelector("#movieFeedbackStatus").textContent = "";
  document.querySelector("#movieModalBody").innerHTML = movieDetailsHtml(movie);
  document.querySelector("#movieModal").hidden = false;
  document.querySelector("#movieModalClose").focus();
}

function closeMovieDetails() {
  document.querySelector("#movieModal").hidden = true;
  state.activeMovieId = null;
  state.activeRecommendationSource = null;
}

function movieDetailsHtml(movie) {
  const scoreRows = [
    ["Average rating", fixed(movie.avg_rating)],
    ["Trusted rating", fixed(movie.bayesian_rating ?? movie.avg_rating)],
    ["Rating count", numberFmt.format(movie.rating_count || 0)],
    ["Recent ratings", numberFmt.format(movie.recent_rating_count || 0)],
    ["Comprehensive score", fixed(movie.comprehensive_score)],
    ["Rating score", fixed(movie.rating_score)],
    ["Popularity score", fixed(movie.popularity_score)],
    ["Tag score", fixed(movie.tag_score)],
    ["Tag evidence", fixed(movie.tag_evidence)],
    ["Freshness score", fixed(movie.freshness_score)],
  ];

  const contextRows = [
    ["Recommendation source", movie.recommendation_bucket ? bucketLabel(movie.recommendation_bucket) : null],
    ["Recommendation reason", movie.recommendation_reason],
    ["Similarity score", movie.similarity_score],
    ["Shared genres", movie.shared_genres],
    ["Shared tags", movie.shared_tags],
    ["Personal score", movie.personal_score],
    ["Vector similarity", movie.vector_similarity],
    ["Vector score", movie.vector_score],
    ["Quality boost", movie.quality_boost],
    ["Collaborative score", movie.collaborative_score],
    ["Similar users", movie.similar_user_count],
    ["Supporting users", movie.collaborative_support],
    ["Neighbor avg rating", movie.neighbor_avg_rating],
    ["Max user similarity", movie.max_user_similarity],
    ["Shared movie count", movie.shared_movie_count],
    ["Preference score", movie.preference_score],
    ["Seed similarity", movie.seed_similarity_score],
  ].filter(([, value]) => value !== undefined && value !== null);

  return `
    <div class="detail-section">
      <h3>Score Breakdown</h3>
      <div class="detail-grid">
        ${scoreRows.map(([label, value]) => detailMetric(label, value)).join("")}
      </div>
    </div>
    ${contextRows.length ? `
      <div class="detail-section">
        <h3>Recommendation Context</h3>
        <div class="detail-grid">
          ${contextRows.map(([label, value]) => detailMetric(label, typeof value === "number" ? fixed(value) : value)).join("")}
        </div>
      </div>
    ` : ""}
    <div class="detail-section">
      <h3>Genres</h3>
      <div>${genreTags(movie.genres)}</div>
    </div>
    <div class="detail-section">
      <h3>Tags</h3>
      <div>${detailTags(movie.tag_details || movie.tags)}</div>
    </div>
  `;
}

function detailMetric(label, value) {
  return `
    <div class="detail-metric">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(value)}</strong>
    </div>
  `;
}

function detailTags(tags) {
  const values = (tags || []).slice(0, 36);
  if (!values.length) return "<span class=\"empty-detail\">No tags available</span>";
  return values.map((tag) => {
    if (typeof tag === "string") {
      return `<span class="tag tag-neutral">${escapeHtml(tag)}</span>`;
    }
    const label = tag.display || tag.tag || "";
    const facet = tag.facet || "theme";
    const count = tag.count ? ` x${tag.count}` : "";
    const confidence = tag.confidence === undefined ? "" : ` ${Number(tag.confidence).toFixed(2)}`;
    return `
      <span class="tag tag-detail tag-${escapeHtml(facet)}" title="facet: ${escapeHtml(facet)}, confidence:${escapeHtml(confidence.trim() || "n/a")}">
        ${escapeHtml(label)}<small>${escapeHtml(facet)}${escapeHtml(count)}</small>
      </span>
    `;
  }).join("");
}

function fixed(value) {
  const number = Number(value || 0);
  return number.toFixed(2);
}

async function recordInteraction(event) {
  try {
    const payload = await apiPost("/api/events", {
      session_id: state.sessionId,
      ...event,
    });
    await renderForYou();
    return payload;
  } catch (error) {
    document.querySelector("#forYouMeta").textContent = `Personalization event failed: ${error.message}`;
    throw error;
  }
}

async function sendMovieFeedback(eventType) {
  if (!state.activeMovieId) return;

  const status = document.querySelector("#movieFeedbackStatus");
  const likeButton = document.querySelector("#movieLikeButton");
  const dislikeButton = document.querySelector("#movieDislikeButton");
  const isLike = eventType === "like";

  status.textContent = isLike ? "Saving like..." : "Saving preference...";
  likeButton.disabled = true;
  dislikeButton.disabled = true;

  try {
    await recordInteraction({
      event_type: eventType,
      movie_id: state.activeMovieId,
      source: state.activeRecommendationSource || "detail",
    });
    status.textContent = isLike ? "Liked. For You updated." : "Saved. For You updated.";
  } catch (error) {
    status.textContent = `Could not save: ${error.message}`;
  } finally {
    likeButton.disabled = false;
    dislikeButton.disabled = false;
  }
}

async function renderForYou() {
  const requestId = ++state.forYouRequest;
  const meta = document.querySelector("#forYouMeta");
  const profile = document.querySelector("#forYouProfile");
  const results = document.querySelector("#forYouResults");

  meta.textContent = "Loading personalized recommendations...";
  results.innerHTML = "";

  try {
    const payload = await apiGet("/api/for-you", { session_id: state.sessionId, n: 10 });
    if (requestId !== state.forYouRequest) return;
    const mode = payload.status === "personalized" ? "personalized" : "cold start";
    const scored = payload.scored_count === undefined ? "" : ` | ${payload.scored_count} scored`;
    const buckets = payload.bucket_counts ? ` | ${bucketCountText(payload.bucket_counts)}` : "";
    meta.textContent = `${payload.count} recommendations | ${payload.event_count} behavior events${scored}${buckets} | ${payload.elapsed_ms.toFixed(3)} ms via ${payload.engine} (${mode})`;
    profile.innerHTML = renderProfileChips(payload.profile);
    results.innerHTML = payload.items.map((movie) => {
      const personal = movie.personal_score === undefined ? "" : ` | personal ${Number(movie.personal_score).toFixed(2)}`;
      const vector = movie.vector_similarity === undefined ? "" : ` | vector ${Number(movie.vector_similarity).toFixed(3)}`;
      return resultItem(movie, personal + vector);
    }).join("");
  } catch (error) {
    meta.textContent = error.message;
    profile.innerHTML = "";
    results.innerHTML = "";
  }
}

function bucketCountText(counts) {
  const interest = counts.interest || 0;
  const collaborative = counts.collaborative || 0;
  const explore = counts.explore || 0;
  const parts = [];
  if (interest) parts.push(`${interest} interest`);
  if (collaborative) parts.push(`${collaborative} similar users`);
  if (explore) parts.push(`${explore} explore`);
  if (parts.length) return parts.join(" + ");
  return "mixed recommendations";
}

function renderProfileChips(profile) {
  const genres = profile?.top_genres || [];
  const tags = profile?.top_tags || [];
  const semanticTags = profile?.semantic_tags || [];
  const longGenres = profile?.long_term_genres || [];
  const shortGenres = profile?.short_term_genres || [];
  const chips = [
    ...(profile?.liked_movie_count ? [`liked: ${profile.liked_movie_count}`] : []),
    ...(profile?.disliked_movie_count ? [`not for me: ${profile.disliked_movie_count}`] : []),
    ...(profile?.short_weight ? [`short weight: ${Number(profile.short_weight).toFixed(2)}`] : []),
    ...(profile?.short_alignment ? [`short align: ${Number(profile.short_alignment).toFixed(2)}`] : []),
    ...longGenres.slice(0, 2).map((item) => `long: ${item.name}`),
    ...shortGenres.slice(0, 2).map((item) => `now: ${item.name}`),
    ...genres.map((item) => `genre: ${item.name}`),
    ...tags.map((item) => `tag: ${item.name}`),
    ...semanticTags.slice(0, 3).map((item) => `semantic: ${item.name}`),
  ];
  if (!chips.length) {
    return "<span class=\"profile-chip\">cold start: diverse high-score movies</span>";
  }
  return chips.slice(0, 8).map((label) => `<span class="profile-chip">${escapeHtml(label)}</span>`).join("");
}

async function renderTagSemantics() {
  const requestId = ++state.tagSemanticRequest;
  const query = document.querySelector("#semanticInput").value.trim();
  const meta = document.querySelector("#semanticMeta");
  const results = document.querySelector("#semanticResults");

  if (!query) {
    meta.textContent = "Enter a tag.";
    results.innerHTML = "";
    return;
  }

  meta.textContent = "Loading tag neighbors...";
  results.innerHTML = "";
  try {
    const payload = await apiGet("/api/tag-semantics", { tag: query, n: 8 });
    if (requestId !== state.tagSemanticRequest) return;
    const summary = payload.summary || {};
    meta.textContent = `${payload.tag} | ${payload.count} neighbors | ${summary.tag_count || 0} tags | ${summary.dimensions || 0} dimensions | ${summary.cache_status || "memory"} cache | ${payload.elapsed_ms.toFixed(3)} ms`;
    results.innerHTML = renderSemanticNeighbors(payload.neighbors || []);
  } catch (error) {
    meta.textContent = error.message;
    results.innerHTML = "";
  }
}

function renderSemanticNeighbors(neighbors) {
  if (!neighbors.length) {
    return "<div class=\"semantic-neighbor empty-neighbor\"><strong>No semantic neighbors found.</strong></div>";
  }
  return neighbors.map((item) => `
    <div class="semantic-neighbor">
      <strong>${escapeHtml(item.tag)}</strong>
      <span>similarity ${Number(item.similarity).toFixed(3)}</span>
      <span>${numberFmt.format(item.movie_count)} movies</span>
      <span>${numberFmt.format(item.shared_movies)} shared</span>
    </div>
  `).join("");
}

async function runSearch(options = {}) {
  const requestId = ++state.searchRequest;
  const kind = document.querySelector("#searchKind").value;
  const query = document.querySelector("#searchInput").value.trim();
  const shouldRecord = options.record !== false;
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
    if (shouldRecord) {
      await recordInteraction({ event_type: "search", kind, query });
    }
  } catch (error) {
    meta.textContent = error.message;
    results.innerHTML = "";
  }
}

async function recommendSimilar(options = {}) {
  const requestId = ++state.similarRequest;
  const title = document.querySelector("#similarInput").value.trim();
  const shouldRecord = options.record !== false;
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
    if (shouldRecord) {
      await recordInteraction({
        event_type: "similar",
        kind: "title",
        query: title,
        movie_id: payload.target.movieId,
      });
    }
  } catch (error) {
    targetEl.textContent = error.message;
    resultsEl.innerHTML = "";
  }
}

async function renderTagAliases() {
  const requestId = ++state.tagAliasRequest;
  const meta = document.querySelector("#tagAliasMeta");
  const configured = document.querySelector("#tagAliasConfigured");
  const candidates = document.querySelector("#tagAliasCandidates");

  meta.textContent = "Scanning tag alias candidates...";
  configured.innerHTML = "";
  candidates.innerHTML = "";

  try {
    const payload = await apiGet("/api/tag-alias-candidates", { n: 12 });
    if (requestId !== state.tagAliasRequest) return;
    const summary = payload.summary || {};
    meta.textContent = `${payload.count} pending | accepted ${summary.accepted_count || 0} | rejected ${summary.rejected_count || 0} | ignored ${summary.ignored_count || 0} | ${summary.raw_tag_count || 0} raw tags -> ${summary.canonical_tag_count || 0} canonical tags | ${payload.elapsed_ms.toFixed(3)} ms`;
    configured.innerHTML = renderConfiguredAliases(payload.configured_aliases || []);
    candidates.innerHTML = renderAliasCandidates(payload.candidates || []);
  } catch (error) {
    meta.textContent = error.message;
    configured.innerHTML = "";
    candidates.innerHTML = "";
  }
}

function renderConfiguredAliases(aliases) {
  const active = aliases
    .filter((item) => item.active_in_dataset)
    .sort((left, right) => Number(right.source_type === "accepted") - Number(left.source_type === "accepted"))
    .slice(0, 10);
  if (!active.length) {
    return "<span class=\"alias-chip\">No configured aliases found in this dataset sample</span>";
  }
  return active.map((item) => `
    <span class="alias-chip alias-${escapeHtml(item.source_type || "configured")}">${escapeHtml(item.source)} -> ${escapeHtml(item.target)} (${numberFmt.format(item.source_count)})</span>
  `).join("");
}

function renderAliasCandidates(candidates) {
  if (!candidates.length) {
    return "<div class=\"alias-row\"><strong>No alias candidates above the confidence threshold.</strong></div>";
  }
  return candidates.map((item) => `
    <div class="alias-row">
      <div class="alias-row-header">
        <div class="alias-pair">
          <span class="alias-source">${escapeHtml(item.source)}</span>
          <span class="alias-arrow">-></span>
          <span class="alias-target">${escapeHtml(item.target)}</span>
        </div>
        <span class="confidence-badge confidence-${escapeHtml(item.confidence_band)}">${escapeHtml(item.confidence_band)}</span>
      </div>
      <div class="alias-metrics">
        <span class="alias-metric">confidence <strong>${Number(item.confidence).toFixed(2)}</strong></span>
        <span class="alias-metric">text <strong>${Number(item.text_similarity).toFixed(2)}</strong></span>
        <span class="alias-metric">movies <strong>${numberFmt.format(item.movie_overlap)}</strong></span>
        <span class="alias-metric">source <strong>${numberFmt.format(item.source_count)}</strong></span>
        <span class="alias-metric">target <strong>${numberFmt.format(item.target_count)}</strong></span>
      </div>
      <div class="alias-reasons">
        ${(item.reasons || []).map((reason) => `<span class="alias-reason">${escapeHtml(reason)}</span>`).join("")}
      </div>
      <div class="alias-actions">
        <button class="alias-decision-button accept-alias" type="button" data-source="${escapeHtml(item.source)}" data-target="${escapeHtml(item.target)}" data-decision="accept">Accept</button>
        <button class="alias-decision-button reject-alias" type="button" data-source="${escapeHtml(item.source)}" data-target="${escapeHtml(item.target)}" data-decision="reject">Reject</button>
        <button class="alias-decision-button ignore-alias ghost-button" type="button" data-source="${escapeHtml(item.source)}" data-target="${escapeHtml(item.target)}" data-decision="ignore">Ignore</button>
      </div>
    </div>
  `).join("");
}

async function sendTagAliasDecision(button) {
  const source = button.dataset.source;
  const target = button.dataset.target;
  const decision = button.dataset.decision;
  const row = button.closest(".alias-row");
  if (!source || !target || !decision || !row) return;

  const buttons = row.querySelectorAll(".alias-decision-button");
  buttons.forEach((item) => {
    item.disabled = true;
  });
  row.classList.add("alias-row-saving");

  try {
    await apiPost("/api/tag-alias-decisions", { source, target, decision });
    row.classList.remove("alias-row-saving");
    row.classList.add(`alias-row-${decision}`);
    await renderTagAliases();
    if (decision === "accept") {
      await renderForYou();
      await renderTopMovies();
    }
  } catch (error) {
    row.classList.remove("alias-row-saving");
    row.insertAdjacentHTML("beforeend", `<div class="alias-error">${escapeHtml(error.message)}</div>`);
    buttons.forEach((item) => {
      item.disabled = false;
    });
  }
}

function renderCharts() {
  const sortRows = state.data.sortRuntime || [];
  const maxSort = Math.max(
    ...sortRows.flatMap((row) => [
      Number(row.merge_sort_seconds),
      Number(row.heap_sort_seconds),
      Number(row.top_n_heap_seconds || 0),
    ]),
    0.001
  );
  document.querySelector("#sortChart").innerHTML = sortRows.map((row) => `
    ${barRow(`${row.data_size}`, Number(row.merge_sort_seconds), maxSort, "merge", "merge")}
    ${barRow("", Number(row.heap_sort_seconds), maxSort, "heap", "heap")}
    ${row.top_n_heap_seconds ? barRow("", Number(row.top_n_heap_seconds), maxSort, "topn", "top-n") : ""}
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
  await renderForYou();
  await renderTopMovies();
  await runSearch({ record: false });
  await recommendSimilar({ record: false });
  await renderTagSemantics();
  await renderTagAliases();
  renderCharts();
}

document.querySelector("#topLimit").addEventListener("change", () => renderTopMovies());
document.querySelector("#algorithmSelect").addEventListener("change", () => renderTopMovies());
document.querySelector("#forYouRefresh").addEventListener("click", () => renderForYou());
document.querySelector("#forYouReset").addEventListener("click", async () => {
  await recordInteraction({ event_type: "reset" });
});
document.querySelector("#searchButton").addEventListener("click", () => runSearch());
document.querySelector("#searchInput").addEventListener("keydown", (event) => {
  if (event.key === "Enter") runSearch();
});
document.querySelector("#similarButton").addEventListener("click", () => recommendSimilar());
document.querySelector("#similarInput").addEventListener("keydown", (event) => {
  if (event.key === "Enter") recommendSimilar();
});
document.querySelector("#semanticButton").addEventListener("click", () => renderTagSemantics());
document.querySelector("#semanticInput").addEventListener("keydown", (event) => {
  if (event.key === "Enter") renderTagSemantics();
});
document.querySelector("#tagAliasRefresh").addEventListener("click", () => renderTagAliases());
document.addEventListener("click", (event) => {
  const aliasButton = event.target.closest(".alias-decision-button");
  if (aliasButton) {
    sendTagAliasDecision(aliasButton);
    return;
  }

  const item = event.target.closest(".result-item[data-movie-id], .movie-row[data-movie-id]");
  if (!item) return;
  const movieId = Number(item.dataset.movieId);
  const source = interactionSourceForItem(item);
  openMovieDetails(movieId, source);
  recordInteraction({ event_type: "view", movie_id: movieId, source });
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !document.querySelector("#movieModal").hidden) {
    closeMovieDetails();
    return;
  }
  if (event.key !== "Enter") return;
  if (event.target.closest?.(".details-button")) return;
  const item = event.target.closest?.(".result-item[data-movie-id], .movie-row[data-movie-id]");
  if (!item) return;
  const movieId = Number(item.dataset.movieId);
  const source = interactionSourceForItem(item);
  openMovieDetails(movieId, source);
  recordInteraction({ event_type: "view", movie_id: movieId, source });
});
document.querySelector("#movieModalClose").addEventListener("click", closeMovieDetails);
document.querySelector("#movieLikeButton").addEventListener("click", () => sendMovieFeedback("like"));
document.querySelector("#movieDislikeButton").addEventListener("click", () => sendMovieFeedback("dislike"));
document.querySelector("#movieModal").addEventListener("click", (event) => {
  if (event.target.id === "movieModal") {
    closeMovieDetails();
  }
});

init().catch((error) => {
  document.querySelector("#datasetStatus").textContent = "Failed to load backend API";
  console.error(error);
});

function interactionSourceForItem(item) {
  if (item.dataset.recommendationSource) return item.dataset.recommendationSource;
  if (item.closest("#searchResults")) return "search";
  if (item.closest("#similarResults")) return "similar";
  if (item.closest("#topMoviesBody")) return "top";
  return "detail";
}
