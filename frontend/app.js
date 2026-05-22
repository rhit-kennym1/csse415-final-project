// In production (Firebase Hosting), /api/** is rewritten to Cloud Run, so
// same-origin is correct. In local dev (frontend on :5000, backend on :8000)
// we point directly at the backend instead.
const API_BASE = window.location.port === "5000"
  ? "http://localhost:8000"
  : window.location.origin;

const sections = {
  input:    document.getElementById("input-section"),
  loading:  document.getElementById("loading-section"),
  results:  document.getElementById("results-section"),
  notfound: document.getElementById("notfound-section"),
  error:    document.getElementById("error-section"),
};

function showOnly(name) {
  for (const [key, el] of Object.entries(sections)) {
    el.hidden = key !== name;
  }
}

function verdictBadge(verdict) {
  const span = document.createElement("span");
  span.className = `badge badge-${verdict}`;
  span.textContent = verdict;
  return span;
}

function renderModelPredictions(predictions) {
  const ul = document.getElementById("model-predictions");
  ul.innerHTML = "";
  for (const p of predictions) {
    const li = document.createElement("li");
    const name = document.createElement("strong");
    name.textContent = p.model;

    li.appendChild(name);
    li.appendChild(verdictBadge(p.verdict));
    ul.appendChild(li);
  }
}

function renderSafeBrowsing(sb) {
  const container = document.getElementById("safe-browsing-result");
  container.innerHTML = "";
  container.appendChild(verdictBadge(sb.verdict));
  if (sb.threat_types && sb.threat_types.length) {
    const list = document.createElement("ul");
    for (const t of sb.threat_types) {
      const li = document.createElement("li");
      li.textContent = t.replace(/_/g, " ").toLowerCase();
      list.appendChild(li);
    }
    container.appendChild(list);
  }
  if (sb.error) {
    const note = document.createElement("p");
    note.className = "footnote";
    note.textContent = `Safe Browsing unavailable: ${sb.error}`;
    container.appendChild(note);
  }
}

function renderAgreement(predictions, sb) {
  const line = document.getElementById("agreement-line");
  if (sb.verdict === "safe" || sb.verdict === "phishing" ||
      sb.verdict === "malware" || sb.verdict === "unwanted") {
    const sbSaysPhish = sb.verdict !== "safe";
    const agreeing = predictions.filter(p =>
      (p.verdict === "phishing") === sbSaysPhish
    ).length;
    line.textContent =
      `${agreeing} of ${predictions.length} models agree with Google Safe Browsing.`;
  } else {
    line.textContent = "Google Safe Browsing did not return a verdict.";
  }
}

function renderFeatures(featuresDisplay) {
  const ul = document.getElementById("feature-list");
  ul.innerHTML = "";
  if (!featuresDisplay) return;
  for (const f of featuresDisplay) {
    const li = document.createElement("li");

    const dot = document.createElement("span");
    dot.className = `signal-dot signal-${f.signal}`;

    const label = document.createElement("span");
    label.className = "feature-label";
    label.textContent = f.label;

    const value = document.createElement("span");
    value.className = `feature-value feature-${f.signal}`;
    value.textContent = f.value;

    li.appendChild(dot);
    li.appendChild(label);
    li.appendChild(value);
    ul.appendChild(li);
  }
}

function renderApproximated(meta) {
  const el = document.getElementById("approximated-footnote");
  if (!meta || !meta.approximated || !meta.approximated.length) {
    el.textContent = "";
    return;
  }
  el.textContent =
    `${meta.approximated.length} features approximated (defaulted to neutral): ` +
    meta.approximated.join(", ") + ".";
}

async function scan(url) {
  showOnly("loading");
  let resp;
  try {
    resp = await fetch(`${API_BASE}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
  } catch (e) {
    showError(`Network error: ${e.message}`);
    return;
  }
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    showError(data.message || data.error || `HTTP ${resp.status}`);
    return;
  }
  if (data.reachable === false) {
    document.getElementById("notfound-url").textContent = data.url;
    showOnly("notfound");
    return;
  }
  document.getElementById("scanned-url").textContent = data.url;
  renderModelPredictions(data.predictions);
  renderSafeBrowsing(data.safe_browsing);
  renderAgreement(data.predictions, data.safe_browsing);
  renderFeatures(data.features_display);
  renderApproximated(data.features_meta);
  showOnly("results");
}

function showError(msg) {
  document.getElementById("error-message").textContent = msg;
  showOnly("error");
}

function reset() {
  document.getElementById("url-input").value = "";
  showOnly("input");
}

// Wake up Cloud Run as soon as the page loads, so the first scan is faster.
fetch(`${API_BASE}/api/health`).catch(() => {});

document.getElementById("scan-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const url = document.getElementById("url-input").value.trim();
  if (url) scan(url);
});
document.getElementById("reset-button").addEventListener("click", reset);
document.getElementById("notfound-reset-button").addEventListener("click", reset);
document.getElementById("error-reset-button").addEventListener("click", reset);
