export async function api(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (options.body && !headers["Content-Type"]) {
    headers["Content-Type"] = "application/json";
  }
  const response = await fetch(path, { ...options, headers });
  const text = await response.text();
  let data = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = { detail: text };
    }
  }
  if (!response.ok) {
    const detail = data?.detail;
    const message = Array.isArray(detail) ? detail.map((item) => item.msg || item).join("; ") : detail;
    const error = new Error(message || response.statusText);
    error.status = response.status;
    error.body = data;
    throw error;
  }
  return data;
}

export const getHealth = () => api("/api/health");
export const getTeams = () => api("/api/teams");
export const getWeeks = (season) =>
  api(season ? `/api/weeks?season=${season}` : "/api/weeks");
export const getPredictions = (season, week) => {
  const params = new URLSearchParams();
  if (season) params.set("season", season);
  if (week) params.set("week", week);
  const query = params.toString();
  return api(`/api/predictions${query ? `?${query}` : ""}`);
};
export const postMatchup = (body) =>
  api("/api/matchups", { method: "POST", body: JSON.stringify(body) });
export const getMetrics = () => api("/api/metrics");
export const getImportance = () => api("/api/metrics/importance");
export const getJobs = () => api("/api/jobs");
export const getJob = (id) => api(`/api/jobs/${id}`);
export const startJob = (kind, holdout_season) =>
  api(`/api/jobs/${kind}`, {
    method: "POST",
    body: JSON.stringify(holdout_season ? { holdout_season } : {}),
  });
