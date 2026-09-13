import { useEffect, useState } from "react";
import { getImportance, getJob, getJobs, getMetrics, startJob } from "../api.js";

export default function Train() {
  const [metrics, setMetrics] = useState(null);
  const [importance, setImportance] = useState([]);
  const [jobs, setJobs] = useState([]);
  const [active, setActive] = useState(null);
  const [error, setError] = useState("");

  async function refresh() {
    const [m, i, j] = await Promise.all([getMetrics(), getImportance(), getJobs()]);
    setMetrics(m);
    setImportance(i.features || []);
    setJobs(j.jobs || []);
    const running = (j.jobs || []).find((job) => job.status === "running" || job.status === "queued");
    if (running) {
      const latest = await getJob(running.id);
      setActive(latest);
    }
  }

  useEffect(() => {
    refresh().catch((err) => setError(err.message));
  }, []);

  useEffect(() => {
    if (!active || !["running", "queued"].includes(active.status)) return undefined;
    const timer = window.setInterval(() => {
      getJob(active.id)
        .then((job) => {
          setActive(job);
          if (!["running", "queued"].includes(job.status)) {
            refresh().catch(() => {});
          }
        })
        .catch(() => {});
    }, 2000);
    return () => window.clearInterval(timer);
  }, [active?.id, active?.status]);

  async function run(kind) {
    setError("");
    try {
      const job = await startJob(kind);
      setActive(job);
    } catch (err) {
      setError(err.message);
    }
  }

  const latest = metrics?.metrics?.latest_week || {};
  const hgb = metrics?.metrics?.hgb_margin || {};
  const ats = metrics?.metrics?.ats_from_margin_model?.ats_ge_0 || {};
  const maxImp = Math.max(...importance.map((row) => Math.abs(row.importance || 0)), 0.0001);

  return (
    <section>
      <p className="kicker">Keep the board honest</p>
      <h1>Train on completed games</h1>
      <p className="lede">
        Collect refreshes only the live season on CFBD. After every FBS game in a week is
        final, the board trains a correction on that week&apos;s picks versus the scores. The
        trees are not refit from scratch each week. Train rebuilds the frozen base from all
        completed games. Only one job runs at a time.
      </p>
      {error && <p className="error">{error}</p>}
      <div className="toolbar">
        <button className="btn ghost" type="button" onClick={() => run("collect")}>
          Collect
        </button>
        <button className="btn ghost" type="button" onClick={() => run("weather")}>
          Weather
        </button>
        <button className="btn" type="button" onClick={() => run("train")}>
          Train
        </button>
        <button className="btn ghost" type="button" onClick={() => run("pipeline")}>
          Full pipeline
        </button>
      </div>
      {!metrics?.ready && (
        <div className="empty">
          <h2>No metrics yet</h2>
          <p>Run training after the SQLite database has games. Weekly updates run automatically once finals are in.</p>
        </div>
      )}
      {metrics?.ready && (
        <div className="metrics">
          <div className="metric">
            <b>{latest.winner_accuracy ?? hgb.winner_accuracy ?? "—"}</b>
            <span>Latest week winner acc</span>
          </div>
          <div className="metric">
            <b>{latest.mae_adjusted ?? latest.mae_base ?? hgb.mae ?? "—"}</b>
            <span>Margin MAE</span>
          </div>
          <div className="metric">
            <b>{ats.accuracy ?? "—"}</b>
            <span>ATS (edge ≥ 0)</span>
          </div>
          <div className="metric">
            <b>{metrics.metrics?.train_games ?? "—"}</b>
            <span>Train games</span>
          </div>
        </div>
      )}
      <div className="builder" style={{ marginTop: 28 }}>
        <div className="panel">
          <p className="kicker">Job log</p>
          <pre className="log">{active?.log || "No job running."}</pre>
          {active && (
            <p>
              {active.kind} · {active.status}
              {active.error ? ` · ${active.error}` : ""}
            </p>
          )}
        </div>
        <div className="panel">
          <p className="kicker">What the model listens to</p>
          {importance.slice(0, 12).map((row) => (
            <div className="imp-bar" key={row.feature}>
              <span>{row.feature}</span>
              <div className="bar">
                <span style={{ width: `${(Math.abs(row.importance) / maxImp) * 100}%` }} />
              </div>
              <span>{Number(row.importance).toFixed(2)}</span>
            </div>
          ))}
          {importance.length === 0 && <p>Feature importance appears after a training run.</p>}
        </div>
      </div>
      {jobs.length > 0 && (
        <p className="lede" style={{ marginTop: 24 }}>
          Recent jobs: {jobs.slice(0, 5).map((job) => `${job.kind} ${job.status}`).join(" · ")}
        </p>
      )}
    </section>
  );
}
