import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { getReplay } from "../api.js";
import { filterMatchups } from "../boardFilters.js";
import MatchupCard from "../components/MatchupCard.jsx";
import MatchupFilters from "../components/MatchupFilters.jsx";

function formatCutoff(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

export default function Replay() {
  const navigate = useNavigate();
  const [week, setWeek] = useState("");
  const [weeks, setWeeks] = useState([]);
  const [meta, setMeta] = useState(null);
  const [games, setGames] = useState([]);
  const [minConf, setMinConf] = useState("");
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError("");
    getReplay(week || undefined)
      .then((data) => {
        if (cancelled) return;
        setMeta(data);
        setWeeks(data.weeks || []);
        setGames(data.games || []);
        if (!week && data.week != null) setWeek(String(data.week));
      })
      .catch((err) => {
        if (cancelled) return;
        setGames([]);
        setError(err.message || "Could not load last season.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [week]);

  const weekOptions = meta?.week_options?.length
    ? meta.week_options
    : (weeks || []).map((value) => ({ week: value, label: `Week ${value}` }));
  const summary = meta?.summary || {};
  const winnerN = summary.winner_n || 0;
  const atsN = summary.ats_n || 0;
  const visible = useMemo(() => filterMatchups(games, { minConf, status }), [games, minConf, status]);

  return (
    <section>
      <p className="kicker">Last completed season</p>
      <h1>{meta?.season ? `${meta.season} late slate` : "Last season"}</h1>
      <p className="lede">
        Model scores first, then the actual final as the grade. Late {meta?.season || "2025"} only,
        from the saved model and SQLite — no CFBD refresh, 2026 board untouched.
      </p>
      <div className="toolbar">
        <label className="field">
          Week
          <select value={week} onChange={(event) => setWeek(event.target.value)}>
            {weekOptions.map((option) => (
              <option key={option.week} value={option.week}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
        <MatchupFilters minConf={minConf} onMinConf={setMinConf} status={status} onStatus={setStatus} />
        {winnerN > 0 && (
          <p className="tally">
            Winner {summary.winner_hits}-{winnerN - summary.winner_hits}
            {atsN > 0 ? ` · ATS ${summary.ats_hits}-${atsN - summary.ats_hits}` : ""}
            {meta?.cutoff ? ` · from ${formatCutoff(meta.cutoff)}` : ""}
          </p>
        )}
      </div>
      {error && (
        <div className="notice error">
          {error}{" "}
          {String(error).toLowerCase().includes("train") && <Link to="/train">Open the train desk</Link>}
        </div>
      )}
      {loading && <p>Scoring the stored slate…</p>}
      {!loading && !error && games.length === 0 && (
        <div className="empty">
          <h2>No held-out games this week</h2>
          <p>Pick another week in the late-season window.</p>
        </div>
      )}
      {!loading && !error && games.length > 0 && visible.length === 0 && (
        <div className="empty">
          <h2>Nothing at this filter</h2>
          <p>Drop the confidence floor or show all games to bring the slate back.</p>
        </div>
      )}
      <div className="grid">
        {visible.map((game) => (
          <MatchupCard
            key={game.id || `${game.away_team?.name}-${game.home_team?.name}`}
            game={game}
            onOpen={() => {
              const params = new URLSearchParams({
                home: game.home_team?.name || "",
                away: game.away_team?.name || "",
              });
              if (game.id) params.set("game_id", String(game.id));
              navigate(`/matchup?${params.toString()}`);
            }}
          />
        ))}
      </div>
    </section>
  );
}
