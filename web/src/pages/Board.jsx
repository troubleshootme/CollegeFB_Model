import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { getHealth, getPredictions, getWeeks } from "../api.js";
import MatchupCard from "../components/MatchupCard.jsx";

export default function Board() {
  const navigate = useNavigate();
  const [health, setHealth] = useState(null);
  const [season, setSeason] = useState("");
  const [week, setWeek] = useState("");
  const [weeks, setWeeks] = useState([]);
  const [games, setGames] = useState([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() => setHealth({ models_ready: false }));
  }, []);

  useEffect(() => {
    getWeeks()
      .then((data) => {
        setWeeks(data.weeks || []);
        if (!season) setSeason(String(data.season || ""));
      })
      .catch(() => {});
  }, []);

  const seasonWeeks = useMemo(
    () => weeks.filter((row) => String(row.season) === String(season)).map((row) => row.week),
    [weeks, season]
  );

  useEffect(() => {
    if (season && !week && seasonWeeks.length) {
      const upcoming = weeks.find((row) => String(row.season) === String(season) && row.upcoming > 0);
      setWeek(String(upcoming?.week || seasonWeeks[0]));
    }
  }, [season, week, seasonWeeks, weeks]);

  useEffect(() => {
    if (!season || !week) return;
    let cancelled = false;
    setLoading(true);
    setError("");
    getPredictions(season, week)
      .then((data) => {
        if (cancelled) return;
        setGames(data.games || []);
        setSeason(String(data.season));
        setWeek(String(data.week));
      })
      .catch((err) => {
        if (cancelled) return;
        setGames([]);
        setError(err.message || "Could not load the board.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [season, week]);

  const seasons = [...new Set(weeks.map((row) => row.season))].sort((a, b) => b - a);

  return (
    <section>
      <p className="kicker">This week on the field</p>
      <h1>Matchups</h1>
      <p className="lede">
        Every card is a prediction: projected score, win chance, and who covers if the market posted a number.
      </p>
      <div className="toolbar">
        <label className="field">
          Season
          <select value={season} onChange={(event) => { setSeason(event.target.value); setWeek(""); }}>
            {seasons.map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          Week
          <select value={week} onChange={(event) => setWeek(event.target.value)}>
            {seasonWeeks.map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </label>
      </div>
      {health && !health.models_ready && (
        <div className="empty">
          <h2>Train the model first</h2>
          <p>There is no saved model yet. Collect is optional if the database is already loaded.</p>
          <Link className="btn" to="/train">
            Retrain on latest games
          </Link>
        </div>
      )}
      {error && health?.models_ready !== false && (
        <div className="notice error">
          {error}{" "}
          {String(error).toLowerCase().includes("train") && (
            <Link to="/train">Open the train desk</Link>
          )}
        </div>
      )}
      {loading && <p>Loading the slate…</p>}
      {!loading && !error && games.length === 0 && health?.models_ready && (
        <div className="empty">
          <h2>No FBS matchups this week</h2>
          <p>Pick another week, or build a hypothetical on the matchup desk.</p>
        </div>
      )}
      <div className="grid">
        {games.map((game) => (
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
