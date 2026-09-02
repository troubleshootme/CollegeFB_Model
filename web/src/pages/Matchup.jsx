import { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { getTeams, postMatchup } from "../api.js";
import MatchupCard from "../components/MatchupCard.jsx";
import TeamPicker from "../components/TeamPicker.jsx";

export default function Matchup() {
  const [params] = useSearchParams();
  const [teams, setTeams] = useState([]);
  const [home, setHome] = useState(params.get("home") || "");
  const [away, setAway] = useState(params.get("away") || "");
  const [neutral, setNeutral] = useState(false);
  const [spread, setSpread] = useState("");
  const [overUnder, setOverUnder] = useState("");
  const [kickoff, setKickoff] = useState("");
  const [temp, setTemp] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    getTeams()
      .then((data) => setTeams(data.teams || []))
      .catch((err) => setError(err.message));
  }, []);

  async function predict(event) {
    event.preventDefault();
    setBusy(true);
    setError("");
    const body = {
      home_team: home,
      away_team: away,
      neutral_site: neutral,
    };
    if (spread !== "") body.spread = Number(spread);
    if (overUnder !== "") body.over_under = Number(overUnder);
    if (kickoff) body.kickoff = new Date(kickoff).toISOString();
    if (temp !== "") body.wx_temp_max = Number(temp);
    if (params.get("game_id")) body.game_id = Number(params.get("game_id"));
    try {
      const payload = await postMatchup(body);
      setResult(payload);
    } catch (err) {
      setResult(null);
      setError(err.message || "Could not score that matchup.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section>
      <p className="kicker">Hypothetical or scheduled</p>
      <h1>Put two teams on the field</h1>
      <p className="lede">
        Away on the left, home on the right. Optional lines and weather override the snapshot the model already has.
      </p>
      <div className="builder">
        <form className="panel" onSubmit={predict}>
          <TeamPicker label="Away" teams={teams} value={away} onChange={setAway} />
          <div style={{ height: 12 }} />
          <TeamPicker label="Home" teams={teams} value={home} onChange={setHome} />
          <div className="toolbar" style={{ marginTop: 16 }}>
            <label className="field">
              Spread (home)
              <input value={spread} onChange={(event) => setSpread(event.target.value)} placeholder="-3.5" />
            </label>
            <label className="field">
              Over / under
              <input value={overUnder} onChange={(event) => setOverUnder(event.target.value)} placeholder="54.5" />
            </label>
            <label className="field">
              Kickoff
              <input type="datetime-local" value={kickoff} onChange={(event) => setKickoff(event.target.value)} />
            </label>
            <label className="field">
              High temp °C
              <input value={temp} onChange={(event) => setTemp(event.target.value)} placeholder="31" />
            </label>
          </div>
          <label className="field" style={{ marginTop: 8, flexDirection: "row", alignItems: "center", gap: 10 }}>
            <input type="checkbox" checked={neutral} onChange={(event) => setNeutral(event.target.checked)} />
            Neutral site
          </label>
          <div style={{ marginTop: 18 }}>
            <button className="btn" type="submit" disabled={busy || !home || !away || home === away}>
              {busy ? "Scoring…" : "Predict this matchup"}
            </button>
          </div>
        </form>
        <div>
          {error && (
            <div className="empty">
              <h2>Could not score it</h2>
              <p className="error">{error}</p>
              {String(error).toLowerCase().includes("train") && (
                <Link className="btn" to="/train">
                  Retrain on latest games
                </Link>
              )}
            </div>
          )}
          {result && <MatchupCard game={result} />}
          {result && (
            <div className="panel" style={{ marginTop: 16 }}>
              <div className="metrics">
                <div className="metric">
                  <b>{result.pred_margin != null ? Number(result.pred_margin).toFixed(1) : "—"}</b>
                  <span>Margin (home)</span>
                </div>
                <div className="metric">
                  <b>{result.conf || "—"}</b>
                  <span>Confidence</span>
                </div>
                <div className="metric">
                  <b>{result.ats_edge != null ? Number(result.ats_edge).toFixed(1) : "—"}</b>
                  <span>ATS edge</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
