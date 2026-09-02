import { contrastText, formatProb, formatScore, TeamLogo } from "./brand.jsx";

export default function MatchupCard({ game, onOpen }) {
  const away = game.away_team || {};
  const home = game.home_team || {};
  const awayColor = away.primary_color || "#1a4a36";
  const homeColor = home.primary_color || "#4a1a1a";
  const awayScore = formatScore(game.projected_away_score);
  const homeScore = formatScore(game.projected_home_score);
  const homeWp = Number(game.home_win_prob || 0);

  return (
    <article
      className="card"
      onClick={() => onOpen?.(game)}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onOpen?.(game);
        }
      }}
      role="button"
      tabIndex={0}
      style={{ color: contrastText(homeColor) }}
    >
      <div className="split" aria-hidden="true">
        <div className="away" style={{ background: awayColor }} />
        <div className="hash-stripe" />
        <div className="home" style={{ background: homeColor }} />
      </div>
      <div className="card-body">
        <div className="card-meta">
          <span>{game.kick_label || game.status || "Matchup"}</span>
          <span>{game.venue || "TBD"}</span>
        </div>
        <div className="teams">
          <div className="side away">
            <TeamLogo team={away} />
            <div className="school">{away.name || away.school}</div>
          </div>
          <div className="scoreboard">
            {awayScore === "—" && homeScore === "—" ? (
              <>
                <div className="nums">
                  {game.pred_margin == null ? "—" : (Number(game.pred_margin) >= 0 ? "+" : "") + Number(game.pred_margin).toFixed(1)}
                </div>
                <div className="at">home margin</div>
              </>
            ) : (
              <>
                <div className="nums">
                  {awayScore}
                  <span className="at"> — </span>
                  {homeScore}
                </div>
                <div className="at">predicted</div>
              </>
            )}
          </div>
          <div className="side home">
            <TeamLogo team={home} />
            <div className="school">{home.name || home.school}</div>
          </div>
        </div>
        <div className="winbar">
          <span>{formatProb(game.away_win_prob)}</span>
          <div className="track">
            <div className="fill" style={{ width: `${Math.round((1 - homeWp) * 100)}%`, background: awayColor }} />
            <div className="fill" style={{ width: `${Math.round(homeWp * 100)}%`, background: homeColor }} />
          </div>
          <span>{formatProb(game.home_win_prob)}</span>
        </div>
        <div className="card-foot">
          <span>Pick {game.pick || "—"}</span>
          <span>
            {game.spread != null ? `spread ${game.spread}` : "no market"}
            {game.cover_side ? ` · cover ${game.cover_side}` : ""}
          </span>
        </div>
      </div>
    </article>
  );
}
