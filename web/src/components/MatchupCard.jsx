import {
  cardInkMode,
  confidenceLabel,
  contrastShadow,
  contrastText,
  formatHomeSpread,
  formatProb,
  formatScore,
  isLightColor,
  TeamLogo,
} from "./brand.jsx";

function verdict(hit) {
  if (hit === true) return { label: "hit", className: "hit" };
  if (hit === false) return { label: "miss", className: "miss" };
  return { label: "push", className: "push" };
}

export default function MatchupCard({ game, onOpen }) {
  const away = game.away_team || {};
  const home = game.home_team || {};
  const awayColor = away.primary_color || "#1a4a36";
  const homeColor = home.primary_color || "#4a1a1a";
  const awayInk = contrastText(awayColor);
  const homeInk = contrastText(homeColor);
  const inkMode = cardInkMode(awayColor, homeColor);
  const isFinal = Boolean(game.completed) && game.home_points != null && game.away_points != null;
  const awayScore = formatScore(game.projected_away_score);
  const homeScore = formatScore(game.projected_home_score);
  const homeWp = Number(game.home_win_prob || 0);
  const winner = verdict(game.winner_hit);
  const ats = verdict(game.ats_hit);
  const conf = String(game.conf || "").toLowerCase();
  const confText = confidenceLabel(conf, game.pick_prob);

  return (
    <article
      className="card"
      data-ink={inkMode}
      onClick={() => onOpen?.(game)}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onOpen?.(game);
        }
      }}
      role="button"
      tabIndex={0}
    >
      <div className="split" aria-hidden="true">
        <div
          className="away"
          data-light={isLightColor(awayColor) ? "true" : "false"}
          style={{ background: awayColor }}
        />
        <div
          className="home"
          data-light={isLightColor(homeColor) ? "true" : "false"}
          style={{ background: homeColor }}
        />
      </div>
      <div className="card-body">
        <div className="card-meta">
          <span>{game.kick_label || game.status || "Matchup"}</span>
          {confText ? (
            <span className={`conf-stamp conf-${conf || "lean"}`}>{confText}</span>
          ) : (
            <span>{game.venue || "TBD"}</span>
          )}
        </div>
        <div className="teams">
          <div
            className="side away"
            style={{ color: awayInk, "--ink-shadow": contrastShadow(awayColor) }}
          >
            <TeamLogo team={away} />
            <div className="school">{away.name || away.school}</div>
          </div>
          <div className="scoreboard">
            {awayScore === "—" && homeScore === "—" ? (
              <>
                <div className="nums">
                  {game.pred_margin == null ? "—" : (Number(game.pred_margin) >= 0 ? "+" : "") + Number(game.pred_margin).toFixed(1)}
                </div>
                <div className="at">model margin</div>
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
          <div
            className="side home"
            style={{ color: homeInk, "--ink-shadow": contrastShadow(homeColor) }}
          >
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
        {isFinal && (
          <div className="card-actual">
            Actual {formatScore(game.away_points)}–{formatScore(game.home_points)}
            <span className={`verdict ${winner.className}`}> winner {winner.label}</span>
            {game.spread != null ? <span className={`verdict ${ats.className}`}> ATS {ats.label}</span> : null}
          </div>
        )}
        <div className="card-foot">
          <span>Pick {game.pick || "—"}</span>
          <span>
            {game.spread != null ? game.spread_label || formatHomeSpread(home, game.spread) : "no market"}
            {game.cover_side ? ` · cover ${game.cover_side}` : ""}
          </span>
        </div>
        {game.venue ? <div className="card-venue">{game.venue}</div> : null}
      </div>
    </article>
  );
}
