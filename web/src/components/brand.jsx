export function TeamLogo({ team, className = "" }) {
  const name = team?.name || team?.school || "?";
  const src = team?.logo_url;
  if (src) {
    return <img className={`logo ${className}`} src={src} alt="" />;
  }
  return <div className={`logo fallback ${className}`}>{name.slice(0, 2).toUpperCase()}</div>;
}

export function formatProb(value) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return `${Math.round(Number(value) * 100)}%`;
}

export function formatScore(value) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return Math.round(Number(value));
}

export function contrastText(hex) {
  if (!hex) return "#F4E4C1";
  const raw = hex.replace("#", "");
  if (raw.length !== 6) return "#F4E4C1";
  const r = parseInt(raw.slice(0, 2), 16);
  const g = parseInt(raw.slice(2, 4), 16);
  const b = parseInt(raw.slice(4, 6), 16);
  const luma = (r * 299 + g * 587 + b * 114) / 1000;
  return luma > 150 ? "#12181A" : "#F4E4C1";
}
