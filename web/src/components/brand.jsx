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
  let points = Math.round(Number(value));
  if (points < 0) points = 0;
  if (points === 1) points = 0;
  return points;
}

export function formatHomeSpread(team, spread) {
  if (spread == null || Number.isNaN(Number(spread))) return "no market";
  const abbr = String(team?.abbreviation || team?.name || team?.school || "HOME").trim().toUpperCase();
  const line = Number(spread);
  if (Math.abs(line) < 1e-9) return `${abbr} PK`;
  const pretty = String(Number(line.toFixed(2)));
  const signed = line > 0 ? `+${pretty}` : pretty;
  return `${abbr} ${signed}`;
}

export function parseHexColor(hex) {
  if (!hex || typeof hex !== "string") return null;
  let raw = hex.trim();
  if (raw.startsWith("rgb")) {
    const parts = raw.match(/[\d.]+/g);
    if (!parts || parts.length < 3) return null;
    return {
      r: Math.max(0, Math.min(255, Number(parts[0]))),
      g: Math.max(0, Math.min(255, Number(parts[1]))),
      b: Math.max(0, Math.min(255, Number(parts[2]))),
    };
  }
  raw = raw.replace(/^#/, "");
  if (raw.length === 3) raw = raw.split("").map((ch) => ch + ch).join("");
  if (raw.length !== 6 || /[^0-9a-f]/i.test(raw)) return null;
  return {
    r: parseInt(raw.slice(0, 2), 16),
    g: parseInt(raw.slice(2, 4), 16),
    b: parseInt(raw.slice(4, 6), 16),
  };
}

export function relativeLuma(hex) {
  const rgb = typeof hex === "object" && hex ? hex : parseHexColor(hex);
  if (!rgb) return 0;
  const lin = (channel) => {
    const value = channel / 255;
    return value <= 0.03928 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * lin(rgb.r) + 0.7152 * lin(rgb.g) + 0.0722 * lin(rgb.b);
}

function contrastRatio(lumaA, lumaB) {
  const [hi, lo] = lumaA > lumaB ? [lumaA, lumaB] : [lumaB, lumaA];
  return (hi + 0.05) / (lo + 0.05);
}

const INK_DARK = "#12181A";
const INK_LIGHT = "#F4E4C1";
const INK_DARK_LUMA = relativeLuma(INK_DARK);
const INK_LIGHT_LUMA = relativeLuma(INK_LIGHT);

export function contrastText(hex) {
  const luma = relativeLuma(hex);
  const darkInk = contrastRatio(luma, INK_DARK_LUMA);
  const lightInk = contrastRatio(luma, INK_LIGHT_LUMA);
  return darkInk >= lightInk ? INK_DARK : INK_LIGHT;
}

export function isLightColor(hex) {
  return contrastText(hex) === INK_DARK;
}

export function contrastShadow(hex) {
  return isLightColor(hex)
    ? "0 1px 0 rgba(255, 255, 255, 0.55), 0 2px 10px rgba(255, 255, 255, 0.28)"
    : "0 2px 10px rgba(0, 0, 0, 0.62)";
}

export function cardInkMode(awayHex, homeHex) {
  const awayLight = isLightColor(awayHex);
  const homeLight = isLightColor(homeHex);
  if (awayLight && homeLight) return "light";
  if (!awayLight && !homeLight) return "dark";
  return "split";
}

export function confidenceLabel(conf, pickProb) {
  const grade = String(conf || "").toLowerCase();
  if (!grade) return pickProb == null ? "" : formatProb(pickProb);
  if (pickProb == null || Number.isNaN(Number(pickProb))) return grade;
  const p = Number(pickProb);
  const shown = Math.max(p, 1 - p);
  return `${grade} ${formatProb(shown)}`;
}
