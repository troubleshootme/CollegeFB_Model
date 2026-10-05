export const CONF_RANK = { lean: 0, likely: 1, strong: 2, lock: 3 };

export const CONF_OPTIONS = [
  { value: "", label: "All grades" },
  { value: "lean", label: "Lean+" },
  { value: "likely", label: "Likely+" },
  { value: "strong", label: "Strong+" },
  { value: "lock", label: "Locks only" },
];

export const STATUS_OPTIONS = [
  { value: "", label: "All games" },
  { value: "upcoming", label: "Upcoming" },
  { value: "final", label: "Finals" },
];

export function isFinal(game) {
  return Boolean(game?.completed) && game?.home_points != null && game?.away_points != null;
}

export function confRank(game) {
  const key = String(game?.conf || "lean").toLowerCase();
  return CONF_RANK[key] ?? 0;
}

export function filterMatchups(games, { minConf = "", status = "" } = {}) {
  const floor = minConf ? CONF_RANK[minConf] ?? 0 : null;
  return (games || []).filter((game) => {
    if (floor != null && confRank(game) < floor) return false;
    if (status === "upcoming" && isFinal(game)) return false;
    if (status === "final" && !isFinal(game)) return false;
    return true;
  });
}
