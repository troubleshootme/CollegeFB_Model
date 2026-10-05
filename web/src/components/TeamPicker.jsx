import { useMemo, useState } from "react";
import { TeamLogo } from "./brand.jsx";

export default function TeamPicker({ label, teams, value, onChange }) {
  const [query, setQuery] = useState(value || "");
  const [open, setOpen] = useState(false);
  const selected = teams.find((team) => team.name === value);

  const matches = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const ranked = teams
      .map((team) => {
        const name = (team.name || "").toLowerCase();
        const abbr = (team.abbreviation || "").toLowerCase();
        let rank = 100;
        if (!needle) rank = 10;
        else if (name === needle || abbr === needle) rank = 0;
        else if (name.startsWith(needle)) rank = 1;
        else if (abbr.startsWith(needle)) rank = 2;
        else if (name.includes(` ${needle}`) || name.includes(needle)) rank = 3;
        else if ([name, abbr, team.conference || ""].join(" ").toLowerCase().includes(needle)) rank = 4;
        return { team, rank };
      })
      .filter((row) => row.rank < 100)
      .sort((a, b) => a.rank - b.rank || a.team.name.localeCompare(b.team.name));
    return ranked.slice(0, 12).map((row) => row.team);
  }, [query, teams]);

  return (
    <label className="field" style={{ position: "relative" }}>
      {label}
      <input
        value={open ? query : selected?.name || query}
        onChange={(event) => {
          setQuery(event.target.value);
          setOpen(true);
        }}
        onFocus={() => {
          setQuery(selected?.name || query);
          setOpen(true);
        }}
        onBlur={() => {
          window.setTimeout(() => setOpen(false), 150);
        }}
        placeholder="Find a team"
        autoComplete="off"
      />
      {open && (
        <div className="picker-list">
          {matches.map((team) => (
            <button
              type="button"
              className="picker-row"
              key={team.id || team.name}
              onMouseDown={(event) => event.preventDefault()}
              onClick={() => {
                onChange(team.name);
                setQuery(team.name);
                setOpen(false);
              }}
            >
              <TeamLogo team={team} />
              <span>
                {team.name}
                {team.conference ? ` · ${team.conference}` : ""}
              </span>
            </button>
          ))}
          {matches.length === 0 && <div className="picker-row">No FBS team matches that name.</div>}
        </div>
      )}
    </label>
  );
}
