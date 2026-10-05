import { CONF_OPTIONS, STATUS_OPTIONS } from "../boardFilters.js";

export default function MatchupFilters({ minConf, onMinConf, status, onStatus }) {
  return (
    <>
      <label className="field">
        Confidence
        <select value={minConf} onChange={(event) => onMinConf(event.target.value)}>
          {CONF_OPTIONS.map((option) => (
            <option key={option.value || "all"} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </label>
      <label className="field">
        Games
        <select value={status} onChange={(event) => onStatus(event.target.value)}>
          {STATUS_OPTIONS.map((option) => (
            <option key={option.value || "all"} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </label>
    </>
  );
}
