import { NavLink } from "react-router-dom";

export default function Shell({ children }) {
  return (
    <div className="shell">
      <header className="mast">
        <div className="wordmark">
          <small>College football</small>
          <strong>Matchup Desk</strong>
        </div>
        <nav className="nav">
          <NavLink to="/" end>
            Board
          </NavLink>
          <NavLink to="/matchup">Matchup</NavLink>
          <NavLink to="/train">Train</NavLink>
        </nav>
      </header>
      <main className="page">{children}</main>
      <footer className="foot">Predict the game. Leave the rest on the sideline.</footer>
    </div>
  );
}
