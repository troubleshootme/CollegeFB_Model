import { Navigate, Route, Routes } from "react-router-dom";
import Shell from "./components/Shell.jsx";
import Board from "./pages/Board.jsx";
import Matchup from "./pages/Matchup.jsx";
import Train from "./pages/Train.jsx";

export default function App() {
  return (
    <Shell>
      <Routes>
        <Route path="/" element={<Board />} />
        <Route path="/matchup" element={<Matchup />} />
        <Route path="/train" element={<Train />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Shell>
  );
}
