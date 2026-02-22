import { useState } from "react";
import Home from "./Home";
import ASLTranslator from "./ASLTranslator";
import CollectData from "./CollectData";

export default function App() {
  const [page, setPage] = useState("home");

  const navigate = (p) => setPage(p);

  if (page === "home") return <Home onNavigate={navigate} />;
  if (page === "collect") return <CollectData onNavigate={navigate} />;
  return <ASLTranslator onNavigate={navigate} />;
}