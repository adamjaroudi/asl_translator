import { useState } from "react";
import ASLTranslator from "./ASLTranslator";
import CollectData from "./CollectData";

export default function App() {
  const [page, setPage] = useState("translator");

  const navigate = (p) => setPage(p);

  if (page === "collect") return <CollectData onNavigate={navigate} />;
  return <ASLTranslator onNavigate={navigate} />;
}