import { useState } from "react";
import ASLTranslator from "./ASLTranslator";
import CollectData from "./CollectData";
import Dictionary from "./Dictionary";

export default function App() {
  const [page, setPage] = useState("translator");
  const navigate = (p) => setPage(p);

  if (page === "collect")     return <CollectData  onNavigate={navigate} />;
  if (page === "dictionary")  return <Dictionary   onNavigate={navigate} />;
  return <ASLTranslator onNavigate={navigate} />;
}