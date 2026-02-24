import { useState } from "react";
import Home from "./Home";
import ASLTranslator from "./ASLTranslator";
import CollectData from "./CollectData";
import Dictionary from "./Dictionary";
import Practice from "./Practice";

export default function App() {
  const [page, setPage] = useState("home");
  const navigate = (p) => setPage(p);

  if (page === "home")        return <Home         onNavigate={navigate} />;
  if (page === "practice")    return <Practice     onNavigate={navigate} />;
  if (page === "collect")     return <CollectData onNavigate={navigate} />;
  if (page === "dictionary")  return <Dictionary   onNavigate={navigate} />;
  return <ASLTranslator onNavigate={navigate} />;
}