import { useState, useEffect } from "react";

const C = {
  bg:      "#f5f5f0",
  surface: "#ffffff",
  border:  "#d4d4d0",
  text:    "#1a1a1a",
  textMid: "#555550",
  textDim: "#888880",
};

export default function Home({ onNavigate }) {
  useEffect(() => {
    const el = document.createElement("style");
    el.textContent = `
      @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
      *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
      body { background: ${C.bg}; font-family: 'IBM Plex Sans', system-ui, sans-serif; }
      button { font-family: inherit; cursor: pointer; }
      button:active { transform: scale(0.98); }
    `;
    document.head.appendChild(el);
    return () => document.head.removeChild(el);
  }, []);

  return (
    <div style={s.root}>
      <header style={s.header}>
        <Logo />
        <nav style={s.nav}>
          <NavBtn onClick={() => onNavigate("translator")}>Translator</NavBtn>
          <NavBtn onClick={() => onNavigate("practice")}>Practice</NavBtn>
          <NavBtn onClick={() => onNavigate("dictionary")}>Dictionary</NavBtn>
          <NavBtn primary onClick={() => onNavigate("collect")}>Collect Data</NavBtn>
        </nav>
      </header>

      <main style={s.main}>
        <section style={s.hero}>
          <div style={s.badge}>Real-time ASL Recognition</div>
          <h1 style={s.title}>Sign language,<br />understood instantly.</h1>
          <p style={s.sub}>
            NeuroSign uses your webcam and on-device machine learning to translate
            American Sign Language into text in real time — no cloud, no delay.
          </p>
          <div style={s.heroRow}>
            <PrimaryBtn onClick={() => onNavigate("translator")}>Start Translating</PrimaryBtn>
            <OutlineBtn onClick={() => onNavigate("practice")}>Practice A–Z</OutlineBtn>
          </div>
        </section>

        <section style={s.cards}>
          <Card n="01" title="Live translation" cta="Open translator →" onClick={() => onNavigate("translator")}
            desc="Static letters A–Z and word signs recognised frame-by-frame. Your sentence builds as you sign." />
          <Card n="02" title="Motion signs" cta="Try motion mode →" onClick={() => onNavigate("translator")}
            desc="J, Z, PLEASE, THANK YOU and more — signs requiring movement captured over 30-frame windows." />
          <Card n="03" title="Train your model" cta="Collect data →" onClick={() => onNavigate("collect")}
            desc="Record your own signing data and retrain to improve accuracy for your hands and environment." />
          <Card n="04" title="Practice & quiz" cta="Start practicing →" onClick={() => onNavigate("practice")}
            desc="Flashcards, type-it quizzes, and live sign-it mode using your camera to learn the alphabet." />
        </section>
      </main>

      <footer style={s.footer}>
        <Logo small />
        <span style={s.footerNote}>
          Requires <code style={s.code}>python server.py</code> running locally and camera access allowed.
        </span>
      </footer>
    </div>
  );
}

function Logo({ small }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
      <div style={{
        width: small ? 22 : 26, height: small ? 22 : 26, background: C.text, color: "#fff",
        display: "flex", alignItems: "center", justifyContent: "center",
        fontSize: small ? 11 : 13, fontWeight: 700, letterSpacing: "-0.02em", flexShrink: 0,
      }}>N</div>
      <span style={{ fontSize: small ? 13 : 15, fontWeight: 700, letterSpacing: "-0.02em", color: C.text }}>
        NeuroSign
      </span>
    </div>
  );
}

function NavBtn({ children, onClick, primary }) {
  const [h, setH] = useState(false);
  return (
    <button onClick={onClick} onMouseEnter={() => setH(true)} onMouseLeave={() => setH(false)}
      style={{
        padding: "6px 13px",
        background: primary ? (h ? "#333" : C.text) : "none",
        border: "none",
        fontSize: 13, fontWeight: 500,
        color: primary ? "#fff" : (h ? C.text : C.textMid),
        transition: "color 0.1s, background 0.1s",
      }}>
      {children}
    </button>
  );
}

function PrimaryBtn({ children, onClick }) {
  const [h, setH] = useState(false);
  return (
    <button onClick={onClick} onMouseEnter={() => setH(true)} onMouseLeave={() => setH(false)}
      style={{
        padding: "13px 30px", background: h ? "#333" : C.text, color: "#fff",
        border: "none", fontSize: 15, fontWeight: 600, transition: "background 0.1s",
      }}>
      {children}
    </button>
  );
}

function OutlineBtn({ children, onClick }) {
  const [h, setH] = useState(false);
  return (
    <button onClick={onClick} onMouseEnter={() => setH(true)} onMouseLeave={() => setH(false)}
      style={{
        padding: "13px 30px", background: h ? "#f0f0eb" : "none",
        border: `1px solid ${C.border}`,
        fontSize: 15, fontWeight: 500, color: C.text, transition: "background 0.1s",
      }}>
      {children}
    </button>
  );
}

function Card({ n, title, desc, cta, onClick }) {
  const [h, setH] = useState(false);
  return (
    <button onClick={onClick} onMouseEnter={() => setH(true)} onMouseLeave={() => setH(false)}
      style={{
        background: h ? "#fafaf8" : C.surface, border: "none",
        padding: "28px 26px", textAlign: "left", cursor: "pointer",
        display: "flex", flexDirection: "column", gap: 7, transition: "background 0.1s",
      }}>
      <span style={{ fontSize: 11, fontWeight: 600, letterSpacing: "0.1em", color: C.textDim, fontFamily: "'IBM Plex Mono', monospace" }}>{n}</span>
      <span style={{ fontSize: 15, fontWeight: 600, color: C.text, marginTop: 2 }}>{title}</span>
      <span style={{ fontSize: 13, color: C.textMid, lineHeight: 1.55, flex: 1 }}>{desc}</span>
      <span style={{ fontSize: 12, fontWeight: 600, color: C.text, marginTop: 8 }}>{cta}</span>
    </button>
  );
}

const s = {
  root: { minHeight: "100vh", background: C.bg, fontFamily: "'IBM Plex Sans', system-ui, sans-serif", color: C.text, display: "flex", flexDirection: "column" },
  header: { background: C.surface, borderBottom: `1px solid ${C.border}`, padding: "0 32px", height: 54, display: "flex", alignItems: "center", justifyContent: "space-between", position: "sticky", top: 0, zIndex: 10 },
  nav: { display: "flex", alignItems: "center", gap: 2 },
  main: { flex: 1, maxWidth: 1080, width: "100%", margin: "0 auto", padding: "72px 32px 80px" },
  hero: { textAlign: "center", marginBottom: 80 },
  badge: { display: "inline-block", padding: "4px 12px", border: `1px solid ${C.border}`, fontSize: 11, fontWeight: 600, letterSpacing: "0.1em", textTransform: "uppercase", color: C.textDim, marginBottom: 28 },
  title: { fontSize: "clamp(34px, 5vw, 54px)", fontWeight: 700, letterSpacing: "-0.03em", lineHeight: 1.1, color: C.text, marginBottom: 22 },
  sub: { fontSize: 17, color: C.textMid, maxWidth: 500, margin: "0 auto 36px", lineHeight: 1.6 },
  heroRow: { display: "flex", gap: 12, justifyContent: "center", flexWrap: "wrap" },
  cards: { display: "grid", gridTemplateColumns: "repeat(4, 1fr)", border: `1px solid ${C.border}`, background: C.border, gap: 1 },
  footer: { borderTop: `1px solid ${C.border}`, background: C.surface, padding: "18px 32px", display: "flex", alignItems: "center", justifyContent: "space-between", gap: 16, flexWrap: "wrap" },
  footerNote: { fontSize: 12, color: C.textDim },
  code: { fontFamily: "'IBM Plex Mono', monospace", background: "#f0f0eb", padding: "1px 5px", fontSize: 11 },
};