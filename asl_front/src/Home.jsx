import { useEffect } from "react";

const C = {
  bg:       "#f5f5f0",
  surface:  "#ffffff",
  border:   "#d4d4d0",
  text:     "#1a1a1a",
  textMid:  "#555550",
  textDim:  "#888880",
  primary:  "#1a1a1a",
  hover:    "#333",
};

const styles = {
  root: {
    minHeight: "100vh",
    background: C.bg,
    fontFamily: "'IBM Plex Sans', system-ui, sans-serif",
    color: C.text,
    display: "flex",
    flexDirection: "column",
  },
  header: {
    background: C.surface,
    borderBottom: `1px solid ${C.border}`,
    padding: "0 28px",
    height: 52,
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
  },
  headerLeft: { display: "flex", alignItems: "center", gap: 10 },
  headerLogo: { fontSize: 22 },
  headerTitle: { fontSize: 18, fontWeight: 600, letterSpacing: "-0.01em", color: C.text },
  main: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    padding: 40,
    textAlign: "center",
  },
  heroTitle: {
    fontSize: "clamp(28px, 5vw, 42px)",
    fontWeight: 600,
    letterSpacing: "-0.02em",
    color: C.text,
    marginBottom: 12,
    lineHeight: 1.2,
  },
  heroSub: {
    fontSize: 18,
    color: C.textMid,
    maxWidth: 480,
    marginBottom: 40,
    lineHeight: 1.5,
  },
  buttonGroup: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: 16,
  },
  btnCamera: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: 12,
    padding: "18px 42px",
    fontSize: 18,
    fontWeight: 600,
    color: C.surface,
    background: C.primary,
    border: "none",
    borderRadius: 8,
    cursor: "pointer",
    boxShadow: "0 2px 8px rgba(0,0,0,0.12)",
  },
  btnSecondary: {
    padding: "10px 24px",
    fontSize: 14,
    fontWeight: 500,
    color: C.textMid,
    background: "none",
    border: `1px solid ${C.border}`,
    borderRadius: 6,
    cursor: "pointer",
  },
  footer: {
    padding: "20px 28px",
    borderTop: `1px solid ${C.border}`,
    background: C.surface,
    fontSize: 13,
    color: C.textDim,
    textAlign: "center",
  },
};

export default function Home({ onNavigate }) {
  useEffect(() => {
    const style = document.createElement("style");
    style.textContent = `
      @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
      body { background: ${C.bg}; }
      button:hover { opacity: 0.9; }
      button:active { transform: scale(0.98); }
      button:focus-visible { outline: 2px solid ${C.primary}; outline-offset: 2px; }
    `;
    document.head.appendChild(style);
    return () => document.head.removeChild(style);
  }, []);

  return (
    <div style={styles.root}>
      <header style={styles.header}>
        <div style={styles.headerLeft}>
          <span style={styles.headerLogo}>🤟</span>
          <span style={styles.headerTitle}>ASL Translator</span>
        </div>
      </header>

      <main style={styles.main}>
        <h1 style={styles.heroTitle}>Sign in real time.<br />See it as text.</h1>
        <p style={styles.heroSub}>
          Use your camera to sign letters and words in American Sign Language.
          The translator shows your sentence as you sign.
        </p>
        <div style={styles.buttonGroup}>
          <button
            style={styles.btnCamera}
            onClick={() => onNavigate("translator")}
          >
            <span style={{ fontSize: 24 }}>📷</span>
            Camera
          </button>
          <button
            style={styles.btnSecondary}
            onClick={() => onNavigate("practice")}
          >
            Practice A–Z (flashcards & test)
          </button>
        </div>
      </main>

      <footer style={styles.footer}>
        Make sure the Python server is running (python server.py) and your camera is allowed.
      </footer>
    </div>
  );
}
