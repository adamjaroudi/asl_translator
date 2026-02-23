import { useState, useRef, useCallback, useEffect } from "react";

const WS_URL     = "ws://localhost:8765";
const RESTART_URL = "ws://localhost:8767";
const CONFIDENCE_THRESHOLD = 0.6;

const SIGN_EMOJI = {
  "I Love You": "🤟", "Good": "👍", "Yes": "✌️", "No": "🤚",
  "More": "🤌", "Help": "🙏", "Book": "📖", "Stop": "✋",
  "Play": "🤙", "Want": "🫳", "With": "🤜", "Same": "☝️",
};

const SUGGESTIONS = ["LOVE", "HELLO", "MORE", "HELP", "GOOD", "STOP", "YES", "NO", "WANT", "PLAY"];

const QUICK_TIPS = [
  "Position hands clearly in frame",
  "Ensure good, even lighting",
  "Use a plain background",
  "Hold each sign for ~1 second",
  "Face the camera directly",
];

function useWebSocket(url, onMessage) {
  const wsRef = useRef(null);
  const [connected, setConnected] = useState(false);
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    const ws = new WebSocket(url);
    ws.onopen    = () => setConnected(true);
    ws.onclose   = () => { setConnected(false); wsRef.current = null; };
    ws.onerror   = () => ws.close();
    ws.onmessage = (e) => onMessage(JSON.parse(e.data));
    wsRef.current = ws;
  }, [url, onMessage]);
  const disconnect = useCallback(() => wsRef.current?.close(), []);
  const send = useCallback((data) => {
    if (wsRef.current?.readyState === WebSocket.OPEN)
      wsRef.current.send(JSON.stringify(data));
  }, []);
  useEffect(() => () => wsRef.current?.close(), []);
  return { connected, connect, disconnect, send };
}

export default function ASLTranslator({ onNavigate }) {
  const [data, setData] = useState({
    frame: null, prediction: "", confidence: 0,
    isWord: false, sentence: "", numHands: 0, stablePct: 0, fps: 0,
    topAlts: [],
  });
  const [active, setActive] = useState(false);
  const [history, setHistory] = useState([]);
  const [speaking, setSpeaking] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [showDict, setShowDict] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [restarting, setRestarting]     = useState(false);

  const handleRestart = useCallback(() => {
    setRestarting(true);
    const ws = new WebSocket(RESTART_URL);
    ws.onopen = () => ws.send(JSON.stringify({ action: "restart" }));
    ws.onmessage = () => {
      ws.close();
      // Reconnect after 2.5s to give server time to restart
      setTimeout(() => setRestarting(false), 2500);
    };
    ws.onerror = () => setRestarting(false);
  }, []);

  const onMessage = useCallback((msg) => {
    if (msg.error) return;
    setData(prev => {
      if (msg.sentence && msg.sentence !== prev.sentence && msg.sentence.trim()) {
        setHistory(h => {
          if (h[h.length - 1] === msg.sentence.trim()) return h;
          return [...h.slice(-49), msg.sentence.trim()];
        });
      }
      return {
        frame:      msg.frame      ?? prev.frame,
        prediction: msg.prediction ?? "",
        confidence: msg.confidence ?? 0,
        isWord:     msg.is_word    ?? false,
        sentence:   msg.sentence   ?? prev.sentence,
        numHands:   msg.num_hands  ?? 0,
        stablePct:  msg.stable_pct ?? 0,
        fps:        msg.fps        ?? prev.fps,
        topAlts:    msg.top_alts   ?? prev.topAlts,
      };
    });
  }, []);

  const { connected, connect, disconnect, send } = useWebSocket(WS_URL, onMessage);

  const handleToggle = () => {
    if (active) { disconnect(); setActive(false); setData(d => ({ ...d, frame: null })); }
    else { connect(); setActive(true); }
  };

  const handleSpeak = () => {
    if (!data.sentence || speaking) return;
    const utt = new SpeechSynthesisUtterance(data.sentence);
    utt.onstart = () => setSpeaking(true);
    utt.onend   = () => setSpeaking(false);
    speechSynthesis.speak(utt);
  };

  const handleDownload = () => {
    if (!data.sentence) return;
    const blob = new Blob([data.sentence], { type: "text/plain" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "asl-translation.txt";
    a.click();
  };

  const confPct  = Math.round(data.confidence * 100);
  const confGood = data.confidence >= CONFIDENCE_THRESHOLD;
  const stablePct = Math.round(data.stablePct * 100);
  const signEmoji = SIGN_EMOJI[data.prediction] || "";

  return (
    <div style={s.root}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #f5f5f0; }
        button { font-family: 'IBM Plex Sans', sans-serif; }
        button:focus-visible { outline: 2px solid #1a1a1a; outline-offset: 2px; }
        @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }
      `}</style>

      {/* ── Header ── */}
      <header style={s.header}>
        <div style={s.headerLeft}>
          <span style={s.headerLogo}></span>
          <span style={s.headerTitle}>ASL Translator</span>
        </div>
        <nav style={s.headerNav}>
          {onNavigate && (
            <button style={s.headerNavBtn} onClick={() => onNavigate("home")}>← Home</button>
          )}
          <button style={s.headerNavBtn} onClick={() => onNavigate ? onNavigate("dictionary") : setShowDict(true)}>Dictionary</button>
          <button style={s.headerNavBtn} onClick={() => setShowHistory(true)}>
            History{history.length > 0 ? ` (${history.length})` : ""}
          </button>
          <button style={s.headerNavBtn}>Settings</button>
          {onNavigate && (
            <button style={{ ...s.headerNavBtn, borderColor: "#1a1a1a", fontWeight: 600 }}
              onClick={() => onNavigate("collect")}>
              Collect Data →
            </button>
          )}
          <button
            style={{ ...s.headerNavBtn, ...(restarting ? { color: "#b45309", borderColor: "#fcd34d" } : {}) }}
            onClick={handleRestart}
            disabled={restarting}
            title="Restart the Python server to load a newly trained model"
          >
            {restarting ? "↺ Restarting…" : "↺ Restart Server"}
          </button>
          {active && (
            <span style={s.fpsTag}>{data.fps > 0 ? `${data.fps} fps` : "—"}</span>
          )}
        </nav>
      </header>

      {/* ── Main two-column layout ── */}
      <div style={s.body}>
        <div style={s.topRow}>

          {/* Camera column */}
          <div style={s.camCol}>
            <div style={s.camWrap}>
              {data.frame
                ? <img src={`data:image/jpeg;base64,${data.frame}`} alt="Camera feed" style={s.camImg} />
                : <div style={s.camBlank}>
                    <span style={s.camBlankIcon}>◻</span>
                    <span style={s.camBlankText}>{active ? "Connecting to camera…" : "Camera inactive"}</span>
                  </div>
              }
              {active && connected && (
                <div style={s.recIndicator}>
                  <span style={{ ...s.recDot, animation: "blink 1s step-end infinite" }} />
                  Recording
                </div>
              )}
              {active && (
                <div style={{
                  ...s.handTag,
                  background: data.numHands > 0 ? "#e8f5e9" : "#fff8e1",
                  color:      data.numHands > 0 ? "#2e7d32" : "#f57f17",
                  borderColor:data.numHands > 0 ? "#a5d6a7" : "#ffe082",
                }}>
                  {data.numHands > 0 ? `${data.numHands} hand${data.numHands > 1 ? "s" : ""}` : "No hands detected"}
                </div>
              )}
            </div>
            <button style={{ ...s.btn, ...s.btnPrimary, marginTop: 10 }} onClick={handleToggle}>
              {active ? "Stop" : "Start / Stop"}
            </button>
          </div>

          {/* Translation column */}
          <div style={s.translationCol}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
              <div style={s.sectionLabel}>Translation</div>
              <label style={s.advancedToggle}>
                <input
                  type="checkbox"
                  checked={showAdvanced}
                  onChange={e => setShowAdvanced(e.target.checked)}
                  style={{ marginRight: 6, accentColor: "#1a1a1a" }}
                />
                Advanced
              </label>
            </div>
            <div style={s.translationBox}>
              <div style={s.translationText}>
                {data.prediction
                  ? `"${data.prediction.toUpperCase()}"`
                  : <span style={{ color: "#aaa", fontWeight: 400, fontSize: 22 }}>Waiting for sign…</span>
                }
              </div>
              {signEmoji && <div style={s.translationEmoji}>{signEmoji}</div>}
            </div>
            {data.prediction && (
              <div style={s.confRow}>
                <div style={s.confBarWrap}>
                  <div style={{
                    ...s.confBar,
                    width: `${confPct}%`,
                    background: confGood ? "#2e7d32" : "#b45309",
                  }} />
                </div>
                <span style={{ ...s.confNum, color: confGood ? "#2e7d32" : "#b45309" }}>
                  {confPct}%
                </span>
              </div>
            )}

            {/* Advanced: top alternatives */}
            {showAdvanced && data.topAlts && data.topAlts.length > 0 && (
              <div style={s.altsPanel}>
                <div style={s.altsPanelTitle}>Also detecting</div>
                {data.topAlts.map(([label, conf], i) => {
                  const pct = Math.round(conf * 100);
                  const isTop = i === 0;
                  return (
                    <div key={label} style={s.altRow}>
                      <span style={{ ...s.altLabel, fontWeight: isTop ? 600 : 400, color: isTop ? C.text : C.textMid }}>
                        {label}
                      </span>
                      <div style={s.altBarWrap}>
                        <div style={{
                          ...s.altBar,
                          width: `${pct}%`,
                          background: isTop ? "#1a1a1a" : "#c8c8c4",
                        }} />
                      </div>
                      <span style={{ ...s.altPct, color: isTop ? C.text : C.textDim }}>{pct}%</span>
                    </div>
                  );
                })}
              </div>
            )}

            <button
              style={{ ...s.btn, ...s.btnSecondary, marginTop: "auto" }}
              onClick={handleSpeak}
              disabled={!data.sentence}
            >
              {speaking ? "▶  Speaking…" : "▶  Audio Playback"}
            </button>
          </div>
        </div>

        {/* ── Bottom three-column row ── */}
        <div style={s.bottomRow}>

          {/* Detected sign + sentence */}
          <div style={s.card}>
            <div style={s.sectionLabel}>Detected Sign</div>
            <div style={s.detectedRow}>
              <span style={s.detectedEmoji}>{data.prediction ? (signEmoji || "—") : "—"}</span>
              <div style={s.detectedRight}>
                <div style={s.detectedName}>{data.prediction || "Waiting for sign…"}</div>
                {data.prediction && (
                  <div style={s.holdRow}>
                    <div style={s.holdTrack}>
                      <div style={{ ...s.holdFill, width: `${stablePct}%` }} />
                    </div>
                    <span style={s.holdNum}>{confPct}%</span>
                  </div>
                )}
              </div>
            </div>
            <div style={s.divider} />
            <div style={s.sectionLabel}>Sentence</div>
            <div style={s.sentenceArea}>
              {data.sentence
                ? <span style={s.sentenceText}>{data.sentence}</span>
                : <span style={s.sentencePlaceholder}>Your sentence will appear here…</span>
              }
            </div>
            <div style={s.sentenceBtns}>
              <button style={{ ...s.btn, ...s.btnMini }} onClick={() => send({ action: "space" })} disabled={!connected}>Space</button>
              <button style={{ ...s.btn, ...s.btnMini }} onClick={() => send({ action: "backspace" })} disabled={!connected}>Backspace</button>
              <button style={{ ...s.btn, ...s.btnMini, color: "#b91c1c", borderColor: "#fca5a5" }} onClick={() => send({ action: "clear" })} disabled={!connected}>Clear</button>
            </div>
          </div>

          {/* Suggestions */}
          <div style={s.card}>
            <div style={s.sectionLabel}>Suggestions</div>
            <div style={s.suggestGrid}>
              {SUGGESTIONS.map(w => (
                <button key={w} style={{ ...s.btn, ...s.btnSuggest }}>{w}</button>
              ))}
            </div>
          </div>

          {/* Quick tips */}
          <div style={s.card}>
            <div style={s.sectionLabel}>Quick Tips</div>
            <ol style={s.tipList}>
              {QUICK_TIPS.map((t, i) => (
                <li key={i} style={s.tipItem}>{t}</li>
              ))}
            </ol>
          </div>
        </div>

        {/* ── Footer bar ── */}
        <div style={s.footerBar}>
          <button style={{ ...s.btn, ...s.btnSecondary }}>Practice Mode</button>
          <button style={{ ...s.btn, ...s.btnSecondary }} onClick={() => setShowHistory(true)}>History</button>
          <button style={{ ...s.btn, ...s.btnSecondary }} onClick={handleDownload} disabled={!data.sentence}>Download</button>
          <div style={{ flex: 1 }} />
          <button style={{ ...s.btn, ...s.btnPrimary }}>Learn ASL →</button>
        </div>
      </div>

      {/* ── Modals ── */}
      {(showHistory || showDict) && (
        <div style={s.overlay} onClick={() => { setShowHistory(false); setShowDict(false); }}>
          <div style={s.modal} onClick={e => e.stopPropagation()}>
            <div style={s.modalHeader}>
              <span style={s.modalTitle}>{showDict ? "Sign Dictionary" : "Translation History"}</span>
              <button style={s.modalClose} onClick={() => { setShowHistory(false); setShowDict(false); }}>✕</button>
            </div>
            {showHistory && (
              history.length === 0
                ? <p style={{ color: "#999", padding: "12px 0", fontSize: 14 }}>No history yet.</p>
                : [...history].reverse().map((item, i) => (
                    <div key={i} style={s.historyRow}>{item}</div>
                  ))
            )}
            {showDict && (
              <div style={s.dictGrid}>
                {Object.entries(SIGN_EMOJI).map(([word, emoji]) => (
                  <div key={word} style={s.dictCell}>
                    <span style={{ fontSize: 28 }}>{emoji}</span>
                    <span style={s.dictCellLabel}>{word}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Design tokens ─────────────────────────────────────────────────────────────
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

const s = {
  root: {
    minHeight: "100vh",
    background: C.bg,
    fontFamily: "'IBM Plex Sans', system-ui, sans-serif",
    fontSize: 14,
    color: C.text,
    display: "flex",
    flexDirection: "column",
  },

  // Header
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
  headerNav: { display: "flex", alignItems: "center", gap: 4 },
  headerNavBtn: {
    padding: "5px 13px",
    background: "none",
    border: `1px solid ${C.border}`,
    borderRadius: 3,
    fontSize: 13,
    fontWeight: 500,
    color: C.textMid,
    cursor: "pointer",
  },
  fpsTag: {
    marginLeft: 8,
    padding: "4px 10px",
    background: "#f0f0eb",
    border: `1px solid ${C.border}`,
    borderRadius: 3,
    fontSize: 12,
    fontFamily: "'IBM Plex Mono', monospace",
    color: C.textDim,
  },

  // Layout
  body: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
    gap: 16,
    padding: "20px 28px 24px",
    maxWidth: 1200,
    width: "100%",
    margin: "0 auto",
  },
  topRow: {
    display: "grid",
    gridTemplateColumns: "1fr 320px",
    gap: 16,
    alignItems: "start",
  },

  // Camera
  camCol: { display: "flex", flexDirection: "column" },
  camWrap: {
    position: "relative",
    background: "#111",
    border: `1px solid ${C.border}`,
    aspectRatio: "16/9",
    overflow: "hidden",
  },
  camImg: { width: "100%", height: "100%", objectFit: "cover", display: "block" },
  camBlank: {
    height: "100%",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    gap: 10,
  },
  camBlankIcon: { fontSize: 36, color: "#444" },
  camBlankText: { color: "#666", fontSize: 13 },
  recIndicator: {
    position: "absolute",
    bottom: 12,
    left: "50%",
    transform: "translateX(-50%)",
    background: "rgba(0,0,0,0.72)",
    color: "#fff",
    fontSize: 13,
    fontWeight: 500,
    padding: "5px 14px",
    display: "flex",
    alignItems: "center",
    gap: 7,
  },
  recDot: {
    display: "inline-block",
    width: 8,
    height: 8,
    borderRadius: "50%",
    background: "#ef4444",
  },
  handTag: {
    position: "absolute",
    top: 10,
    right: 10,
    padding: "3px 10px",
    fontSize: 12,
    fontWeight: 500,
    border: "1px solid",
  },

  // Translation panel
  translationCol: {
    background: C.surface,
    border: `1px solid ${C.border}`,
    padding: "18px 20px",
    display: "flex",
    flexDirection: "column",
    gap: 12,
    minHeight: 260,
  },
  translationBox: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    gap: 10,
    padding: "10px 0",
    borderTop: `1px solid ${C.border}`,
    borderBottom: `1px solid ${C.border}`,
  },
  translationText: {
    fontSize: 28,
    fontWeight: 600,
    letterSpacing: "-0.02em",
    color: C.text,
    textAlign: "center",
    lineHeight: 1.2,
  },
  translationEmoji: { fontSize: 52 },
  confRow: { display: "flex", alignItems: "center", gap: 10 },
  confBarWrap: {
    flex: 1,
    height: 6,
    background: "#e5e5e0",
    overflow: "hidden",
  },
  confBar: { height: "100%", transition: "width 0.2s ease" },
  confNum: { fontSize: 13, fontFamily: "'IBM Plex Mono', monospace", fontWeight: 500, minWidth: 36, textAlign: "right" },

  // Advanced alternatives panel
  advancedToggle: {
    display: "flex", alignItems: "center", fontSize: 12,
    fontWeight: 500, color: C.textMid, cursor: "pointer",
    userSelect: "none",
  },
  altsPanel: {
    border: `1px solid ${C.border}`,
    padding: "10px 12px",
    background: "#fafaf8",
    display: "flex",
    flexDirection: "column",
    gap: 7,
  },
  altsPanelTitle: {
    fontSize: 10, fontWeight: 600, letterSpacing: "0.1em",
    textTransform: "uppercase", color: C.textDim, marginBottom: 2,
  },
  altRow: { display: "flex", alignItems: "center", gap: 8 },
  altLabel: { fontSize: 13, minWidth: 68, fontFamily: "'IBM Plex Mono', monospace" },
  altBarWrap: { flex: 1, height: 5, background: "#e5e5e0", overflow: "hidden" },
  altBar: { height: "100%", transition: "width 0.15s ease" },
  altPct: { fontSize: 12, fontFamily: "'IBM Plex Mono', monospace", minWidth: 32, textAlign: "right" },

  // Bottom row
  bottomRow: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr 1fr",
    gap: 16,
  },
  card: {
    background: C.surface,
    border: `1px solid ${C.border}`,
    padding: "16px 18px",
    display: "flex",
    flexDirection: "column",
    gap: 10,
  },
  sectionLabel: {
    fontSize: 11,
    fontWeight: 600,
    letterSpacing: "0.08em",
    textTransform: "uppercase",
    color: C.textDim,
    marginBottom: 2,
  },

  // Detected sign
  detectedRow: { display: "flex", alignItems: "center", gap: 14 },
  detectedEmoji: { fontSize: 38, lineHeight: 1, minWidth: 44 },
  detectedRight: { flex: 1, display: "flex", flexDirection: "column", gap: 6 },
  detectedName: { fontSize: 18, fontWeight: 600, color: C.text },
  holdRow: { display: "flex", alignItems: "center", gap: 8 },
  holdTrack: { flex: 1, height: 5, background: "#e5e5e0", overflow: "hidden" },
  holdFill: { height: "100%", background: C.primary, transition: "width 0.1s linear" },
  holdNum: { fontSize: 12, fontFamily: "'IBM Plex Mono', monospace", color: C.textDim, minWidth: 32, textAlign: "right" },

  divider: { borderTop: `1px solid ${C.border}`, margin: "2px 0" },

  // Sentence
  sentenceArea: {
    minHeight: 48,
    padding: "10px 12px",
    background: "#fafaf8",
    border: `1px solid ${C.border}`,
    fontSize: 15,
    fontWeight: 500,
    lineHeight: 1.5,
    color: C.text,
    wordBreak: "break-word",
  },
  sentenceText: { color: C.text },
  sentencePlaceholder: { color: C.textDim, fontWeight: 400, fontStyle: "italic" },
  sentenceBtns: { display: "flex", gap: 6 },

  // Suggestions
  suggestGrid: { display: "flex", flexWrap: "wrap", gap: 6 },

  // Tips
  tipList: { paddingLeft: 18, display: "flex", flexDirection: "column", gap: 7 },
  tipItem: { fontSize: 13, color: C.textMid, lineHeight: 1.4 },

  // Buttons
  btn: {
    padding: "7px 14px",
    border: `1px solid ${C.border}`,
    borderRadius: 3,
    fontSize: 13,
    fontWeight: 500,
    cursor: "pointer",
    background: C.surface,
    color: C.text,
    transition: "background 0.1s",
    lineHeight: 1.4,
  },
  btnPrimary: {
    background: C.primary,
    color: "#fff",
    border: `1px solid ${C.primary}`,
  },
  btnSecondary: {
    background: C.surface,
    color: C.text,
    border: `1px solid ${C.border}`,
  },
  btnMini: {
    padding: "5px 10px",
    fontSize: 12,
    flex: 1,
  },
  btnSuggest: {
    padding: "6px 12px",
    fontSize: 13,
    fontWeight: 600,
    letterSpacing: "0.04em",
  },

  // Footer
  footerBar: {
    display: "flex",
    gap: 8,
    alignItems: "center",
    paddingTop: 4,
  },

  // Modals
  overlay: {
    position: "fixed",
    inset: 0,
    background: "rgba(0,0,0,0.35)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    zIndex: 100,
  },
  modal: {
    background: C.surface,
    border: `1px solid ${C.border}`,
    padding: "24px 28px",
    width: "min(500px, 92vw)",
    maxHeight: "78vh",
    overflowY: "auto",
    boxShadow: "0 8px 32px rgba(0,0,0,0.16)",
  },
  modalHeader: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 18,
    paddingBottom: 14,
    borderBottom: `1px solid ${C.border}`,
  },
  modalTitle: { fontSize: 16, fontWeight: 600, color: C.text },
  modalClose: {
    background: "none",
    border: `1px solid ${C.border}`,
    borderRadius: 3,
    width: 28,
    height: 28,
    cursor: "pointer",
    fontSize: 14,
    color: C.textMid,
  },
  historyRow: {
    padding: "9px 0",
    borderBottom: `1px solid #f0f0eb`,
    fontSize: 14,
    color: C.textMid,
  },
  dictGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(4, 1fr)",
    gap: 10,
  },
  dictCell: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: 6,
    padding: "12px 8px",
    border: `1px solid ${C.border}`,
    background: "#fafaf8",
  },
  dictCellLabel: { fontSize: 12, fontWeight: 500, color: C.textMid, textAlign: "center" },
};