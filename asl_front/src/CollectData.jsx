import { useState, useRef, useCallback, useEffect } from "react";

const WS_URL      = "ws://localhost:8766";
const RESTART_URL = "ws://localhost:8767";
const CUSTOM_WORDS_KEY = "asl_custom_words"; // shared with Dictionary.jsx

const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");

const DEFAULT_WORD_SIGNS = [
  "[I-LOVE-YOU]", "[GOOD]", "[MORE]", "[HELP]", "[BOOK]",
  "[STOP]", "[PLAY]", "[WANT]", "[WITH]", "[SAME]",
  "[NO]", "[YES]", "[FRIEND]", "[WORK]", "[FINISH]",
  "[GO]", "[SIT]", "[BIG]", "[SMALL]", "[LOVE]", "[EAT]", "[DRINK]",
];

function wordToLabel(word) {
  // Convert "i love you" -> "[I-LOVE-YOU]"
  return "[" + word.trim().toUpperCase().replace(/\s+/g, "-") + "]";
}

function loadCustomWordLabels() {
  try {
    const raw = JSON.parse(localStorage.getItem(CUSTOM_WORDS_KEY) || "[]");
    return raw.map(w => wordToLabel(w.word || w));
  } catch { return []; }
}

function formatLabel(label) {
  if (label.startsWith("[") && label.endsWith("]"))
    return label.slice(1, -1).replace(/-/g, " ");
  return label;
}

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

export default function CollectData({ onNavigate }) {
  const [frame, setFrame]           = useState(null);
  const [numHands, setNumHands]     = useState(0);
  const [recording, setRecording]   = useState(false);
  const [currentLabel, setCurrentLabel] = useState("");
  const [sampleCount, setSampleCount]   = useState(0);
  const [totalSamples, setTotalSamples] = useState(0);
  const [classStats, setClassStats] = useState({});
  const [active, setActive]         = useState(false);
  const [fps, setFps]               = useState(0);
  const [tab, setTab]               = useState("letters"); // "letters" | "words"
  const [saveMsg, setSaveMsg]         = useState("");
  const [confirmTrim, setConfirmTrim]   = useState(false);
  const [training, setTraining]         = useState(false);
  const [trainLog, setTrainLog]         = useState([]);
  const [showTrainLog, setShowTrainLog] = useState(false);
  const [restarting, setRestarting]     = useState(false);
  const trainLogRef = useRef(null);

  // Word signs = defaults + any custom words added via Dictionary or here
  const [customWordSigns, setCustomWordSigns] = useState(loadCustomWordLabels);
  const WORD_SIGNS = [
    ...DEFAULT_WORD_SIGNS,
    ...customWordSigns.filter(l => !DEFAULT_WORD_SIGNS.includes(l)),
  ];

  // Add new sign modal
  const [showAddSign, setShowAddSign]   = useState(false);
  const [newSignInput, setNewSignInput] = useState("");
  const [newSignError, setNewSignError] = useState("");

  const handleRestart = useCallback(() => {
    setRestarting(true);
    const ws = new WebSocket(RESTART_URL);
    ws.onopen    = () => ws.send(JSON.stringify({ action: "restart" }));
    ws.onmessage = () => { ws.close(); setTimeout(() => setRestarting(false), 2500); };
    ws.onerror   = () => setRestarting(false);
  }, []);

  const onMessage = useCallback((msg) => {
    if (msg.frame)        setFrame(msg.frame);
    if (msg.num_hands !== undefined) setNumHands(msg.num_hands);
    if (msg.fps !== undefined)       setFps(msg.fps);
    if (msg.sample_count !== undefined) setSampleCount(msg.sample_count);
    if (msg.total_samples !== undefined) setTotalSamples(msg.total_samples);
    if (msg.class_stats)  setClassStats(msg.class_stats);
    if (msg.saved) {
      setSaveMsg(`Saved ${msg.saved} samples to ${msg.path}`);
      setTimeout(() => setSaveMsg(""), 3000);
    }
    if (msg.train_log) {
      setTrainLog(prev => [...prev, msg.train_log]);
      setShowTrainLog(true);
      setTimeout(() => {
        if (trainLogRef.current)
          trainLogRef.current.scrollTop = trainLogRef.current.scrollHeight;
      }, 30);
    }
    if (msg.train_start) { setTraining(true); setTrainLog([]); setShowTrainLog(true); }
    if (msg.train_done)  { setTraining(false); }
  }, []);

  const { connected, connect, disconnect, send } = useWebSocket(WS_URL, onMessage);

  const handleToggle = () => {
    if (active) {
      send({ action: "stop_recording" });
      disconnect();
      setActive(false);
      setFrame(null);
      setRecording(false);
    } else {
      connect();
      setActive(true);
    }
  };

  const startRecording = (label) => {
    if (!connected) return;
    setCurrentLabel(label);
    setSampleCount(0);
    setRecording(true);
    send({ action: "start_recording", label });
  };

  const stopRecording = () => {
    if (!connected) return;
    setRecording(false);
    send({ action: "stop_recording" });
  };

  const handleSave = () => {
    if (!connected) return;
    send({ action: "save" });
  };

  const handleTrim = () => {
    if (!connected) return;
    send({ action: "trim", keep: 100 });
    setConfirmTrim(false);
  };

  const handleTrain = () => {
    if (!connected || training) return;
    send({ action: "train" });
  };

  const handleAddSign = () => {
    const label = wordToLabel(newSignInput);
    if (!newSignInput.trim()) { setNewSignError("Enter a word or phrase."); return; }
    if (WORD_SIGNS.includes(label)) { setNewSignError("This sign already exists."); return; }
    // Save to localStorage so Dictionary picks it up too
    try {
      const existing = JSON.parse(localStorage.getItem(CUSTOM_WORDS_KEY) || "[]");
      const word = newSignInput.trim().toLowerCase();
      if (!existing.find(w => (w.word || w) === word)) {
        existing.unshift({ word, videoId: null });
        localStorage.setItem(CUSTOM_WORDS_KEY, JSON.stringify(existing));
      }
    } catch {}
    setCustomWordSigns(prev => [label, ...prev]);
    setCurrentLabel(label);
    setTab("words");
    setNewSignInput(""); setNewSignError(""); setShowAddSign(false);
  };

  const samplesForLabel = (label) => classStats[label] || 0;
  const totalForLabel   = (label) => samplesForLabel(label);
  const TARGET = 200;

  return (
    <div style={s.root}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #f5f5f0; }
        button { font-family: 'IBM Plex Sans', sans-serif; cursor: pointer; }
        button:disabled { opacity: 0.4; cursor: default; }
        @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }
      `}</style>

      {/* Header */}
      <header style={s.header}>
        <div style={s.headerLeft}>
          <button style={s.backBtn} onClick={() => onNavigate("home")}>← Home</button>
          <span style={s.dividerV} />
          <span style={s.headerTitle}>Collect Training Data</span>
        </div>
        <div style={s.headerRight}>
          {active && <span style={s.fpsTag}>{fps > 0 ? `${fps} fps` : "—"}</span>}
          <span style={{ ...s.statusDot, background: connected ? "#2e7d32" : "#9e9e9e" }} />
          <span style={s.statusText}>{connected ? "Connected" : "Disconnected"}</span>
          <button
            style={{ ...s.btn, ...(restarting ? { color: "#b45309", borderColor: "#fcd34d" } : {}) }}
            onClick={handleRestart}
            disabled={restarting}
            title="Restart server to load newly trained model"
          >
            {restarting ? "↺ Restarting…" : "↺ Restart Server"}
          </button>
          <button style={{ ...s.btn, ...s.btnPrimary }} onClick={handleToggle}>
            {active ? "Stop Camera" : "Start Camera"}
          </button>
        </div>
      </header>

      <div style={s.body}>
        {/* Left: camera + controls */}
        <div style={s.leftCol}>
          <div style={s.camWrap}>
            {frame
              ? <img src={`data:image/jpeg;base64,${frame}`} alt="feed" style={s.camImg} />
              : <div style={s.camBlank}>
                  <span style={s.camBlankIcon}>◻</span>
                  <span style={s.camBlankText}>{active ? "Connecting…" : "Camera inactive"}</span>
                </div>
            }
            {recording && (
              <div style={s.recBadge}>
                <span style={{ ...s.recDot, animation: "blink 1s step-end infinite" }} />
                Recording: <strong style={{ marginLeft: 4 }}>{formatLabel(currentLabel)}</strong>
                <span style={s.recCount}>{sampleCount} samples</span>
              </div>
            )}
            {active && (
              <div style={{
                ...s.handTag,
                background:  numHands > 0 ? "#e8f5e9" : "#fff8e1",
                color:       numHands > 0 ? "#2e7d32" : "#f57f17",
                borderColor: numHands > 0 ? "#a5d6a7" : "#ffe082",
              }}>
                {numHands > 0 ? `${numHands} hand${numHands > 1 ? "s" : ""}` : "No hands"}
              </div>
            )}
          </div>

          {/* Recording controls */}
          <div style={s.recControls}>
            <div style={s.recCtrlRow}>
              <div style={s.recLabelDisplay}>
                {currentLabel
                  ? <><span style={s.recLabelPill}>{formatLabel(currentLabel)}</span><span style={s.recLabelSub}>{sampleCount} collected this session</span></>
                  : <span style={s.recLabelEmpty}>Select a sign below to record</span>
                }
              </div>
              <button
                style={{ ...s.btn, ...(recording ? s.btnDanger : s.btnRecord) }}
                onClick={recording ? stopRecording : () => currentLabel && startRecording(currentLabel)}
                disabled={!connected || (!recording && !currentLabel)}
              >
                {recording ? "■ Stop" : "● Record"}
              </button>
            </div>
          </div>

          {/* Stats */}
          <div style={s.statsBox}>
            <div style={s.statsRow}>
              <div style={s.statItem}>
                <span style={s.statNum}>{totalSamples}</span>
                <span style={s.statLabel}>Total samples</span>
              </div>
              <div style={s.statItem}>
                <span style={s.statNum}>{Object.keys(classStats).length}</span>
                <span style={s.statLabel}>Classes</span>
              </div>
              <div style={s.statItem}>
                <span style={s.statNum}>{recording ? sampleCount : "—"}</span>
                <span style={s.statLabel}>This session</span>
              </div>
            </div>
            <div style={s.saveBtnRow}>
              <button style={{ ...s.btn, ...s.btnPrimary, flex: 1 }} onClick={handleSave} disabled={!connected || totalSamples === 0}>
                Save to CSV
              </button>
              {!confirmTrim
                ? <button
                    style={{ ...s.btn, color: "#b91c1c", borderColor: "#fca5a5" }}
                    onClick={() => setConfirmTrim(true)}
                    disabled={!connected || totalSamples <= 100}
                    title="Keep only first 100 rows of CSV"
                  >
                    Trim CSV
                  </button>
                : <div style={s.confirmRow}>
                    <span style={s.confirmText}>Keep only first 100 rows?</span>
                    <button style={{ ...s.btn, background: "#b91c1c", color: "#fff", border: "1px solid #b91c1c" }} onClick={handleTrim}>
                      Yes, trim
                    </button>
                    <button style={s.btn} onClick={() => setConfirmTrim(false)}>Cancel</button>
                  </div>
              }
            </div>
            {saveMsg && <div style={s.saveMsg}>{saveMsg}</div>}

            {/* Train model */}
            <div style={s.divider} />
            <button
              style={{ ...s.btn, ...s.btnPrimary, ...(training ? { opacity: 0.7 } : {}) }}
              onClick={handleTrain}
              disabled={!connected || training || totalSamples === 0}
            >
              {training ? "⏳ Training…" : "▶ Train Model"}
            </button>
            {training && (
              <div style={s.trainingNote}>
                Training in progress — this takes a few minutes. Do not close the server.
              </div>
            )}
            {showTrainLog && trainLog.length > 0 && (
              <div style={s.trainLogWrap}>
                <div style={s.trainLogHeader}>
                  <span style={s.trainLogTitle}>
                    {training ? "Training output" : "Training complete ✓"}
                  </span>
                  <button style={s.trainLogClose} onClick={() => setShowTrainLog(false)}>✕</button>
                </div>
                <div style={s.trainLog} ref={trainLogRef}>
                  {trainLog.map((line, i) => (
                    <div key={i} style={s.trainLogLine}>{line}</div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Tips */}
          <div style={s.tipsBox}>
            <div style={s.tipsTitle}>Tips for good data</div>
            <ul style={s.tipsList}>
              <li>Aim for 200+ samples per sign</li>
              <li>Vary hand angle & position slightly</li>
              <li>Use consistent, even lighting</li>
              <li>Record in same conditions you'll use the translator</li>
              <li>After collecting, run <code style={s.code}>python train_model.py</code></li>
            </ul>
          </div>
        </div>

        {/* Right: sign selector */}
        <div style={s.rightCol}>
          <div style={s.tabs}>
            <button
              style={{ ...s.tabBtn, ...(tab === "letters" ? s.tabBtnActive : {}) }}
              onClick={() => setTab("letters")}
            >
              Letters (A–Z)
            </button>
            <button
              style={{ ...s.tabBtn, ...(tab === "words" ? s.tabBtnActive : {}) }}
              onClick={() => setTab("words")}
            >
              Word Signs
            </button>
            <div style={{ flex: 1 }} />
            {tab === "words" && (
              <button
                style={{ ...s.btn, ...s.btnPrimary, fontSize: 12, padding: "5px 12px", margin: "6px 0" }}
                onClick={() => { setNewSignInput(""); setNewSignError(""); setShowAddSign(true); }}
              >
                + Add Sign
              </button>
            )}
          </div>

          <div style={s.signGrid}>
            {(tab === "letters" ? LETTERS : WORD_SIGNS).map((label) => {
              const count   = samplesForLabel(label);
              const pct     = Math.min(count / TARGET, 1);
              const isActive = currentLabel === label;
              return (
                <button
                  key={label}
                  style={{
                    ...s.signBtn,
                    ...(isActive ? s.signBtnActive : {}),
                    ...(count >= TARGET ? s.signBtnDone : {}),
                  }}
                  onClick={() => {
                    setCurrentLabel(label);
                    if (recording) send({ action: "stop_recording" });
                    setRecording(false);
                  }}
                  disabled={!connected}
                >
                  <span style={s.signBtnLabel}>{formatLabel(label)}</span>
                  <div style={s.signBtnBar}>
                    <div style={{ ...s.signBtnFill, width: `${pct * 100}%`, background: count >= TARGET ? "#2e7d32" : "#1a1a1a" }} />
                  </div>
                  <span style={s.signBtnCount}>{count}</span>
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* ── Add Sign Modal ── */}
      {showAddSign && (
        <div style={s.overlay} onClick={() => setShowAddSign(false)}>
          <div style={s.modal} onClick={e => e.stopPropagation()}>
            <div style={s.modalHeader}>
              <span style={s.modalTitle}>Add a new sign to collect</span>
              <button style={s.modalClose} onClick={() => setShowAddSign(false)}>✕</button>
            </div>
            <div style={s.modalBody}>
              <label style={s.fieldLabel}>Word or phrase</label>
              <input
                style={s.fieldInput}
                type="text"
                placeholder="e.g. bathroom, i love you"
                value={newSignInput}
                onChange={e => { setNewSignInput(e.target.value); setNewSignError(""); }}
                onKeyDown={e => e.key === "Enter" && handleAddSign()}
                autoFocus
              />
              {newSignInput && (
                <div style={s.fieldPreview}>
                  Label will be: <strong>{wordToLabel(newSignInput)}</strong>
                </div>
              )}
              {newSignError && <div style={s.fieldError}>{newSignError}</div>}
              <div style={s.modalFooter}>
                <button style={{ ...s.btn, ...s.btnPrimary }} onClick={handleAddSign}>Add sign</button>
                <button style={s.btn} onClick={() => setShowAddSign(false)}>Cancel</button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────

const C = {
  bg:      "#f5f5f0",
  surface: "#ffffff",
  border:  "#d4d4d0",
  text:    "#1a1a1a",
  textMid: "#555550",
  textDim: "#888880",
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
  header: {
    background: C.surface,
    borderBottom: `1px solid ${C.border}`,
    padding: "0 28px",
    height: 52,
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 16,
  },
  headerLeft: { display: "flex", alignItems: "center", gap: 14 },
  backBtn: {
    background: "none", border: "none", fontSize: 13,
    fontWeight: 500, color: C.textMid, padding: 0,
  },
  dividerV: { width: 1, height: 18, background: C.border, display: "block" },
  headerTitle: { fontSize: 15, fontWeight: 600, color: C.text },
  headerRight: { display: "flex", alignItems: "center", gap: 10 },
  fpsTag: {
    padding: "3px 9px", background: "#f0f0eb", border: `1px solid ${C.border}`,
    fontSize: 12, fontFamily: "'IBM Plex Mono', monospace", color: C.textDim,
  },
  statusDot: { width: 8, height: 8, borderRadius: "50%", display: "inline-block" },
  statusText: { fontSize: 13, color: C.textMid },

  body: {
    flex: 1,
    display: "grid",
    gridTemplateColumns: "480px 1fr",
    gap: 20,
    padding: "20px 28px 28px",
    maxWidth: 1200,
    width: "100%",
    margin: "0 auto",
    alignItems: "start",
  },

  // Camera
  leftCol: { display: "flex", flexDirection: "column", gap: 14 },
  camWrap: {
    position: "relative",
    background: "#111",
    border: `1px solid ${C.border}`,
    aspectRatio: "4/3",
    overflow: "hidden",
  },
  camImg: { width: "100%", height: "100%", objectFit: "cover", display: "block" },
  camBlank: {
    height: "100%", display: "flex", flexDirection: "column",
    alignItems: "center", justifyContent: "center", gap: 10,
  },
  camBlankIcon: { fontSize: 32, color: "#444" },
  camBlankText: { color: "#666", fontSize: 13 },
  recBadge: {
    position: "absolute", bottom: 10, left: "50%", transform: "translateX(-50%)",
    background: "rgba(0,0,0,0.75)", color: "#fff", fontSize: 13, fontWeight: 500,
    padding: "6px 16px", display: "flex", alignItems: "center", gap: 8,
    whiteSpace: "nowrap",
  },
  recDot: { display: "inline-block", width: 8, height: 8, borderRadius: "50%", background: "#ef4444" },
  recCount: { marginLeft: 8, color: "#aaa", fontFamily: "'IBM Plex Mono', monospace", fontSize: 12 },
  handTag: {
    position: "absolute", top: 10, right: 10, padding: "3px 10px",
    fontSize: 12, fontWeight: 500, border: "1px solid",
  },

  recControls: {
    background: C.surface,
    border: `1px solid ${C.border}`,
    padding: "14px 16px",
  },
  recCtrlRow: { display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 },
  recLabelDisplay: { display: "flex", alignItems: "center", gap: 10, flex: 1 },
  recLabelPill: {
    padding: "3px 10px", background: "#1a1a1a", color: "#fff",
    fontSize: 13, fontWeight: 600, letterSpacing: "0.05em",
  },
  recLabelSub: { fontSize: 12, color: C.textDim, fontFamily: "'IBM Plex Mono', monospace" },
  recLabelEmpty: { fontSize: 13, color: C.textDim, fontStyle: "italic" },

  statsBox: {
    background: C.surface,
    border: `1px solid ${C.border}`,
    padding: "14px 16px",
    display: "flex",
    flexDirection: "column",
    gap: 12,
  },
  statsRow: { display: "flex", gap: 0 },
  statItem: {
    flex: 1, display: "flex", flexDirection: "column", gap: 2,
    paddingRight: 16, borderRight: `1px solid ${C.border}`,
    marginRight: 16,
  },
  statNum: { fontSize: 22, fontWeight: 600, fontFamily: "'IBM Plex Mono', monospace", color: C.text },
  statLabel: { fontSize: 11, color: C.textDim, textTransform: "uppercase", letterSpacing: "0.06em" },
  saveBtnRow: { display: "flex", gap: 8 },
  confirmRow: { display: "flex", alignItems: "center", gap: 6, flex: 1, flexWrap: "wrap" },
  confirmText: { fontSize: 12, color: "#b91c1c", fontWeight: 500, flex: 1 },
  saveMsg: {
    fontSize: 12, color: "#2e7d32", padding: "6px 10px",
    background: "#e8f5e9", border: "1px solid #a5d6a7",
  },
  divider: { borderTop: `1px solid ${C.border}` },
  trainingNote: {
    fontSize: 12, color: "#b45309", padding: "6px 10px",
    background: "#fffbeb", border: "1px solid #fcd34d",
  },
  trainLogWrap: {
    border: `1px solid ${C.border}`,
    overflow: "hidden",
  },
  trainLogHeader: {
    display: "flex", justifyContent: "space-between", alignItems: "center",
    padding: "7px 10px", background: "#f0f0eb", borderBottom: `1px solid ${C.border}`,
  },
  trainLogTitle: { fontSize: 11, fontWeight: 600, letterSpacing: "0.06em", color: C.textMid },
  trainLogClose: {
    background: "none", border: "none", fontSize: 13,
    color: C.textDim, cursor: "pointer", padding: "0 2px",
  },
  trainLog: {
    maxHeight: 180, overflowY: "auto", padding: "8px 10px",
    background: "#1a1a1a", fontFamily: "'IBM Plex Mono', monospace",
    fontSize: 11, lineHeight: 1.6,
  },
  trainLogLine: { color: "#d4d4d0", whiteSpace: "pre-wrap", wordBreak: "break-all" },

  // Modal
  overlay: {
    position: "fixed", inset: 0, background: "rgba(0,0,0,0.4)",
    display: "flex", alignItems: "center", justifyContent: "center", zIndex: 100,
  },
  modal: {
    background: C.surface, border: `1px solid ${C.border}`,
    width: "100%", maxWidth: 420, boxShadow: "0 8px 32px rgba(0,0,0,0.12)",
  },
  modalHeader: {
    display: "flex", alignItems: "center", justifyContent: "space-between",
    padding: "14px 18px", borderBottom: `1px solid ${C.border}`,
  },
  modalTitle:  { fontSize: 15, fontWeight: 600 },
  modalClose:  { background: "none", border: "none", fontSize: 16, color: C.textDim, cursor: "pointer" },
  modalBody:   { padding: "18px 18px 14px" },
  modalFooter: { display: "flex", gap: 8, marginTop: 18 },
  fieldLabel:  { display: "block", fontSize: 12, fontWeight: 600, color: C.textMid, marginBottom: 5 },
  fieldInput: {
    width: "100%", padding: "8px 10px", border: `1px solid ${C.border}`,
    fontSize: 13, color: C.text, background: C.bg, outline: "none",
  },
  fieldPreview: {
    marginTop: 7, fontSize: 12, color: C.textMid,
    padding: "5px 8px", background: "#f0f0eb", border: `1px solid ${C.border}`,
    fontFamily: "'IBM Plex Mono', monospace",
  },
  fieldError: {
    marginTop: 7, fontSize: 12, color: "#b91c1c",
    padding: "5px 8px", background: "#fef2f2", border: "1px solid #fca5a5",
  },

  tipsBox: {
    border: `1px solid ${C.border}`,
    padding: "14px 16px",
    background: "#fafaf8",
  },
  tipsTitle: { fontSize: 11, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.08em", color: C.textDim, marginBottom: 8 },
  tipsList: { paddingLeft: 16, display: "flex", flexDirection: "column", gap: 5 },
  code: { fontFamily: "'IBM Plex Mono', monospace", background: "#f0f0eb", padding: "1px 5px", fontSize: 11 },

  // Right panel
  rightCol: { display: "flex", flexDirection: "column", gap: 0 },
  tabs: {
    display: "flex",
    borderBottomWidth: 1, borderBottomStyle: "solid", borderBottomColor: C.border,
    marginBottom: 14,
  },
  tabBtn: {
    padding: "9px 18px", background: "none", border: "none",
    borderBottomWidth: 2, borderBottomStyle: "solid", borderBottomColor: "transparent",
    fontSize: 13, fontWeight: 500, color: C.textMid, marginBottom: -1,
  },
  tabBtnActive: { borderBottomColor: "#1a1a1a", color: C.text, fontWeight: 600 },

  signGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fill, minmax(100px, 1fr))",
    gap: 6,
  },
  signBtn: {
    padding: "10px 8px 8px",
    border: `1px solid ${C.border}`,
    background: C.surface,
    display: "flex", flexDirection: "column", alignItems: "center", gap: 5,
    cursor: "pointer", transition: "border-color 0.1s",
  },
  signBtnActive: {
    border: "1px solid #1a1a1a",
    background: "#fafaf8",
  },
  signBtnDone: {
    border: "1px solid #a5d6a7",
    background: "#f1f8f1",
  },
  signBtnLabel: { fontSize: 13, fontWeight: 600, color: C.text },
  signBtnBar: { width: "100%", height: 3, background: "#e5e5e0", overflow: "hidden" },
  signBtnFill: { height: "100%", transition: "width 0.3s ease" },
  signBtnCount: { fontSize: 11, fontFamily: "'IBM Plex Mono', monospace", color: C.textDim },

  // Buttons
  btn: {
    padding: "7px 14px", border: `1px solid ${C.border}`,
    fontSize: 13, fontWeight: 500, background: C.surface, color: C.text,
  },
  btnPrimary: { background: "#1a1a1a", color: "#fff", border: "1px solid #1a1a1a" },
  btnRecord: { background: "#1a1a1a", color: "#fff", border: "1px solid #1a1a1a", padding: "7px 20px" },
  btnDanger: { background: "#b91c1c", color: "#fff", border: "1px solid #b91c1c", padding: "7px 20px" },
};