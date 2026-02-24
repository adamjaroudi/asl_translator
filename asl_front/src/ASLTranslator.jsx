import { useState, useRef, useCallback, useEffect } from "react";

const WS_STATIC   = "ws://localhost:8765";
const WS_MOTION   = "ws://localhost:8768";
const RESTART_URL = "ws://localhost:8767";
const CONFIDENCE_THRESHOLD = 0.6;

const SIGN_EMOJI = {
  "I Love You": "🤟", "Good": "👍", "Yes": "✌️", "No": "🤚",
  "More": "🤌", "Help": "🙏", "Book": "📖", "Stop": "✋",
  "Play": "🤙", "Want": "🫳", "With": "🤜", "Same": "☝️",
  "Please": "🙏", "Thank You": "🙌", "Where": "🤷",
  "How": "🤲", "Come": "👋", "Go Away": "👋", "Name": "👆",
};

const SUGGESTIONS = ["LOVE", "HELLO", "MORE", "HELP", "GOOD", "STOP", "YES", "NO", "WANT", "PLAY"];

// Context-aware completions: given the current partial sentence, suggest next words
const WORD_COMPLETIONS = {
  "HELLO": ["MY", "NAME", "HOW", "ARE", "YOU"],
  "MY":    ["NAME", "HELP", "FRIEND", "BOOK", "WANT"],
  "NAME":  ["IS", "GOOD", "YES", "FRIEND"],
  "HOW":   ["ARE", "YOU", "GOOD", "HELP"],
  "ARE":   ["YOU", "GOOD", "YES", "STOP"],
  "YOU":   ["GOOD", "WANT", "HELP", "STOP", "PLAY", "LOVE"],
  "I":     ["LOVE", "WANT", "NEED", "HELP", "GOOD"],
  "LOVE":  ["YOU", "GOOD", "MORE", "PLAY"],
  "WANT":  ["MORE", "HELP", "GOOD", "STOP", "PLAY"],
  "MORE":  ["HELP", "GOOD", "STOP", "PLAY", "WANT"],
  "HELP":  ["YOU", "ME", "GOOD", "MORE", "YES"],
  "STOP":  ["MORE", "GOOD", "YES", "NO"],
  "GOOD":  ["MORE", "STOP", "YES", "LOVE", "PLAY"],
  "YES":   ["MORE", "GOOD", "LOVE", "STOP"],
  "NO":    ["MORE", "STOP", "GOOD", "HELP"],
  "PLEASE": ["HELP", "MORE", "STOP", "GOOD"],
  "THANK":  ["YOU", "GOOD", "YES"],
  "WHERE":  ["YOU", "HELP", "GOOD", "NAME"],
};

function getContextSuggestions(sentence) {
  if (!sentence || !sentence.trim()) return SUGGESTIONS.slice(0, 8);
  const words = sentence.trim().toUpperCase().split(/\s+/);
  const lastWord = words[words.length - 1];
  // Check if last word is a partial letter sequence (all caps single chars)
  const isSpellingWord = lastWord.length > 1 && /^[A-Z]+$/.test(lastWord);
  const lookupWord = isSpellingWord ? lastWord : lastWord;
  const completions = WORD_COMPLETIONS[lookupWord];
  if (completions) return completions.slice(0, 8);
  // Fall back to default suggestions
  return SUGGESTIONS.slice(0, 8);
}

const QUICK_TIPS = [
  "Position hands clearly in frame",
  "Ensure good, even lighting",
  "Use a plain background",
  "Hold each sign for ~1 second",
  "Face the camera directly",
];

const MOTION_TIPS = [
  "Make full, deliberate motions",
  "Start and end each sign cleanly",
  "Keep hand in frame throughout",
  "Wait for the buffer bar to fill",
  "Motion gate triggers on movement",
];

function useWebSocket(url) {
  const wsRef = useRef(null);
  const onMessageRef = useRef(null);
  const [connected, setConnected] = useState(false);
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    const ws = new WebSocket(url);
    ws.onopen    = () => setConnected(true);
    ws.onclose   = () => { setConnected(false); wsRef.current = null; };
    ws.onerror   = () => ws.close();
    ws.onmessage = (e) => { try { onMessageRef.current?.(JSON.parse(e.data)); } catch {} };
    wsRef.current = ws;
  }, [url]);
  const disconnect = useCallback(() => { wsRef.current?.close(); wsRef.current = null; }, []);
  const send = useCallback((data) => {
    if (wsRef.current?.readyState === WebSocket.OPEN)
      wsRef.current.send(JSON.stringify(data));
  }, []);
  useEffect(() => () => wsRef.current?.close(), []);
  return { connected, connect, disconnect, send, onMessageRef };
}

export default function ASLTranslator({ onNavigate }) {
  const [mode, setMode] = useState("static"); // "static" | "motion"

  const [staticData, setStaticData] = useState({
    frame: null, prediction: "", confidence: 0,
    isWord: false, sentence: "", numHands: 0, stablePct: 0, fps: 0, topAlts: [],
  });

  const [motionData, setMotionData] = useState({
    frame: null, motionSign: "", motionConf: 0, motionAlts: [],
    staticSign: "", staticConf: 0, sentence: "", numHands: 0,
    bufferPct: 0, motionVal: 0, motionActive: false, stablePct: 0,
    fps: 0, modelClasses: [],
  });

  const [active, setActive]                 = useState(false);
  const [history, setHistory]               = useState([]);
  const [speaking, setSpeaking]             = useState(false);
  const [showHistory, setShowHistory]       = useState(false);
  const [showAdvanced, setShowAdvanced]     = useState(false);
  const [restarting, setRestarting]         = useState(false);
  const [staticError, setStaticError]       = useState("");
  const [motionError, setMotionError]       = useState("");
  const [confidenceThreshold, setConfidenceThreshold] = useState(CONFIDENCE_THRESHOLD);
  const [twoPlayer, setTwoPlayer]           = useState(false);
  const [tpRole, setTpRole]                 = useState("signer"); // "signer" | "reader"
  const [copied, setCopied]                 = useState(false);

  const handleRestart = useCallback(() => {
    setRestarting(true);
    const ws = new WebSocket(RESTART_URL);
    ws.onopen    = () => ws.send(JSON.stringify({ action: "restart" }));
    ws.onmessage = () => { ws.close(); setTimeout(() => setRestarting(false), 2500); };
    ws.onerror   = () => setRestarting(false);
  }, []);

  const { connected: staticConnected, connect: staticConnect, disconnect: staticDisconnect,
          send: staticSend, onMessageRef: staticMsgRef } = useWebSocket(WS_STATIC);
  const { connected: motionConnected, connect: motionConnect, disconnect: motionDisconnect,
          send: motionSend, onMessageRef: motionMsgRef } = useWebSocket(WS_MOTION);

  staticMsgRef.current = (msg) => {
    if (msg.error) { setStaticError(msg.error); setActive(false); staticDisconnect(); return; }
    setStaticData(prev => {
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
  };

  motionMsgRef.current = (msg) => {
    if (msg.error) { setMotionError(msg.error); setActive(false); motionDisconnect(); return; }
    setMotionData(prev => {
      if (msg.sentence && msg.sentence !== prev.sentence && msg.sentence.trim()) {
        setHistory(h => {
          if (h[h.length - 1] === msg.sentence.trim()) return h;
          return [...h.slice(-49), msg.sentence.trim()];
        });
      }
      return {
        frame:        msg.frame         ?? prev.frame,
        motionSign:   msg.motion_sign   ?? "",
        motionConf:   msg.motion_conf   ?? 0,
        motionAlts:   msg.motion_alts   ?? prev.motionAlts,
        staticSign:   msg.static_sign   ?? "",
        staticConf:   msg.static_conf   ?? 0,
        sentence:     msg.sentence      ?? prev.sentence,
        numHands:     msg.num_hands     ?? 0,
        bufferPct:    msg.buffer_pct    ?? 0,
        motionVal:    msg.motion_val    ?? 0,
        motionActive: msg.motion_active ?? false,
        stablePct:    msg.stable_pct    ?? 0,
        fps:          msg.fps           ?? prev.fps,
        modelClasses: msg.model_classes ?? prev.modelClasses,
      };
    });
  };

  // Use a ref for mode so callbacks always read the current value
  const modeRef = useRef(mode);
  useEffect(() => { modeRef.current = mode; }, [mode]);

  const isMotion   = mode === "motion";
  const connected  = isMotion ? motionConnected : staticConnected;

  // Always call the right WS based on current modeRef — never stale
  const connectCurrent    = useCallback(() => {
    if (modeRef.current === "motion") motionConnect(); else staticConnect();
  }, [motionConnect, staticConnect]);

  const disconnectCurrent = useCallback(() => {
    motionDisconnect();
    staticDisconnect();
  }, [motionDisconnect, staticDisconnect]);

  const sendCurrent = useCallback((data) => {
    if (modeRef.current === "motion") motionSend(data); else staticSend(data);
  }, [motionSend, staticSend]);

  const handleToggle = () => {
    if (active) {
      disconnectCurrent();
      setActive(false);
      setStaticData(d => ({ ...d, frame: null }));
      setMotionData(d => ({ ...d, frame: null }));
    } else {
      setStaticError("");
      setMotionError("");
      connectCurrent();
      setActive(true);
    }
  };

  const handleModeSwitch = (newMode) => {
    if (newMode === mode) return;
    if (active) {
      disconnectCurrent();
      setActive(false);
      setStaticData(d => ({ ...d, frame: null }));
      setMotionData(d => ({ ...d, frame: null }));
    }
    setMode(newMode);
  };

  const d         = staticData;
  const m         = motionData;
  const frame     = isMotion ? m.frame    : d.frame;
  const numHands  = isMotion ? m.numHands : d.numHands;
  const fps       = isMotion ? m.fps      : d.fps;
  const sentence  = isMotion ? m.sentence : d.sentence;

  const confPct   = Math.round(d.confidence * 100);
  const confGood  = d.confidence >= confidenceThreshold;
  const stablePct = Math.round(d.stablePct * 100);
  const signEmoji = SIGN_EMOJI[d.prediction] || "";

  const mConfPct   = Math.round(m.motionConf * 100);
  const mConfGood  = m.motionConf >= 0.65;
  const mSignEmoji = SIGN_EMOJI[m.motionSign] || "";
  const bufferPct  = Math.round(m.bufferPct * 100);
  const motionPct  = Math.min(Math.round((m.motionVal / 0.045) * 100), 100);

  const handleSpeak = () => {
    if (!sentence || speaking) return;
    const utt = new SpeechSynthesisUtterance(sentence);
    utt.onstart = () => setSpeaking(true);
    utt.onend   = () => setSpeaking(false);
    speechSynthesis.speak(utt);
  };

  const handleCopy = () => {
    if (!sentence) return;
    navigator.clipboard.writeText(sentence).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  const handleDownload = () => {
    if (!sentence) return;
    const blob = new Blob([sentence], { type: "text/plain" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "asl-translation.txt";
    a.click();
  };

  // Two-player: in reader mode just show the sentence big
  if (twoPlayer && tpRole === "reader") return (
    <div style={{ ...s.root, background:"#0f0f0f" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #0f0f0f; }
        button { font-family: 'IBM Plex Sans', sans-serif; }
      `}</style>
      <header style={{ ...s.header, background:"#1a1a1a", borderBottomColor:"#333" }}>
        <div style={{ display:"flex", alignItems:"center", gap:12 }}>
          <img src="/logo192.png" alt="NeuroSign" style={{ ...s.headerLogo }} />
          <span style={{ ...s.headerTitle, color:"#fff" }}>Reader View</span>
          <span style={{ fontSize:12, color:"#888", padding:"2px 8px", border:"1px solid #333" }}>Two-Player Mode</span>
        </div>
        <button style={{ ...s.headerNavBtn, borderColor:"#333", color:"#888" }} onClick={() => setTpRole("signer")}>
          ← Back to Signer
        </button>
      </header>
      <div style={{ flex:1, display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center", padding:40, gap:32 }}>
        <div style={{ fontSize:"clamp(28px,6vw,72px)", fontWeight:700, color:"#fff", textAlign:"center", lineHeight:1.3, maxWidth:900, wordBreak:"break-word", minHeight:120 }}>
          {sentence || <span style={{ color:"#444", fontWeight:400, fontSize:"clamp(20px,4vw,48px)" }}>Waiting for signs…</span>}
        </div>
        <div style={{ display:"flex", gap:12 }}>
          <button style={{ padding:"10px 24px", background:"#1a1a1a", border:"1px solid #333", color:"#888", fontSize:14, cursor:"pointer" }}
            onClick={() => sendCurrent({ action:"clear" })}>Clear</button>
          <button style={{ padding:"10px 24px", background:"#1a1a1a", border:"1px solid #333", color:"#888", fontSize:14, cursor:"pointer" }}
            onClick={handleSpeak} disabled={!sentence}>▶ Speak</button>
          <button style={{ padding:"10px 24px", background:"#1a1a1a", border:"1px solid #333", color:"#888", fontSize:14, cursor:"pointer" }}
            onClick={handleCopy} disabled={!sentence}>{copied ? "✓ Copied" : "Copy"}</button>
        </div>
      </div>
    </div>
  );

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

      {/* Header */}
      <header style={s.header}>
        <div style={s.headerLeft}>
          <img src="/logo192.png" alt="NeuroSign" style={s.headerLogo} />
          <span style={s.headerTitle}>NeuroSign</span>
          <div style={s.modeToggle}>
            <button
              style={{ ...s.modeBtn, ...(mode === "static" ? s.modeBtnActive : {}) }}
              onClick={() => handleModeSwitch("static")}
            >
              Static
            </button>
            <button
              style={{ ...s.modeBtn, ...(mode === "motion" ? s.modeBtnActive : {}) }}
              onClick={() => handleModeSwitch("motion")}
            >
              Motion
            </button>
          </div>
        </div>
        <nav style={s.headerNav}>
          {onNavigate && (
            <button style={s.headerNavBtn} onClick={() => onNavigate("home")}>← Home</button>
          )}
          <button style={s.headerNavBtn} onClick={() => onNavigate ? onNavigate("dictionary") : null}>Dictionary</button>
          <button style={s.headerNavBtn} onClick={() => setShowHistory(true)}>
            History{history.length > 0 ? ` (${history.length})` : ""}
          </button>
          {onNavigate && (
            <button style={{ ...s.headerNavBtn, border: "1px solid #1a1a1a", fontWeight: 600 }}
              onClick={() => onNavigate("collect")}>
              Collect Data →
            </button>
          )}
          <button
            style={{ ...s.headerNavBtn, ...(restarting ? { color: "#b45309", border: "1px solid #fcd34d" } : {}) }}
            onClick={handleRestart}
            disabled={restarting}
            title="Restart the Python server to load a newly trained model"
          >
            {restarting ? "↺ Restarting…" : "↺ Restart Server"}
          </button>
          {active && <span style={s.fpsTag}>{fps > 0 ? `${fps} fps` : "—"}</span>}
        </nav>
      </header>

      <div style={s.body}>
        <div style={s.topRow}>

          {/* Camera */}
          <div style={s.camCol}>
            <div style={s.camWrap}>
              {frame
                ? <img src={`data:image/jpeg;base64,${frame}`} alt="Camera feed" style={s.camImg} />
                : <div style={s.camBlank}>
                    <span style={s.camBlankIcon}>{(isMotion ? motionError : staticError) ? "⚠" : "◻"}</span>
                    <span style={{ ...s.camBlankText, color: (isMotion ? motionError : staticError) ? "#b91c1c" : "#666", maxWidth: 280, textAlign: "center" }}>
                      {(isMotion ? motionError : staticError) || (active ? "Connecting to camera…" : "Camera inactive")}
                    </span>
                  </div>
              }
              {active && connected && (
                <div style={s.recIndicator}>
                  <span style={{ ...s.recDot, animation: "blink 1s step-end infinite" }} />
                  {isMotion ? "Motion Mode" : "Recording"}
                </div>
              )}
              {active && (
                <div style={{
                  ...s.handTag,
                  background:  numHands > 0 ? "#e8f5e9" : "#fff8e1",
                  color:       numHands > 0 ? "#2e7d32" : "#f57f17",
                  border: numHands > 0 ? "1px solid #a5d6a7" : "1px solid #ffe082",
                }}>
                  {numHands > 0 ? `${numHands} hand${numHands > 1 ? "s" : ""}` : "No hands detected"}
                </div>
              )}
              {/* Motion bars — bottom overlay on camera */}
              {isMotion && active && (
                <div style={s.motionBars}>
                  <div style={s.motionBarRow}>
                    <span style={s.motionBarLabel}>Buffer</span>
                    <div style={s.motionBarTrack}>
                      <div style={{ ...s.motionBarFill, width: `${bufferPct}%`, background: bufferPct >= 100 ? "#2e7d32" : "#fff" }} />
                    </div>
                    <span style={s.motionBarNum}>{bufferPct}%</span>
                  </div>
                  <div style={s.motionBarRow}>
                    <span style={s.motionBarLabel}>Motion</span>
                    <div style={s.motionBarTrack}>
                      <div style={{ ...s.motionBarFill, width: `${motionPct}%`, background: m.motionActive ? "#f59e0b" : "#888" }} />
                    </div>
                    <span style={{ ...s.motionBarNum, color: m.motionActive ? "#f59e0b" : "#888" }}>
                      {m.motionActive ? "active" : "still"}
                    </span>
                  </div>
                </div>
              )}
            </div>
            <button style={{ ...s.btn, ...s.btnPrimary, marginTop: 10 }} onClick={handleToggle}>
              {active ? "Stop" : "Start / Stop"}
            </button>
          </div>

          {/* Translation panel */}
          <div style={s.translationCol}>
            {!isMotion ? (
              <>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                  <div style={s.sectionLabel}>Translation</div>
                  <label style={s.advancedToggle}>
                    <input type="checkbox" checked={showAdvanced}
                      onChange={e => setShowAdvanced(e.target.checked)}
                      style={{ marginRight: 6, accentColor: "#1a1a1a" }} />
                    Advanced
                  </label>
                </div>
                <div style={s.translationBox}>
                  <div style={s.translationText}>
                    {d.prediction
                      ? `"${d.prediction.toUpperCase()}"`
                      : <span style={{ color: "#aaa", fontWeight: 400, fontSize: 22 }}>Waiting for sign…</span>}
                  </div>
                  {signEmoji && <div style={s.translationEmoji}>{signEmoji}</div>}
                </div>
                {d.prediction && (
                  <div style={s.confRow}>
                    <div style={s.confBarWrap}>
                      <div style={{ ...s.confBar, width: `${confPct}%`, background: confGood ? "#2e7d32" : "#b45309" }} />
                    </div>
                    <span style={{ ...s.confNum, color: confGood ? "#2e7d32" : "#b45309" }}>{confPct}%</span>
                  </div>
                )}
                {showAdvanced && d.topAlts?.length > 0 && (
                  <div style={s.altsPanel}>
                    <div style={s.altsPanelTitle}>Also detecting</div>
                    {d.topAlts.map(([label, conf], i) => {
                      const pct = Math.round(conf * 100);
                      const isTop = i === 0;
                      return (
                        <div key={label} style={s.altRow}>
                          <span style={{ ...s.altLabel, fontWeight: isTop ? 600 : 400, color: isTop ? C.text : C.textMid }}>{label}</span>
                          <div style={s.altBarWrap}>
                            <div style={{ ...s.altBar, width: `${pct}%`, background: isTop ? "#1a1a1a" : "#c8c8c4" }} />
                          </div>
                          <span style={{ ...s.altPct, color: isTop ? C.text : C.textDim }}>{pct}%</span>
                        </div>
                      );
                    })}
                  </div>
                )}
              </>
            ) : (
              <>
                <div style={s.sectionLabel}>Motion Sign</div>
                <div style={s.translationBox}>
                  <div style={s.translationText}>
                    {m.motionSign
                      ? `"${m.motionSign.toUpperCase()}"`
                      : <span style={{ color: "#aaa", fontWeight: 400, fontSize: 18 }}>
                          {m.motionActive ? "Classifying…" : "Move your hand to sign"}
                        </span>
                    }
                  </div>
                  {mSignEmoji && <div style={s.translationEmoji}>{mSignEmoji}</div>}
                </div>
                {m.motionSign && (
                  <div style={s.confRow}>
                    <div style={s.confBarWrap}>
                      <div style={{ ...s.confBar, width: `${mConfPct}%`, background: mConfGood ? "#2e7d32" : "#b45309" }} />
                    </div>
                    <span style={{ ...s.confNum, color: mConfGood ? "#2e7d32" : "#b45309" }}>{mConfPct}%</span>
                  </div>
                )}
                {m.staticSign && (
                  <div style={s.altsPanel}>
                    <div style={s.altsPanelTitle}>Static reference</div>
                    <div style={s.altRow}>
                      <span style={{ ...s.altLabel, fontWeight: 600 }}>{m.staticSign}</span>
                      <div style={s.altBarWrap}>
                        <div style={{ ...s.altBar, width: `${Math.round(m.staticConf * 100)}%`, background: "#c8c8c4" }} />
                      </div>
                      <span style={{ ...s.altPct, color: C.textDim }}>{Math.round(m.staticConf * 100)}%</span>
                    </div>
                  </div>
                )}
                {m.motionAlts?.length > 0 && (
                  <div style={s.altsPanel}>
                    <div style={s.altsPanelTitle}>Alternatives</div>
                    {m.motionAlts.map(([label, conf]) => (
                      <div key={label} style={s.altRow}>
                        <span style={{ ...s.altLabel, color: C.textMid }}>{label}</span>
                        <div style={s.altBarWrap}>
                          <div style={{ ...s.altBar, width: `${Math.round(conf * 100)}%`, background: "#c8c8c4" }} />
                        </div>
                        <span style={{ ...s.altPct, color: C.textDim }}>{Math.round(conf * 100)}%</span>
                      </div>
                    ))}
                  </div>
                )}
              </>
            )}
            <button style={{ ...s.btn, ...s.btnSecondary, marginTop: "auto" }}
              onClick={handleSpeak} disabled={!sentence}>
              {speaking ? "▶  Speaking…" : "▶  Audio Playback"}
            </button>
          </div>
        </div>

        {/* Bottom three-column row */}
        <div style={s.bottomRow}>
          <div style={s.card}>
            <div style={s.sectionLabel}>Detected Sign</div>
            <div style={s.detectedRow}>
              <span style={s.detectedEmoji}>
                {isMotion ? (m.motionSign ? (mSignEmoji || "—") : "—") : (d.prediction ? (signEmoji || "—") : "—")}
              </span>
              <div style={s.detectedRight}>
                <div style={s.detectedName}>
                  {isMotion ? (m.motionSign || "Waiting for motion…") : (d.prediction || "Waiting for sign…")}
                </div>
                {((isMotion && m.motionSign) || (!isMotion && d.prediction)) && (
                  <div style={s.holdRow}>
                    <div style={s.holdTrack}>
                      <div style={{ ...s.holdFill, width: `${isMotion ? Math.round(m.stablePct * 100) : stablePct}%` }} />
                    </div>
                    <span style={s.holdNum}>{isMotion ? mConfPct : confPct}%</span>
                  </div>
                )}
              </div>
            </div>
            <div style={s.divider} />
            <div style={s.sectionLabel}>Sentence</div>
            <div style={s.sentenceArea}>
              {sentence
                ? <span style={s.sentenceText}>{sentence}</span>
                : <span style={s.sentencePlaceholder}>Your sentence will appear here…</span>
              }
            </div>
            <div style={s.sentenceBtns}>
              <button style={{ ...s.btn, ...s.btnMini }} onClick={() => sendCurrent({ action: "space" })} disabled={!connected}>Space</button>
              <button style={{ ...s.btn, ...s.btnMini }} onClick={() => sendCurrent({ action: "backspace" })} disabled={!connected}>Backspace</button>
              <button style={{ ...s.btn, ...s.btnMini, color: "#b91c1c", border: "1px solid #fca5a5" }} onClick={() => sendCurrent({ action: "clear" })} disabled={!connected}>Clear</button>
            </div>
          </div>

          <div style={s.card}>
            <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", marginBottom:2 }}>
              <div style={s.sectionLabel}>Suggestions</div>
              {sentence && <span style={{ fontSize:11, color:C.textDim }}>based on context</span>}
            </div>
            <div style={s.suggestGrid}>
              {getContextSuggestions(sentence).map(w => (
                <button key={w} style={{ ...s.btn, ...s.btnSuggest, ...(connected ? {} : { opacity:0.45 }) }}
                  title={connected ? `Insert "${w}" into sentence` : "Start camera to use suggestions"}
                  onClick={() => {
                    if (connected) {
                      sendCurrent({ action:"insert_word", word:w });
                    } else {
                      // Fallback: copy to clipboard if camera not active
                      navigator.clipboard.writeText(w).catch(()=>{});
                      setCopied(true); setTimeout(() => setCopied(false), 1500);
                    }
                  }}>
                  {w}
                </button>
              ))}
            </div>
            <p style={{ fontSize:11, color:C.textDim, marginTop:4 }}>
              {connected ? "Tap to insert word into sentence" : "Tap to copy · start camera to insert"}
            </p>
          </div>

          <div style={s.card}>
            <div style={s.sectionLabel}>{isMotion ? "Motion Tips" : "Quick Tips"}</div>
            <ol style={s.tipList}>
              {(isMotion ? MOTION_TIPS : QUICK_TIPS).map((t, i) => (
                <li key={i} style={s.tipItem}>{t}</li>
              ))}
            </ol>
            {isMotion && m.modelClasses.length > 0 && (
              <>
                <div style={s.divider} />
                <div style={s.sectionLabel}>Supported signs</div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginTop: 2 }}>
                  {m.modelClasses.map(c => (
                    <span key={c} style={s.classTag}>
                      {c.replace(/[\[\]]/g, "").replace(/_/g, " ")}
                    </span>
                  ))}
                </div>
              </>
            )}
          </div>
        </div>

        <div style={s.footerBar}>
          <button style={{ ...s.btn, ...s.btnSecondary }} onClick={() => onNavigate?.("practice")}>Practice Mode</button>
          <button style={{ ...s.btn, ...s.btnSecondary }} onClick={() => setShowHistory(true)}>History</button>
          <button style={{ ...s.btn, ...s.btnSecondary }} onClick={handleCopy} disabled={!sentence}>
            {copied ? "✓ Copied!" : "Copy"}
          </button>
          <button style={{ ...s.btn, ...s.btnSecondary }} onClick={handleDownload} disabled={!sentence}>Download</button>
          <button style={{ ...s.btn, ...(twoPlayer ? s.btnPrimary : s.btnSecondary) }}
            onClick={() => { setTwoPlayer(t => !t); if (!twoPlayer) setTpRole("signer"); }}>
            {twoPlayer ? "✓ Two-Player" : "Two-Player"}
          </button>
          {twoPlayer && (
            <button style={{ ...s.btn, ...s.btnPrimary }} onClick={() => setTpRole("reader")}>
              👀 Reader View
            </button>
          )}
          <div style={{ flex:1 }} />
          <div style={{ display:"flex", alignItems:"center", gap:8 }}>
            <span style={{ fontSize:11, color:C.textDim, fontFamily:"'IBM Plex Mono', monospace", whiteSpace:"nowrap" }}>
              Threshold: {Math.round(confidenceThreshold * 100)}%
            </span>
            <input type="range" min={40} max={95} step={5}
              value={Math.round(confidenceThreshold * 100)}
              onChange={e => setConfidenceThreshold(Number(e.target.value) / 100)}
              style={{ width:90, accentColor:C.text, cursor:"pointer" }} />
          </div>
          <button style={{ ...s.btn, ...s.btnPrimary }} onClick={() => onNavigate?.("dictionary")}>Dictionary →</button>
        </div>
      </div>

      {showHistory && (
        <div style={s.overlay} onClick={() => setShowHistory(false)}>
          <div style={s.modal} onClick={e => e.stopPropagation()}>
            <div style={s.modalHeader}>
              <span style={s.modalTitle}>Translation History</span>
              <button style={s.modalClose} onClick={() => setShowHistory(false)}>✕</button>
            </div>
            {history.length === 0
              ? <p style={{ color: "#999", padding: "12px 0", fontSize: 14 }}>No history yet.</p>
              : [...history].reverse().map((item, i) => <div key={i} style={s.historyRow}>{item}</div>)
            }
          </div>
        </div>
      )}
    </div>
  );
}

const C = {
  bg:      "#f5f5f0",
  surface: "#ffffff",
  border:  "#d4d4d0",
  text:    "#1a1a1a",
  textMid: "#555550",
  textDim: "#888880",
  primary: "#1a1a1a",
};

const s = {
  root: {
    minHeight: "100vh", background: C.bg,
    fontFamily: "'IBM Plex Sans', system-ui, sans-serif",
    fontSize: 14, color: C.text, display: "flex", flexDirection: "column",
  },
  header: {
    background: C.surface, borderBottom: `1px solid ${C.border}`,
    padding: "0 28px", height: 52, display: "flex",
    alignItems: "center", justifyContent: "space-between",
  },
  headerLeft: { display: "flex", alignItems: "center", gap: 12 },
  headerLogo: { width: 28, height: 28, borderRadius: 6, objectFit: "cover" },
  headerTitle: { fontSize: 18, fontWeight: 600, letterSpacing: "-0.01em", color: C.text },
  headerNav: { display: "flex", alignItems: "center", gap: 4 },
  headerNavBtn: {
    padding: "5px 13px", background: "none", border: `1px solid ${C.border}`,
    borderRadius: 3, fontSize: 13, fontWeight: 500, color: C.textMid, cursor: "pointer",
  },
  fpsTag: {
    marginLeft: 8, padding: "4px 10px", background: "#f0f0eb",
    border: `1px solid ${C.border}`, borderRadius: 3,
    fontSize: 12, fontFamily: "'IBM Plex Mono', monospace", color: C.textDim,
  },
  modeToggle: { display: "flex", border: `1px solid ${C.border}`, overflow: "hidden" },
  modeBtn: {
    padding: "4px 14px", fontSize: 12, fontWeight: 500, background: "none",
    border: "none", borderRight: `1px solid ${C.border}`,
    color: C.textMid, cursor: "pointer",
  },
  modeBtnActive: { background: C.primary, color: "#fff" },
  body: {
    flex: 1, display: "flex", flexDirection: "column", gap: 16,
    padding: "20px 28px 24px", maxWidth: 1200, width: "100%", margin: "0 auto",
  },
  topRow: { display: "grid", gridTemplateColumns: "1fr 320px", gap: 16, alignItems: "start" },
  camCol: { display: "flex", flexDirection: "column" },
  camWrap: {
    position: "relative", background: "#111",
    border: `1px solid ${C.border}`, aspectRatio: "16/9", overflow: "hidden",
  },
  camImg: { width: "100%", height: "100%", objectFit: "cover", display: "block" },
  camBlank: {
    height: "100%", display: "flex", flexDirection: "column",
    alignItems: "center", justifyContent: "center", gap: 10,
  },
  camBlankIcon: { fontSize: 36, color: "#444" },
  camBlankText: { color: "#666", fontSize: 13 },
  recIndicator: {
    position: "absolute", bottom: 48, left: "50%", transform: "translateX(-50%)",
    background: "rgba(0,0,0,0.72)", color: "#fff", fontSize: 13, fontWeight: 500,
    padding: "5px 14px", display: "flex", alignItems: "center", gap: 7,
  },
  recDot: { display: "inline-block", width: 8, height: 8, borderRadius: "50%", background: "#ef4444" },
  handTag: { position: "absolute", top: 10, right: 10, padding: "3px 10px", fontSize: 12, fontWeight: 500, border: "1px solid" },
  motionBars: {
    position: "absolute", bottom: 0, left: 0, right: 0,
    background: "rgba(0,0,0,0.65)", padding: "8px 14px",
    display: "flex", flexDirection: "column", gap: 5,
  },
  motionBarRow: { display: "flex", alignItems: "center", gap: 8 },
  motionBarLabel: { fontSize: 11, color: "#aaa", width: 42, fontFamily: "'IBM Plex Mono', monospace" },
  motionBarTrack: { flex: 1, height: 3, background: "rgba(255,255,255,0.15)", overflow: "hidden" },
  motionBarFill: { height: "100%", transition: "width 0.1s ease" },
  motionBarNum: { fontSize: 11, color: "#aaa", width: 40, textAlign: "right", fontFamily: "'IBM Plex Mono', monospace" },
  translationCol: {
    background: C.surface, border: `1px solid ${C.border}`,
    padding: "18px 20px", display: "flex", flexDirection: "column", gap: 12, minHeight: 260,
  },
  translationBox: {
    flex: 1, display: "flex", flexDirection: "column", alignItems: "center",
    justifyContent: "center", gap: 10, padding: "10px 0",
    borderTop: `1px solid ${C.border}`, borderBottom: `1px solid ${C.border}`,
  },
  translationText: { fontSize: 28, fontWeight: 600, letterSpacing: "-0.02em", color: C.text, textAlign: "center", lineHeight: 1.2 },
  translationEmoji: { fontSize: 52 },
  confRow: { display: "flex", alignItems: "center", gap: 10 },
  confBarWrap: { flex: 1, height: 6, background: "#e5e5e0", overflow: "hidden" },
  confBar: { height: "100%", transition: "width 0.2s ease" },
  confNum: { fontSize: 13, fontFamily: "'IBM Plex Mono', monospace", fontWeight: 500, minWidth: 36, textAlign: "right" },
  advancedToggle: { display: "flex", alignItems: "center", fontSize: 12, fontWeight: 500, color: C.textMid, cursor: "pointer", userSelect: "none" },
  altsPanel: { border: `1px solid ${C.border}`, padding: "10px 12px", background: "#fafaf8", display: "flex", flexDirection: "column", gap: 7 },
  altsPanelTitle: { fontSize: 10, fontWeight: 600, letterSpacing: "0.1em", textTransform: "uppercase", color: C.textDim, marginBottom: 2 },
  altRow: { display: "flex", alignItems: "center", gap: 8 },
  altLabel: { fontSize: 13, minWidth: 68, fontFamily: "'IBM Plex Mono', monospace" },
  altBarWrap: { flex: 1, height: 5, background: "#e5e5e0", overflow: "hidden" },
  altBar: { height: "100%", transition: "width 0.15s ease" },
  altPct: { fontSize: 12, fontFamily: "'IBM Plex Mono', monospace", minWidth: 32, textAlign: "right" },
  bottomRow: { display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16 },
  card: { background: C.surface, border: `1px solid ${C.border}`, padding: "16px 18px", display: "flex", flexDirection: "column", gap: 10 },
  sectionLabel: { fontSize: 11, fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase", color: C.textDim, marginBottom: 2 },
  detectedRow: { display: "flex", alignItems: "center", gap: 14 },
  detectedEmoji: { fontSize: 38, lineHeight: 1, minWidth: 44 },
  detectedRight: { flex: 1, display: "flex", flexDirection: "column", gap: 6 },
  detectedName: { fontSize: 18, fontWeight: 600, color: C.text },
  holdRow: { display: "flex", alignItems: "center", gap: 8 },
  holdTrack: { flex: 1, height: 5, background: "#e5e5e0", overflow: "hidden" },
  holdFill: { height: "100%", background: C.primary, transition: "width 0.1s linear" },
  holdNum: { fontSize: 12, fontFamily: "'IBM Plex Mono', monospace", color: C.textDim, minWidth: 32, textAlign: "right" },
  divider: { borderTop: `1px solid ${C.border}`, margin: "2px 0" },
  sentenceArea: { minHeight: 48, padding: "10px 12px", background: "#fafaf8", border: `1px solid ${C.border}`, fontSize: 15, fontWeight: 500, lineHeight: 1.5, color: C.text, wordBreak: "break-word" },
  sentenceText: { color: C.text },
  sentencePlaceholder: { color: C.textDim, fontWeight: 400, fontStyle: "italic" },
  sentenceBtns: { display: "flex", gap: 6 },
  suggestGrid: { display: "flex", flexWrap: "wrap", gap: 6 },
  tipList: { paddingLeft: 18, display: "flex", flexDirection: "column", gap: 7 },
  tipItem: { fontSize: 13, color: C.textMid, lineHeight: 1.4 },
  classTag: { fontSize: 11, padding: "2px 7px", border: `1px solid ${C.border}`, background: "#fafaf8", color: C.textMid, fontFamily: "'IBM Plex Mono', monospace" },
  btn: { padding: "7px 14px", border: `1px solid ${C.border}`, borderRadius: 3, fontSize: 13, fontWeight: 500, cursor: "pointer", background: C.surface, color: C.text, transition: "background 0.1s", lineHeight: 1.4 },
  btnPrimary: { background: C.primary, color: "#fff", border: `1px solid ${C.primary}` },
  btnSecondary: { background: C.surface, color: C.text, border: `1px solid ${C.border}` },
  btnMini: { padding: "5px 10px", fontSize: 12, flex: 1 },
  btnSuggest: { padding: "6px 12px", fontSize: 13, fontWeight: 600, letterSpacing: "0.04em" },
  footerBar: { display: "flex", gap: 8, alignItems: "center", paddingTop: 4 },
  overlay: { position: "fixed", inset: 0, background: "rgba(0,0,0,0.35)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 100 },
  modal: { background: C.surface, border: `1px solid ${C.border}`, padding: "24px 28px", width: "min(500px, 92vw)", maxHeight: "78vh", overflowY: "auto", boxShadow: "0 8px 32px rgba(0,0,0,0.16)" },
  modalHeader: { display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 18, paddingBottom: 14, borderBottom: `1px solid ${C.border}` },
  modalTitle: { fontSize: 16, fontWeight: 600, color: C.text },
  modalClose: { background: "none", border: `1px solid ${C.border}`, borderRadius: 3, width: 28, height: 28, cursor: "pointer", fontSize: 14, color: C.textMid },
  historyRow: { padding: "9px 0", borderBottom: `1px solid #f0f0eb`, fontSize: 14, color: C.textMid },
};