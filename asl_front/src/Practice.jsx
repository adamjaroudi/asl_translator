import { useState, useEffect, useRef, useCallback } from "react";

const C = {
  bg:      "#f5f5f0",
  surface: "#ffffff",
  border:  "#d4d4d0",
  text:    "#1a1a1a",
  textMid: "#555550",
  textDim: "#888880",
};

const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
const WORD_SIGNS = [
  { label: "I Love You", key: "I Love You" },
  { label: "Good",       key: "Good"       },
  { label: "Yes",        key: "Yes"        },
  { label: "No",         key: "No"         },
  { label: "More",       key: "More"       },
  { label: "Help",       key: "Help"       },
  { label: "Book",       key: "Book"       },
  { label: "Stop",       key: "Stop"       },
  { label: "Play",       key: "Play"       },
  { label: "Want",       key: "Want"       },
  { label: "With",       key: "With"       },
  { label: "Same",       key: "Same"       },
];
const PRACTICE_IMAGE_BASE   = "/practice";
const TEST_QUESTION_COUNT   = 10;
const QUESTION_TIME_SECONDS = 12;
const WS_SIGN_IT            = "ws://localhost:8765";

const SR_INTERVALS = [1, 2, 4, 8];
const SR_KEY       = "asl_sr_data";

const CIRCLE_SIZE   = 120;
const CIRCLE_STROKE = 8;
const CIRCLE_R      = (CIRCLE_SIZE - CIRCLE_STROKE) / 2;
const CIRCLE_C      = 2 * Math.PI * CIRCLE_R;

const HINTS = {
  A:"Fist, thumb to side", B:"Flat hand, fingers up", C:"Curved C shape",
  D:"Point up, others curve", E:"Fingers curled in", F:"OK sign shape",
  G:"Point sideways", H:"Two fingers sideways", I:"Pinky up",
  J:"Pinky traces J", K:"V with thumb out", L:"L shape",
  M:"Three fingers on thumb", N:"Two fingers on thumb", O:"O shape",
  P:"K pointing down", Q:"G pointing down", R:"Fingers crossed",
  S:"Fist, thumb in front", T:"Thumb between fingers", U:"Two fingers up",
  V:"V shape", W:"Three fingers up", X:"Index hooked",
  Y:"Thumb & pinky out", Z:"Index traces Z",
};

function loadSRData() {
  try { return JSON.parse(localStorage.getItem(SR_KEY) || "{}"); }
  catch { return {}; }
}
function saveSRData(data) {
  try { localStorage.setItem(SR_KEY, JSON.stringify(data)); } catch {}
}
function getSRQueue(srData) {
  return [...LETTERS].sort((a, b) => {
    const da = srData[a] || { interval: 0, streak: 0 };
    const db = srData[b] || { interval: 0, streak: 0 };
    // Primary: lower interval = higher priority (unseen/weak first)
    if (da.interval !== db.interval) return da.interval - db.interval;
    // Secondary: lower streak = higher priority (recently failed first)
    return (da.streak || 0) - (db.streak || 0);
  });
}
function updateSR(srData, letter, wasCorrect) {
  const prev   = srData[letter] || { correct: 0, interval: 0, streak: 0 };
  const streak = wasCorrect ? (prev.streak || 0) + 1 : 0;
  const iLevel = wasCorrect ? Math.min((prev.interval || 0) + 1, SR_INTERVALS.length - 1) : 0;
  return { ...srData, [letter]: { correct: wasCorrect ? prev.correct + 1 : prev.correct, interval: iLevel, streak } };
}

function shuffle(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}
function fmt(s) {
  const m = Math.floor(s / 60), sec = s % 60;
  return `${m}:${sec.toString().padStart(2,"0")}`;
}

function LetterImage({ letter, style: extraStyle = {} }) {
  const [phase, setPhase] = useState(0);
  useEffect(() => { setPhase(0); }, [letter]);
  const srcs = [`${PRACTICE_IMAGE_BASE}/${letter}.PNG`, `${PRACTICE_IMAGE_BASE}/${letter}.png`];
  if (phase >= 2) return (
    <div style={{ ...imgStyle.placeholder, ...extraStyle }}>
      <span style={imgStyle.placeholderLetter}>{letter}</span>
    </div>
  );
  return <img key={`${letter}-${phase}`} src={srcs[phase]} alt={`ASL ${letter}`}
    style={{ ...imgStyle.img, ...extraStyle }} onError={() => setPhase(p => p + 1)} />;
}
const imgStyle = {
  img:              { width:"100%", maxWidth:200, height:160, objectFit:"contain" },
  placeholder:      { width:"100%", maxWidth:200, height:160, display:"flex", alignItems:"center", justifyContent:"center", border:`2px dashed ${C.border}` },
  placeholderLetter:{ fontSize:56, fontWeight:700, color:C.textDim },
};

function CircleTimer({ value, max, urgent }) {
  const pct = value / max;
  return (
    <div style={{ position:"relative", display:"inline-flex" }}>
      <svg style={{ transform:"rotate(-90deg)" }} width={CIRCLE_SIZE} height={CIRCLE_SIZE}
        viewBox={`0 0 ${CIRCLE_SIZE} ${CIRCLE_SIZE}`}>
        <circle cx={CIRCLE_SIZE/2} cy={CIRCLE_SIZE/2} r={CIRCLE_R}
          fill="none" stroke={C.border} strokeWidth={CIRCLE_STROKE} />
        <circle cx={CIRCLE_SIZE/2} cy={CIRCLE_SIZE/2} r={CIRCLE_R}
          fill="none" stroke={urgent ? "#b91c1c" : C.text} strokeWidth={CIRCLE_STROKE}
          strokeLinecap="round" strokeDasharray={CIRCLE_C}
          strokeDashoffset={CIRCLE_C * (1 - pct)}
          style={{ transition:"stroke-dashoffset 0.3s, stroke 0.3s" }} />
      </svg>
      <span style={{ position:"absolute", left:"50%", top:"50%", transform:"translate(-50%,-50%)",
        fontSize:28, fontWeight:700, fontFamily:"'IBM Plex Mono', monospace", color:C.text }}>{value}</span>
    </div>
  );
}

function useSignItWS() {
  const wsRef    = useRef(null);
  const onMsgRef = useRef(null);
  const [connected, setConnected] = useState(false);
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    const ws = new WebSocket(WS_SIGN_IT);
    ws.onopen    = () => setConnected(true);
    ws.onclose   = () => { setConnected(false); wsRef.current = null; };
    ws.onerror   = () => ws.close();
    ws.onmessage = (e) => { try { onMsgRef.current?.(JSON.parse(e.data)); } catch {} };
    wsRef.current = ws;
  }, []);
  const disconnect = useCallback(() => { wsRef.current?.close(); wsRef.current = null; }, []);
  useEffect(() => () => wsRef.current?.close(), []);
  return { connected, connect, disconnect, onMsgRef };
}

function Shell({ title, right, onBack, children }) {
  return (
    <div style={s.root}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: ${C.bg}; }
        button { font-family: 'IBM Plex Sans', sans-serif; cursor: pointer; }
        button:disabled { opacity: 0.35; cursor: default; }
        @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }
      `}</style>
      <header style={s.header}>
        <div style={{ display:"flex", alignItems:"center", gap:12 }}>
          <img src="/logo192.png" alt="NeuroSign" style={s.logoMark} />
          <span style={s.logoText}>NeuroSign</span>
          {title && <><span style={s.sep}>·</span><span style={s.pageTitle}>{title}</span></>}
        </div>
        <div style={{ display:"flex", alignItems:"center", gap:10 }}>
          {right}
          {onBack && <button style={s.backBtn} onClick={onBack}>← Back</button>}
        </div>
      </header>
      {children}
    </div>
  );
}

function SRMeter({ srData }) {
  const mastered = LETTERS.filter(l => (srData[l]?.interval || 0) >= SR_INTERVALS.length - 1).length;
  const learning = LETTERS.filter(l => (srData[l]?.interval || 0) > 0 && (srData[l]?.interval || 0) < SR_INTERVALS.length - 1).length;
  const pct      = Math.round((mastered / 26) * 100);
  return (
    <div style={{ display:"flex", alignItems:"center", gap:10 }}>
      <div style={{ fontSize:12, color:C.textDim, fontFamily:"'IBM Plex Mono', monospace", whiteSpace:"nowrap" }}>
        {mastered}/26 mastered
      </div>
      <div style={{ width:120, height:6, background:C.border, overflow:"hidden" }}>
        <div style={{ height:"100%", display:"flex" }}>
          <div style={{ width:`${pct}%`, background:"#2e7d32", transition:"width 0.4s" }} />
          <div style={{ width:`${Math.round((learning/26)*100)}%`, background:"#f59e0b" }} />
        </div>
      </div>
      <div style={{ fontSize:11, color:C.textDim }}>{pct}%</div>
    </div>
  );
}

export default function Practice({ onNavigate }) {
  const [mode, setMode] = useState(null);
  const [srData, setSrData] = useState(loadSRData);

  const [fcOrder, setFcOrder]       = useState(() => getSRQueue(loadSRData()));
  const [fcIdx, setFcIdx]           = useState(0);
  const [flipped, setFlipped]       = useState(false);
  const [fcFeedback, setFcFeedback] = useState(null);

  const [tOrder, setTOrder]         = useState([]);
  const [tIdx, setTIdx]             = useState(0);
  const [tCorrect, setTCorrect]     = useState(0);
  const [tInput, setTInput]         = useState("");
  const [tAnswered, setTAnswered]   = useState(false);
  const [tIsRight, setTIsRight]     = useState(false);
  const [tDone, setTDone]           = useState(false);
  const [tElapsed, setTElapsed]     = useState(0);
  const [tCountdown, setTCountdown] = useState(QUESTION_TIME_SECONDS);
  const tStartRef = useRef(null);
  const tNextRef  = useRef(null);
  const inputRef  = useRef(null);

  const [siScope, setSiScope]         = useState("letters");
  const [siOrder, setSiOrder]         = useState([]);
  const [siIdx, setSiIdx]             = useState(0);
  const [siCorrect, setSiCorrect]     = useState(0);
  const [siAnswered, setSiAnswered]   = useState(false);
  const [siDone, setSiDone]           = useState(false);
  const [siElapsed, setSiElapsed]     = useState(0);
  const [siCountdown, setSiCountdown] = useState(QUESTION_TIME_SECONDS);
  const [camFrame, setCamFrame]       = useState(null);
  const [camPred, setCamPred]         = useState("");
  const [camConf, setCamConf]         = useState(0);
  const [camHands, setCamHands]       = useState(0);
  const [camOn, setCamOn]             = useState(false);
  const siStartRef = useRef(null);
  const siNextRef  = useRef(null);

  const { connect: wsConnect, disconnect: wsDisc, onMsgRef } = useSignItWS();
  onMsgRef.current = (msg) => {
    if (msg.frame !== undefined)      setCamFrame(msg.frame);
    if (msg.prediction !== undefined) setCamPred(msg.prediction);
    if (msg.confidence !== undefined) setCamConf(msg.confidence);
    if (msg.num_hands  !== undefined) setCamHands(msg.num_hands);
  };

  useEffect(() => { saveSRData(srData); }, [srData]);

  const fcLetter   = fcOrder[fcIdx];
  const fcSrLevel  = srData[fcLetter]?.interval || 0;
  const srLabels   = ["New","Learning","Review","Good","Mastered"];
  const srColors   = ["#888880","#f59e0b","#3b82f6","#2e7d32","#166534"];

  const fcAnswer = (correct) => {
    setFcFeedback(correct ? "correct" : "wrong");
    setSrData(prev => {
      const next = updateSR(prev, fcLetter, correct);
      saveSRData(next);
      return next;
    });
    setTimeout(() => {
      setFcFeedback(null); setFlipped(false);
      if (fcIdx + 1 >= fcOrder.length) {
        setSrData(prev => { setFcOrder(getSRQueue(prev)); return prev; });
        setFcIdx(0);
      } else { setFcIdx(i => i + 1); }
    }, 700);
  };

  const startTest = () => {
    const weighted = [];
    LETTERS.forEach(l => {
      const d = srData[l];
      const interval = d?.interval ?? 0;
      const streak   = d?.streak   ?? 0;
      // Base reps from interval (lower = more reps)
      let reps = Math.max(1, SR_INTERVALS.length - interval);
      // Bonus reps if recently failed (streak === 0 but has been seen)
      if (d && streak === 0 && interval === 0) reps += 2;
      for (let i = 0; i < reps; i++) weighted.push(l);
    });
    setTOrder(shuffle(weighted).slice(0, TEST_QUESTION_COUNT));
    setTIdx(0); setTCorrect(0); setTInput(""); setTAnswered(false);
    setTIsRight(false); setTDone(false); setTElapsed(0); setTCountdown(QUESTION_TIME_SECONDS);
    tStartRef.current = Date.now();
  };
  useEffect(() => {
    if (mode !== "test" || tDone || !tOrder.length) return;
    const id = setInterval(() => setTElapsed(Math.floor((Date.now() - tStartRef.current) / 1000)), 1000);
    return () => clearInterval(id);
  }, [mode, tDone, tOrder.length]);
  useEffect(() => {
    if (mode !== "test" || tDone || !tOrder.length || tAnswered) return;
    const id = setInterval(() => setTCountdown(c => {
      if (c <= 1) { setTAnswered(true); setTIsRight(false); setTimeout(() => tNextRef.current?.(), 1500); return 0; }
      return c - 1;
    }), 1000);
    return () => clearInterval(id);
  }, [mode, tDone, tOrder.length, tAnswered, tIdx]);
  useEffect(() => {
    if (mode === "test" && !tAnswered) setTimeout(() => inputRef.current?.focus(), 50);
  }, [mode, tIdx, tAnswered]);
  const tNext = () => {
    if (tIdx + 1 >= tOrder.length) { setTElapsed(Math.floor((Date.now() - tStartRef.current) / 1000)); setTDone(true); }
    else { setTIdx(i => i + 1); setTAnswered(false); setTIsRight(false); setTInput(""); setTCountdown(QUESTION_TIME_SECONDS); }
  };
  tNextRef.current = tNext;
  const tQuestion = tOrder[tIdx];

  const startSignIt = (scope) => {
    const pool = scope === "letters"
      ? shuffle([...LETTERS]).slice(0, TEST_QUESTION_COUNT)
      : shuffle(WORD_SIGNS.map(w => w.key)).slice(0, Math.min(TEST_QUESTION_COUNT, WORD_SIGNS.length));
    setSiOrder(pool); setSiIdx(0); setSiCorrect(0); setSiAnswered(false); setSiDone(false);
    setSiElapsed(0); setSiCountdown(QUESTION_TIME_SECONDS);
    setCamPred(""); setCamFrame(null); siStartRef.current = Date.now();
  };
  useEffect(() => {
    if (mode !== "signit" || siDone || !siOrder.length || !camOn) return;
    const id = setInterval(() => setSiElapsed(Math.floor((Date.now() - siStartRef.current) / 1000)), 1000);
    return () => clearInterval(id);
  }, [mode, siDone, siOrder.length, camOn]);
  useEffect(() => {
    if (mode !== "signit" || siDone || !siOrder.length || siAnswered || !camOn) return;
    const id = setInterval(() => setSiCountdown(c => {
      if (c <= 1) { setSiAnswered(true); setTimeout(() => siNextRef.current?.(), 1500); return 0; }
      return c - 1;
    }), 1000);
    return () => clearInterval(id);
  }, [mode, siDone, siOrder.length, siAnswered, siIdx, camOn]);
  useEffect(() => {
    if (mode !== "signit" || siAnswered || !siOrder[siIdx] || !camOn) return;
    const target = siOrder[siIdx];
    const match  = camPred === target && camConf >= 0.75;
    if (match) {
      setSiAnswered(true); setSiCorrect(c => c + 1);
      setSrData(prev => updateSR(prev, target, true));
      setTimeout(() => siNextRef.current?.(), 1400);
    }
  }, [camPred, camConf, siAnswered, mode, camOn, siIdx, siOrder, siScope]);
  const siNext = () => {
    if (siIdx + 1 >= siOrder.length) { setSiElapsed(Math.floor((Date.now() - siStartRef.current) / 1000)); setSiDone(true); }
    else { setSiIdx(i => i + 1); setSiAnswered(false); setSiCountdown(QUESTION_TIME_SECONDS); setCamPred(""); }
  };
  siNextRef.current = siNext;
  useEffect(() => {
    if (mode !== "signit" && camOn) { wsDisc(); setCamOn(false); setCamFrame(null); }
  }, [mode]);
  const toggleCam = () => {
    if (camOn) { wsDisc(); setCamOn(false); setCamFrame(null); }
    else { wsConnect(); setCamOn(true); if (!siStartRef.current) siStartRef.current = Date.now(); }
  };
  const siQuestion = siOrder[siIdx];
  const siIsRight = camPred === siQuestion && camConf >= 0.75;

  const goBack = () => {
    if (mode === null) { onNavigate?.("home"); return; }
    if (camOn) { wsDisc(); setCamOn(false); setCamFrame(null); }
    setMode(null); setFlipped(false); setFcIdx(0); setFcFeedback(null);
  };

  // ── Menu ────────────────────────────────────────────────────────────────────
  if (mode === null) return (
    <Shell onBack={goBack} right={<SRMeter srData={srData} />}>
      <main style={s.menuMain}>
        <div style={s.menuHero}>
          <h1 style={s.menuTitle}>Practice</h1>
          <p style={s.menuSub}>Learn and drill ASL signs three ways</p>
        </div>
        <div style={s.menuCards}>
          <ModeCard icon="🃏" label="Flashcards" num="01"
            desc="Spaced repetition — letters you struggle with appear more often. Rate each card to track your progress."
            cta="Start flashcards →"
            onClick={() => { setFcOrder(getSRQueue(srData)); setFcIdx(0); setFlipped(false); setFcFeedback(null); setMode("flashcards"); }} />
          <ModeCard icon="✏️" label="Type It" num="02"
            desc="See the sign image with a countdown timer. Type the letter — questions are weighted toward your weak spots."
            cta="Start quiz →"
            onClick={() => { startTest(); setMode("test"); }} />
          <ModeCard icon="🤚" label="Sign It" num="03"
            desc="Sign the letter you see — the model detects your hand and moves on automatically."
            cta="Start signing →"
            onClick={() => { setSiScope("letters"); setMode("signit"); startSignIt("letters"); }} />
        </div>
        <ProgressBreakdown srData={srData} />
        <div style={{ textAlign:"center", marginTop:8 }}>
          <button style={{ ...s.outlineBtn, fontSize:12, color:C.textDim }}
            onClick={() => { const reset = {}; setSrData(reset); saveSRData(reset); setFcOrder(shuffle([...LETTERS])); }}>
            Reset spaced repetition progress
          </button>
        </div>
      </main>
    </Shell>
  );

  // ── Flashcards ──────────────────────────────────────────────────────────────
  if (mode === "flashcards") return (
    <Shell title="Flashcards" onBack={goBack} right={<SRMeter srData={srData} />}>
      <main style={s.center}>
        <div style={s.fcCard} onClick={() => !fcFeedback && setFlipped(f => !f)} tabIndex={0}
          onKeyDown={e => e.key === "Enter" && !fcFeedback && setFlipped(f => !f)}
          role="button" aria-label="Flip card">
          <div style={{ ...s.fcInner, transform: flipped ? "rotateY(180deg)" : "none",
            outline: fcFeedback === "correct" ? "2px solid #2e7d32" : fcFeedback === "wrong" ? "2px solid #b91c1c" : "none" }}>
            <div style={s.fcFront}>
              <LetterImage letter={fcLetter} />
              <p style={s.fcPrompt}>What letter? · tap to flip</p>
              <div style={{ display:"flex", alignItems:"center", gap:6, marginTop:6 }}>
                <span style={{ fontSize:11, fontWeight:600, color:srColors[Math.min(fcSrLevel,4)], padding:"2px 8px", border:`1px solid ${srColors[Math.min(fcSrLevel,4)]}`, opacity:0.8 }}>
                  {srLabels[Math.min(fcSrLevel,4)]}
                </span>
                {(srData[fcLetter]?.streak || 0) > 1 && (
                  <span style={{ fontSize:11, color:C.textDim }}>🔥 {srData[fcLetter].streak}</span>
                )}
              </div>
            </div>
            <div style={{ ...s.fcFront, ...s.fcBack }}>
              <span style={s.fcLetter}>{fcLetter}</span>
              <span style={s.fcHint}>{HINTS[fcLetter]}</span>
            </div>
          </div>
        </div>
        {flipped && !fcFeedback ? (
          <div style={{ display:"flex", gap:12, marginTop:20 }}>
            <button style={{ ...s.navBtn, background:"#fef2f2", borderColor:"#fca5a5", color:"#b91c1c", padding:"10px 28px", fontWeight:600 }}
              onClick={() => fcAnswer(false)}>✗ Missed it</button>
            <button style={{ ...s.navBtn, background:"#f0fdf4", borderColor:"#a5d6a7", color:"#166534", padding:"10px 28px", fontWeight:600 }}
              onClick={() => fcAnswer(true)}>✓ Got it</button>
          </div>
        ) : !fcFeedback ? (
          <div style={s.navRow}>
            <button style={s.navBtn} disabled={fcIdx === 0}
              onClick={e => { e.stopPropagation(); setFcIdx(i => Math.max(0,i-1)); setFlipped(false); }}>← Prev</button>
            <span style={s.counter}>{fcIdx + 1} / 26</span>
            <button style={s.navBtn}
              onClick={e => { e.stopPropagation(); setFcIdx(i => Math.min(25,i+1)); setFlipped(false); }}>Next →</button>
          </div>
        ) : null}
      </main>
    </Shell>
  );

  // ── Test results ─────────────────────────────────────────────────────────────
  if (mode === "test" && tDone) {
    const pct = Math.round((tCorrect / tOrder.length) * 100);
    const weakLetters = LETTERS
      .filter(l => (srData[l]?.interval ?? 0) === 0 && srData[l])
      .slice(0, 6);
    return (
      <Shell title="Type It" onBack={goBack}>
        <main style={s.center}>
          <div style={s.resultBox}>
            <div style={s.resultScore}>{tCorrect} / {tOrder.length}</div>
            <div style={s.resultPct}>{pct}% correct · {fmt(tElapsed)}</div>
            {weakLetters.length > 0 && (
              <div style={{ marginTop:4, padding:"10px 18px", border:`1px solid #fcd34d`, background:"#fffbeb", fontSize:13, color:"#92400e", textAlign:"center" }}>
                Still struggling with: {weakLetters.map((l,i) => (
                  <span key={l} style={{ fontWeight:700, fontFamily:"'IBM Plex Mono', monospace" }}>
                    {l}{i < weakLetters.length-1 ? ", " : ""}
                  </span>
                ))} — try Flashcards for these
              </div>
            )}
            <button style={s.pill} onClick={() => { startTest(); setTDone(false); }}>Try again</button>
            <button style={s.outlineBtn} onClick={goBack}>Back to Practice</button>
          </div>
        </main>
      </Shell>
    );
  }

  // ── Test question ────────────────────────────────────────────────────────────
  if (mode === "test" && tQuestion) return (
    <Shell title="Type It" onBack={goBack}
      right={<span style={s.counter}>{tIdx+1} / {TEST_QUESTION_COUNT} · {fmt(tElapsed)}</span>}>
      <main style={s.center}>
        <div style={s.qBox}>
          <CircleTimer value={tCountdown} max={QUESTION_TIME_SECONDS} urgent={tCountdown <= 3} />
          <LetterImage letter={tQuestion} style={{ maxWidth:200, height:160 }} />
          <p style={s.qText}>Which letter is this sign?</p>
          <input ref={inputRef}
            style={{ ...s.letterInput,
              border: tAnswered ? `2px solid ${tIsRight ? "#2e7d32" : "#b91c1c"}` : `2px solid ${C.border}`,
              background: tAnswered ? (tIsRight ? "#f0fdf4" : "#fef2f2") : C.surface,
              color: tAnswered ? (tIsRight ? "#166534" : "#b91c1c") : C.text,
            }}
            type="text" maxLength={1} value={tInput} autoComplete="off"
            disabled={tAnswered} placeholder="A–Z"
            onChange={e => {
              const v = e.target.value.toUpperCase().replace(/[^A-Z]/,"");
              setTInput(v);
              if (v.length === 1 && !tAnswered) {
                const right = v === tQuestion;
                setTAnswered(true); setTIsRight(right);
                if (right) setTCorrect(c => c + 1);
                setSrData(prev => updateSR(prev, tQuestion, right));
                setTimeout(() => tNextRef.current?.(), 1400);
              }
            }}
          />
          {tAnswered && (
            <p style={{ ...s.feedback, color: tIsRight ? "#166534" : "#b91c1c" }}>
              {tIsRight ? "✓ Correct!" : `✗ It was ${tQuestion}`}
            </p>
          )}
        </div>
      </main>
    </Shell>
  );

  // ── Sign-It results ──────────────────────────────────────────────────────────
  if (mode === "signit" && siDone) {
    const pct = Math.round((siCorrect / siOrder.length) * 100);
    return (
      <Shell title="Sign It" onBack={goBack}>
        <main style={s.center}>
          <div style={s.resultBox}>
            <div style={s.resultScore}>{siCorrect} / {siOrder.length}</div>
            <div style={s.resultPct}>{pct}% correct · {fmt(siElapsed)}</div>
            <button style={s.pill} onClick={() => { setSiOrder([]); setCamPred(""); startSignIt("letters"); }}>Try again</button>
            <button style={s.outlineBtn} onClick={goBack}>Back to Practice</button>
          </div>
        </main>
      </Shell>
    );
  }

  // ── Sign-It question ─────────────────────────────────────────────────────────
  if (mode === "signit" && siQuestion) return (
    <Shell title="Sign It — Letters" onBack={goBack}
      right={<span style={s.counter}>{siIdx+1} / {siOrder.length} · {fmt(siElapsed)}</span>}>
      <main style={{ ...s.center, paddingTop:32 }}>
        <div style={s.siLayout}>
          <div style={s.siLeft}>
            <p style={s.sectionLabel}>Sign this letter</p>
            <>
              <div style={s.siBigLetter}>{siQuestion}</div>
              <p style={s.siHint}>{HINTS[siQuestion] || ""}</p>
            </>
            {camOn && (
              <div style={{ marginTop:20 }}>
                <CircleTimer value={siCountdown} max={QUESTION_TIME_SECONDS} urgent={siCountdown <= 3} />
              </div>
            )}
            {siAnswered && (
              <div style={{ marginTop:16, padding:"10px 18px",
                background: siIsRight ? "#f0fdf4" : "#fef2f2",
                border: `1px solid ${siIsRight ? "#a5d6a7" : "#fca5a5"}`,
                fontSize:15, fontWeight:600, color: siIsRight ? "#166534" : "#b91c1c" }}>
                {siIsRight ? "✓ Correct!" : `Time's up — it was ${siQuestion}`}
              </div>
            )}
          </div>
          <div style={s.siRight}>
            <p style={s.sectionLabel}>Your camera</p>
            <div style={s.camWrap}>
              {camFrame
                ? <img src={`data:image/jpeg;base64,${camFrame}`} alt="cam" style={s.camImg} />
                : <div style={s.camBlank}>
                    <span style={s.camBlankIcon}>◻</span>
                    <span style={s.camBlankText}>{camOn ? "Connecting…" : "Camera off"}</span>
                  </div>
              }
              {camOn && (
                <div style={{ ...s.handTag, background: camHands > 0 ? "#e8f5e9" : "#fff8e1",
                  color: camHands > 0 ? "#2e7d32" : "#f57f17",
                  border: `1px solid ${camHands > 0 ? "#a5d6a7" : "#ffe082"}` }}>
                  {camHands > 0 ? `${camHands} hand${camHands > 1 ? "s" : ""}` : "No hands"}
                </div>
              )}
              {camOn && camPred && (
                <div style={{ ...s.camPred, background: siIsRight ? "rgba(21,128,61,0.85)" : "rgba(0,0,0,0.72)" }}>
                  {camPred} · {Math.round(camConf * 100)}%
                </div>
              )}
            </div>
            <button style={{ ...s.navBtn, marginTop:8, width:"100%", padding:"9px 0" }} onClick={toggleCam}>
              {camOn ? "Stop Camera" : "Start Camera"}
            </button>
            {!camOn && <p style={s.camNote}>Start camera to begin signing</p>}
          </div>
        </div>
      </main>
    </Shell>
  );

  return null;
}

function ProgressBreakdown({ srData }) {
  const srLabels = ["New","Learning","Review","Good","Mastered"];
  const srColors = ["#d1d5db","#f59e0b","#3b82f6","#2e7d32","#166534"];
  const srBg     = ["#f9fafb","#fffbeb","#eff6ff","#f0fdf4","#dcfce7"];

  // Group letters by SR level
  const groups = [0,1,2,3,4].map(level => ({
    level, label: srLabels[level], color: srColors[level], bg: srBg[level],
    letters: LETTERS.filter(l => (srData[l]?.interval ?? 0) === level),
  }));

  const anyData = LETTERS.some(l => srData[l]);
  if (!anyData) return (
    <div style={{ textAlign:"center", margin:"32px 0", padding:"24px", border:`1px dashed ${C.border}`, color:C.textDim, fontSize:13 }}>
      No progress recorded yet — start with Flashcards or Type It to build your history.
    </div>
  );

  return (
    <div style={{ marginTop:32, border:`1px solid ${C.border}`, background:C.surface }}>
      <div style={{ padding:"14px 20px", borderBottom:`1px solid ${C.border}`, display:"flex", alignItems:"center", justifyContent:"space-between" }}>
        <span style={{ fontSize:13, fontWeight:600, color:C.text }}>Your Progress</span>
        <span style={{ fontSize:11, color:C.textDim, fontFamily:"'IBM Plex Mono', monospace" }}>
          {LETTERS.filter(l => (srData[l]?.interval ?? 0) >= 4).length}/26 mastered
        </span>
      </div>
      <div style={{ display:"grid", gridTemplateColumns:"repeat(5,1fr)", gap:0 }}>
        {groups.map(({ level, label, color, bg, letters }) => (
          <div key={level} style={{ padding:"14px 16px", borderRight: level < 4 ? `1px solid ${C.border}` : "none", background: bg }}>
            <div style={{ fontSize:11, fontWeight:600, color, letterSpacing:"0.05em", marginBottom:10 }}>
              {label} ({letters.length})
            </div>
            <div style={{ display:"flex", flexWrap:"wrap", gap:4 }}>
              {letters.length === 0
                ? <span style={{ fontSize:11, color:C.textDim, fontStyle:"italic" }}>none</span>
                : letters.map(l => (
                    <span key={l} style={{
                      fontSize:12, fontWeight:700, fontFamily:"'IBM Plex Mono', monospace",
                      padding:"2px 6px", background:C.surface, border:`1px solid ${color}`, color,
                    }}>{l}</span>
                  ))
              }
            </div>
            {level === 0 && letters.length > 0 && (
              <div style={{ marginTop:8, fontSize:10, color:"#b45309" }}>← practice these first</div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function ModeCard({ icon, label, num, desc, cta, onClick }) {
  const [h, setH] = useState(false);
  return (
    <button onClick={onClick} onMouseEnter={() => setH(true)} onMouseLeave={() => setH(false)}
      style={{ ...s.modeCard, background: h ? "#fafaf8" : C.surface }}>
      <div style={s.modeNum}>{num}</div>
      <div style={s.modeIcon}>{icon}</div>
      <div style={s.modeLabel}>{label}</div>
      <div style={s.modeDesc}>{desc}</div>
      <div style={s.modeCta}>{cta}</div>
    </button>
  );
}

const s = {
  root: { minHeight:"100vh", background:C.bg, fontFamily:"'IBM Plex Sans', system-ui, sans-serif", color:C.text, display:"flex", flexDirection:"column" },
  header: { background:C.surface, borderBottom:`1px solid ${C.border}`, padding:"0 28px", height:52, display:"flex", alignItems:"center", justifyContent:"space-between", gap:16 },
  logoMark: { width:28, height:28, borderRadius:6, objectFit:"cover", flexShrink:0 },
  logoText: { fontSize:15, fontWeight:700, letterSpacing:"-0.02em", color:C.text },
  sep: { fontSize:16, color:C.textDim, margin:"0 6px" },
  pageTitle: { fontSize:14, fontWeight:500, color:C.textMid },
  backBtn: { padding:"6px 14px", background:"none", border:`1px solid ${C.border}`, fontSize:13, fontWeight:500, color:C.textMid, cursor:"pointer" },
  counter: { fontSize:13, color:C.textDim, fontFamily:"'IBM Plex Mono', monospace" },
  sectionLabel: { fontSize:11, fontWeight:600, letterSpacing:"0.08em", textTransform:"uppercase", color:C.textDim, marginBottom:8 },
  menuMain: { flex:1, maxWidth:960, width:"100%", margin:"0 auto", padding:"56px 32px 64px" },
  menuHero: { textAlign:"center", marginBottom:52 },
  menuTitle: { fontSize:"clamp(28px, 5vw, 42px)", fontWeight:700, letterSpacing:"-0.03em", color:C.text, marginBottom:12 },
  menuSub: { fontSize:16, color:C.textMid },
  menuCards: { display:"grid", gridTemplateColumns:"repeat(3, 1fr)", gap:1, border:`1px solid ${C.border}`, background:C.border },
  modeCard: { padding:"32px 28px", border:"none", textAlign:"left", cursor:"pointer", display:"flex", flexDirection:"column", gap:8, transition:"background 0.1s" },
  modeNum: { fontSize:11, fontWeight:600, letterSpacing:"0.1em", color:C.textDim, fontFamily:"'IBM Plex Mono', monospace" },
  modeIcon: { fontSize:32, marginTop:4, marginBottom:4 },
  modeLabel: { fontSize:18, fontWeight:700, letterSpacing:"-0.01em", color:C.text },
  modeDesc: { fontSize:13, color:C.textMid, lineHeight:1.55, flex:1 },
  modeCta: { fontSize:12, fontWeight:600, color:C.text, marginTop:8 },
  center: { flex:1, display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center", padding:"32px 24px" },
  pill: { padding:"11px 28px", background:C.text, color:"#fff", border:"none", fontSize:14, fontWeight:600, cursor:"pointer" },
  outlineBtn: { padding:"9px 20px", background:"none", border:`1px solid ${C.border}`, fontSize:13, fontWeight:500, color:C.text, cursor:"pointer" },
  navBtn: { padding:"8px 18px", background:C.surface, border:`1px solid ${C.border}`, fontSize:13, fontWeight:500, color:C.text, cursor:"pointer" },
  navRow: { display:"flex", alignItems:"center", gap:16, marginTop:24 },
  fcCard: { width:"100%", maxWidth:340, cursor:"pointer", perspective:1000 },
  fcInner: { position:"relative", width:"100%", minHeight:240, transformStyle:"preserve-3d", transition:"transform 0.4s ease", border:`1px solid ${C.border}`, boxShadow:"0 2px 12px rgba(0,0,0,0.06)" },
  fcFront: { position:"absolute", inset:0, background:C.surface, display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center", padding:28, backfaceVisibility:"hidden" },
  fcBack: { transform:"rotateY(180deg)" },
  fcLetter: { fontSize:80, fontWeight:700, color:C.text, lineHeight:1 },
  fcHint: { fontSize:14, color:C.textDim, marginTop:12, textAlign:"center" },
  fcPrompt: { fontSize:13, color:C.textMid, marginTop:12 },
  qBox: { background:C.surface, border:`1px solid ${C.border}`, padding:"36px 40px", maxWidth:440, width:"100%", display:"flex", flexDirection:"column", alignItems:"center", gap:16 },
  qText: { fontSize:16, fontWeight:600, color:C.text },
  letterInput: { width:76, height:76, textAlign:"center", fontSize:40, fontWeight:700, fontFamily:"'IBM Plex Mono', monospace", outline:"none", transition:"border-color 0.2s, background 0.2s, color 0.2s" },
  feedback: { fontSize:14, fontWeight:600 },
  resultBox: { textAlign:"center", display:"flex", flexDirection:"column", alignItems:"center", gap:12 },
  resultScore: { fontSize:60, fontWeight:700, letterSpacing:"-0.03em", color:C.text },
  resultPct: { fontSize:16, color:C.textMid, marginBottom:8 },
  siLayout: { display:"grid", gridTemplateColumns:"1fr 1fr", gap:36, maxWidth:760, width:"100%", alignItems:"start" },
  siLeft: { background:C.surface, border:`1px solid ${C.border}`, padding:"28px 24px", display:"flex", flexDirection:"column", alignItems:"center" },
  siBigLetter: { fontSize:110, fontWeight:700, lineHeight:1, letterSpacing:"-0.03em", color:C.text },
  siHint: { fontSize:13, color:C.textDim, marginTop:8, textAlign:"center" },
  siRight: { display:"flex", flexDirection:"column" },
  camWrap: { position:"relative", background:"#111", border:`1px solid ${C.border}`, aspectRatio:"4/3", overflow:"hidden" },
  camImg: { width:"100%", height:"100%", objectFit:"cover", display:"block" },
  camBlank: { height:"100%", display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center", gap:8 },
  camBlankIcon: { fontSize:28, color:"#555" },
  camBlankText: { fontSize:13, color:"#888" },
  handTag: { position:"absolute", top:8, right:8, padding:"3px 8px", fontSize:11, fontWeight:500 },
  camPred: { position:"absolute", bottom:10, left:"50%", transform:"translateX(-50%)", color:"#fff", fontSize:15, fontWeight:700, padding:"5px 14px", fontFamily:"'IBM Plex Mono', monospace", whiteSpace:"nowrap" },
  camNote: { fontSize:12, color:C.textDim, textAlign:"center", marginTop:8 },
};
