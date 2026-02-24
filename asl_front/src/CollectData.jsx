import { useState, useRef, useCallback, useEffect } from "react";

const WS_URL        = "ws://localhost:8766";
const WS_MOTION_URL = "ws://localhost:8769";
const RESTART_URL   = "ws://localhost:8767";
const CUSTOM_WORDS_KEY = "asl_custom_words";

const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");

const DEFAULT_WORD_SIGNS = [
  "[I-LOVE-YOU]","[GOOD]","[MORE]","[HELP]","[BOOK]",
  "[STOP]","[PLAY]","[WANT]","[WITH]","[SAME]",
  "[NO]","[YES]","[FRIEND]","[WORK]","[FINISH]",
  "[GO]","[SIT]","[BIG]","[SMALL]","[LOVE]","[EAT]","[DRINK]",
];

const MOTION_SIGNS = [
  { key:"J",           label:"J",         desc:"I-hand traces J curve" },
  { key:"Z",           label:"Z",         desc:"Index traces Z shape" },
  { key:"[PLEASE]",    label:"PLEASE",    desc:"Flat hand circles on chest" },
  { key:"[THANK_YOU]", label:"THANK YOU", desc:"Hand from chin outward" },
  { key:"[WHERE]",     label:"WHERE",     desc:"Index wags side-to-side" },
  { key:"[HOW]",       label:"HOW",       desc:"Curved hands roll forward" },
  { key:"[COME]",      label:"COME",      desc:"Index curls toward body" },
  { key:"[GO_AWAY]",   label:"GO AWAY",   desc:"Wrist flick outward" },
  { key:"[NAME]",      label:"NAME",      desc:"H-hand taps twice" },
];

const SEQ_LENGTH   = 30;
const MOTION_TARGET = 100;
const STATIC_TARGET = 200;

function wordToLabel(word) { return "[" + word.trim().toUpperCase().replace(/\s+/g,"-") + "]"; }
function loadCustomWordLabels() {
  try { return JSON.parse(localStorage.getItem(CUSTOM_WORDS_KEY)||"[]").map(w=>wordToLabel(w.word||w)); }
  catch { return []; }
}
function fmt(label) {
  if (label.startsWith("[")&&label.endsWith("]")) return label.slice(1,-1).replace(/-/g," ");
  return label;
}

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
    if (wsRef.current?.readyState === WebSocket.OPEN) wsRef.current.send(JSON.stringify(data));
  }, []);
  useEffect(() => () => wsRef.current?.close(), []);
  return { connected, connect, disconnect, send, onMessageRef };
}

export default function CollectData({ onNavigate }) {
  const [tab, setTab] = useState("letters");

  // ── Static state ──────────────────────────────────────────────────────────
  const [frame, setFrame]             = useState(null);
  const [numHands, setNumHands]       = useState(0);
  const [recording, setRecording]     = useState(false);
  const [currentLabel, setCurrentLabel] = useState("");
  const [sampleCount, setSampleCount] = useState(0);
  const [totalSamples, setTotalSamples] = useState(0);
  const [classStats, setClassStats]   = useState({});
  const [active, setActive]           = useState(false);
  const [fps, setFps]                 = useState(0);
  const [saveMsg, setSaveMsg]         = useState("");
  const [trimLabel, setTrimLabel]     = useState("");   // label currently confirming delete
  const [training, setTraining]       = useState(false);
  const [trainLog, setTrainLog]       = useState([]);
  const [showTrainLog, setShowTrainLog] = useState(false);
  const [restarting, setRestarting]   = useState(false);
  const trainLogRef = useRef(null);

  const [customWordSigns, setCustomWordSigns] = useState(loadCustomWordLabels);
  const WORD_SIGNS = [...DEFAULT_WORD_SIGNS, ...customWordSigns.filter(l=>!DEFAULT_WORD_SIGNS.includes(l))];
  const [showAddSign, setShowAddSign] = useState(false);
  const [newSignInput, setNewSignInput] = useState("");
  const [newSignError, setNewSignError] = useState("");

  // ── Motion state ──────────────────────────────────────────────────────────
  const [motionFrame, setMotionFrame]       = useState(null);
  const [motionNumHands, setMotionNumHands] = useState(0);
  const [motionState, setMotionState]       = useState("idle");   // idle | recording | done
  const [motionProgress, setMotionProgress] = useState(0);
  const [motionFrames, setMotionFrames]     = useState(0);
  const [hasPending, setHasPending]         = useState(false);
  const [motionCountMap, setMotionCountMap] = useState({});
  const [motionFps, setMotionFps]           = useState(0);
  const [motionActive, setMotionActive]     = useState(false);
  const [selectedSign, setSelectedSign]     = useState("");       // currently targeted label
  const [autoAdvance, setAutoAdvance]       = useState(true);     // auto-save + loop
  const [autoReps, setAutoReps]             = useState(5);        // sequences per burst
  const [autoDone, setAutoDone]             = useState(0);        // completed this burst
  const [autoRunning, setAutoRunning]       = useState(false);
  const [motionTraining, setMotionTraining] = useState(false);
  const [motionTrainLog, setMotionTrainLog] = useState([]);
  const [showMotionLog, setShowMotionLog]   = useState(false);
  const [lastSaved, setLastSaved]           = useState(null);
  const [motionTrimLabel, setMotionTrimLabel] = useState("");
  const motionLogRef = useRef(null);
  const autoRunningRef = useRef(false);

  // ── Static WS ─────────────────────────────────────────────────────────────
  const { connected, connect, disconnect, send, onMessageRef } = useWebSocket(WS_URL);
  onMessageRef.current = (msg) => {
    if (msg.frame)                       setFrame(msg.frame);
    if (msg.num_hands !== undefined)     setNumHands(msg.num_hands);
    if (msg.fps !== undefined)           setFps(msg.fps);
    if (msg.sample_count !== undefined)  setSampleCount(msg.sample_count);
    if (msg.total_samples !== undefined) setTotalSamples(msg.total_samples);
    if (msg.class_stats)                 setClassStats(msg.class_stats);
    if (msg.saved) { setSaveMsg(`Saved ${msg.saved} samples → ${msg.path}`); setTimeout(()=>setSaveMsg(""),3000); }
    if (msg.train_start) { setTraining(true); setTrainLog([]); setShowTrainLog(true); }
    if (msg.train_log) { setTrainLog(p=>[...p,msg.train_log]); setShowTrainLog(true); setTimeout(()=>{ if(trainLogRef.current) trainLogRef.current.scrollTop=trainLogRef.current.scrollHeight; },30); }
    if (msg.train_done) setTraining(false);
    if (msg.trimmed) { setSaveMsg(`Deleted all ${msg.label} samples from CSV`); setTimeout(()=>setSaveMsg(""),3000); }
  };

  const handleRestart = useCallback(() => {
    setRestarting(true);
    const ws = new WebSocket(RESTART_URL);
    ws.onopen    = () => ws.send(JSON.stringify({ action:"restart" }));
    ws.onmessage = () => { ws.close(); setTimeout(()=>setRestarting(false),2500); };
    ws.onerror   = () => setRestarting(false);
  }, []);

  const handleToggle = () => {
    if (active) { send({ action:"stop_recording" }); disconnect(); setActive(false); setFrame(null); setRecording(false); }
    else { connect(); setActive(true); }
  };
  const startRecording = (label) => { if (!connected) return; setCurrentLabel(label); setSampleCount(0); setRecording(true); send({ action:"start_recording", label }); };
  const stopRecording  = () => { if (!connected) return; setRecording(false); send({ action:"stop_recording" }); };
  const handleSave     = () => { if (connected) send({ action:"save" }); };
  const handleTrimLabel = (label) => { if (connected) { send({ action:"trim_label", label }); setTrimLabel(""); } };
  const handleTrain    = () => { if (connected && !training) send({ action:"train" }); };

  const handleAddSign = () => {
    const label = wordToLabel(newSignInput);
    if (!newSignInput.trim()) { setNewSignError("Enter a word or phrase."); return; }
    if (WORD_SIGNS.includes(label)) { setNewSignError("Already exists."); return; }
    try {
      const ex = JSON.parse(localStorage.getItem(CUSTOM_WORDS_KEY)||"[]");
      const w = newSignInput.trim().toLowerCase();
      if (!ex.find(x=>(x.word||x)===w)) { ex.unshift({ word:w, videoId:null }); localStorage.setItem(CUSTOM_WORDS_KEY,JSON.stringify(ex)); }
    } catch {}
    setCustomWordSigns(p=>[label,...p]);
    setCurrentLabel(label); setTab("words");
    setNewSignInput(""); setNewSignError(""); setShowAddSign(false);
  };

  // ── Motion WS ─────────────────────────────────────────────────────────────
  const { connected:motionConnected, connect:motionConnect, disconnect:motionDisconnect, send:motionSend, onMessageRef:motionOnMessageRef } = useWebSocket(WS_MOTION_URL);

  motionOnMessageRef.current = (msg) => {
    if (msg.frame)                         setMotionFrame(msg.frame);
    if (msg.num_hands !== undefined)       setMotionNumHands(msg.num_hands);
    if (msg.fps !== undefined)             setMotionFps(msg.fps);
    if (msg.state !== undefined)           setMotionState(msg.state);
    if (msg.progress !== undefined)        setMotionProgress(msg.progress);
    if (msg.frames_recorded !== undefined) setMotionFrames(msg.frames_recorded);
    if (msg.has_pending !== undefined)     setHasPending(msg.has_pending);
    if (msg.count_map !== undefined)       setMotionCountMap(msg.count_map);
    if (msg.saved_sequence) {
      setLastSaved({ label:msg.label, count:msg.count });
      setAutoDone(d => {
        const next = d + 1;
        // if auto-advance and still have reps, trigger next recording after short delay
        if (autoRunningRef.current && next < autoRepsRef.current) {
          setTimeout(() => { if (autoRunningRef.current) motionSend({ action:"start_recording" }); }, 600);
        } else if (autoRunningRef.current) {
          autoRunningRef.current = false;
          setAutoRunning(false);
        }
        return next;
      });
      setSelectedSign("");
      setTimeout(() => setLastSaved(null), 2500);
    }
    if (msg.trimmed_motion) { setLastSaved({ label:msg.label, count:0, deleted:true }); setMotionTrimLabel(""); setTimeout(()=>setLastSaved(null),2500); }
    if (msg.train_start) { setMotionTraining(true); setMotionTrainLog([]); setShowMotionLog(true); }
    if (msg.train_log)   { setMotionTrainLog(p=>[...p,msg.train_log]); setTimeout(()=>{ if(motionLogRef.current) motionLogRef.current.scrollTop=motionLogRef.current.scrollHeight; },30); }
    if (msg.train_done)  setMotionTraining(false);
  };

  // keep refs in sync so the closure above can read latest values
  const autoRepsRef = useRef(autoReps);
  useEffect(() => { autoRepsRef.current = autoReps; }, [autoReps]);

  const handleMotionToggle = () => {
    if (motionActive) {
      autoRunningRef.current = false; setAutoRunning(false);
      motionSend({ action:"stop_recording" });
      motionDisconnect(); setMotionActive(false); setMotionFrame(null);
      setMotionState("idle"); setHasPending(false);
    } else { motionConnect(); setMotionActive(true); }
  };

  const handleTabChange = (newTab) => {
    if (newTab === "motion" && active) { send({ action:"stop_recording" }); disconnect(); setActive(false); setFrame(null); setRecording(false); }
    if (newTab !== "motion" && motionActive) { autoRunningRef.current=false; setAutoRunning(false); motionSend({ action:"stop_recording" }); motionDisconnect(); setMotionActive(false); setMotionFrame(null); setMotionState("idle"); setHasPending(false); }
    setTab(newTab);
  };

  // Start a burst of auto-advance recordings
  const startAutoBurst = () => {
    if (!motionConnected || !selectedSign) return;
    setAutoDone(0);
    autoRunningRef.current = true;
    setAutoRunning(true);
    motionSend({ action:"start_recording" });
  };

  const stopAuto = () => {
    autoRunningRef.current = false;
    setAutoRunning(false);
    motionSend({ action:"stop_recording" });
  };

  // When a sequence is done and auto-advance is on, auto-save with selected label
  useEffect(() => {
    if (!autoRunningRef.current) return;
    if (motionState === "done" && hasPending && selectedSign) {
      motionSend({ action:"label_sequence", label:selectedSign });
    }
  }, [motionState, hasPending, selectedSign]);

  const startMotionRec = () => motionSend({ action:"start_recording" });
  const stopMotionRec  = () => motionSend({ action:"stop_recording" });
  const discardMotion  = () => { motionSend({ action:"discard" }); setSelectedSign(""); };
  const saveMotion     = () => { if (selectedSign) motionSend({ action:"label_sequence", label:selectedSign }); };
  const trainMotion    = () => { if (!motionTraining) { setMotionTrainLog([]); motionSend({ action:"train_motion" }); } };
  const trimMotionLabel = (label) => { motionSend({ action:"trim_motion_label", label }); setMotionTrimLabel(""); };

  return (
    <div style={s.root}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #f5f5f0; }
        button { font-family: 'IBM Plex Sans', sans-serif; cursor: pointer; }
        button:disabled { opacity: 0.38; cursor: default; }
        @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }
      `}</style>

      {/* Header */}
      <header style={s.header}>
        <div style={s.hLeft}>
          <div style={s.logoMark}>N</div>
          <span style={s.logoText}>NeuroSign</span>
          <span style={s.sep}>·</span>
          <span style={s.pageTitle}>Collect Data</span>
        </div>
        <div style={s.hRight}>
          {tab !== "motion" && active   && <FpsTag fps={fps} />}
          {tab === "motion" && motionActive && <FpsTag fps={motionFps} />}
          <StatusDot on={(tab==="motion" ? motionConnected : connected)} />
          <button style={{ ...s.btn, ...(restarting ? { color:"#b45309", border:"1px solid #fcd34d" } : {}) }}
            onClick={handleRestart} disabled={restarting}>
            {restarting ? "↺ Restarting…" : "↺ Restart"}
          </button>
          <button style={{ ...s.btn, ...s.btnPrimary }}
            onClick={tab==="motion" ? handleMotionToggle : handleToggle}>
            {(tab==="motion" ? motionActive : active) ? "Stop Camera" : "Start Camera"}
          </button>
        </div>
      </header>

      <div style={s.body}>
        {/* ── Left column: camera + controls ── */}
        <div style={s.leftCol}>

          {tab !== "motion" ? (
            <>
              {/* Camera */}
              <CamView frame={frame} active={active} numHands={numHands}>
                {recording && (
                  <div style={s.recBadge}>
                    <span style={{ ...s.recDot, animation:"blink 1s step-end infinite" }} />
                    {fmt(currentLabel)} · <span style={{ fontFamily:"'IBM Plex Mono', monospace" }}>{sampleCount}</span>
                  </div>
                )}
              </CamView>

              {/* Record controls */}
              <div style={s.panel}>
                <div style={s.recRow}>
                  <div style={{ flex:1 }}>
                    {currentLabel
                      ? <><span style={s.labelPill}>{fmt(currentLabel)}</span><span style={s.labelSub}>{sampleCount} this session</span></>
                      : <span style={s.labelEmpty}>Select a sign below to record</span>}
                  </div>
                  <button style={{ ...s.btn, ...(recording ? s.btnDanger : s.btnRecord) }}
                    onClick={recording ? stopRecording : ()=>currentLabel&&startRecording(currentLabel)}
                    disabled={!connected||(!recording&&!currentLabel)}>
                    {recording ? "■ Stop" : "● Record"}
                  </button>
                </div>
              </div>

              {/* Stats + save */}
              <div style={s.panel}>
                <div style={s.statsRow}>
                  <Stat num={totalSamples} label="Total" />
                  <Stat num={Object.keys(classStats).length} label="Classes" />
                  <Stat num={recording ? sampleCount : "—"} label="Session" />
                </div>
                <Divider />
                <div style={{ display:"flex", gap:8 }}>
                  <button style={{ ...s.btn, ...s.btnPrimary, flex:1 }} onClick={handleSave} disabled={!connected||totalSamples===0}>
                    Save CSV
                  </button>
                  <button style={{ ...s.btn, ...s.btnPrimary }} onClick={handleTrain} disabled={!connected||training||totalSamples===0}>
                    {training ? "⏳ Training…" : "▶ Train Model"}
                  </button>
                </div>
                {saveMsg && <div style={s.saveMsg}>{saveMsg}</div>}
                {training && <div style={s.warnMsg}>Training in progress — don't close the server.</div>}
                {showTrainLog && trainLog.length > 0 && (
                  <TrainLog lines={trainLog} done={!training} onClose={()=>setShowTrainLog(false)} ref={trainLogRef} />
                )}
              </div>
            </>
          ) : (
            <>
              {/* Motion camera */}
              <CamView frame={motionFrame} active={motionActive} numHands={motionNumHands}>
                {motionActive && (
                  <div style={s.motionBar}>
                    <div style={{
                      height:"100%", width:`${motionProgress*100}%`,
                      background: motionState==="recording" ? "#ef4444" : motionState==="done" ? "#2e7d32" : "transparent",
                      transition:"width 0.08s linear",
                    }} />
                  </div>
                )}
                {motionState==="recording" && (
                  <div style={s.recBadge}>
                    <span style={{ ...s.recDot, animation:"blink 1s step-end infinite" }} />
                    {selectedSign ? fmt(selectedSign) : "Recording"} · <span style={{ fontFamily:"'IBM Plex Mono', monospace" }}>{motionFrames}/{SEQ_LENGTH}</span>
                  </div>
                )}
              </CamView>

              {/* Auto-advance controls */}
              <div style={s.panel}>
                <div style={s.sectionLabel}>Target sign</div>
                <div style={{ display:"flex", flexWrap:"wrap", gap:5, marginBottom:10 }}>
                  {MOTION_SIGNS.map(({ key, label }) => (
                    <button key={key}
                      style={{ ...s.btn, ...(selectedSign===key ? s.btnPrimary : {}), fontSize:12, padding:"4px 10px" }}
                      onClick={()=>setSelectedSign(selectedSign===key ? "" : key)}
                      disabled={autoRunning}>
                      {label}
                    </button>
                  ))}
                </div>

                {/* Auto-advance burst */}
                <div style={s.sectionLabel}>Auto-advance mode</div>
                <div style={{ display:"flex", alignItems:"center", gap:8, marginBottom:8 }}>
                  <span style={{ fontSize:13, color:"#555550" }}>Reps per burst:</span>
                  {[3,5,10,20].map(n => (
                    <button key={n}
                      style={{ ...s.btn, ...(autoReps===n ? s.btnPrimary : {}), fontSize:12, padding:"4px 10px" }}
                      onClick={()=>setAutoReps(n)} disabled={autoRunning}>
                      {n}
                    </button>
                  ))}
                </div>
                <div style={s.recRow}>
                  <div style={{ flex:1 }}>
                    {autoRunning
                      ? <span style={s.labelPill}>{autoDone} / {autoReps} recorded</span>
                      : selectedSign
                        ? <span style={s.labelPill}>{fmt(selectedSign)} selected</span>
                        : <span style={s.labelEmpty}>Select a sign above, then start burst</span>}
                  </div>
                  {!autoRunning
                    ? <button style={{ ...s.btn, ...s.btnRecord }}
                        onClick={startAutoBurst} disabled={!motionConnected||!selectedSign}>
                        ▶ Start Burst
                      </button>
                    : <button style={{ ...s.btn, ...s.btnDanger }} onClick={stopAuto}>■ Stop</button>}
                </div>
                <p style={s.helpText}>
                  Burst mode records {autoReps} sequences automatically — just keep signing. Sign completes, saves, and re-records without pressing anything.
                </p>
              </div>

              {/* Manual recording (when not in burst) */}
              {!autoRunning && (
                <div style={s.panel}>
                  <div style={s.sectionLabel}>Manual recording</div>
                  <div style={s.recRow}>
                    <div style={{ flex:1, display:"flex", gap:6 }}>
                      {motionState !== "recording" && (
                        <button style={{ ...s.btn, ...s.btnRecord }} onClick={startMotionRec} disabled={!motionConnected}>● Record</button>
                      )}
                      {motionState === "recording" && (
                        <button style={{ ...s.btn, ...s.btnDanger }} onClick={stopMotionRec}>■ Stop</button>
                      )}
                      {motionState === "done" && hasPending && (
                        <button style={{ ...s.btn, color:"#b91c1c", border:"1px solid #fca5a5" }} onClick={discardMotion}>Discard</button>
                      )}
                    </div>
                    {motionState === "done" && hasPending && (
                      <button style={{ ...s.btn, ...s.btnPrimary }} onClick={saveMotion} disabled={!selectedSign}>
                        Save{selectedSign ? ` as ${fmt(selectedSign)}` : " (select sign)"}
                      </button>
                    )}
                  </div>
                </div>
              )}

              {lastSaved && (
                <div style={lastSaved.deleted ? s.warnMsg : s.saveMsg}>
                  {lastSaved.deleted
                    ? `Deleted all ${fmt(lastSaved.label)} motion data`
                    : `Saved — ${lastSaved.count} sequences for ${fmt(lastSaved.label)}`}
                </div>
              )}

              {/* Stats + train */}
              <div style={s.panel}>
                <div style={s.statsRow}>
                  <Stat num={Object.values(motionCountMap).reduce((a,b)=>a+b,0)} label="Sequences" />
                  <Stat num={Object.keys(motionCountMap).length} label="Signs" />
                </div>
                <Divider />
                <button style={{ ...s.btn, ...s.btnPrimary, ...(motionTraining?{opacity:0.7}:{}) }}
                  onClick={trainMotion} disabled={!motionConnected||motionTraining}>
                  {motionTraining ? "⏳ Training…" : "▶ Train Motion Model"}
                </button>
                {motionTraining && <div style={s.warnMsg}>Training — don't close the server.</div>}
                {showMotionLog && motionTrainLog.length > 0 && (
                  <TrainLog lines={motionTrainLog} done={!motionTraining} onClose={()=>setShowMotionLog(false)} ref={motionLogRef} />
                )}
              </div>
            </>
          )}
        </div>

        {/* ── Right column: sign grid ── */}
        <div style={s.rightCol}>
          <div style={s.tabs}>
            {["letters","words","motion"].map(t => (
              <button key={t} style={{ ...s.tab, ...(tab===t ? s.tabActive : {}) }}
                onClick={()=>handleTabChange(t)}>
                {t === "letters" ? "Letters A–Z" : t === "words" ? "Word Signs" : "Motion Signs"}
              </button>
            ))}
            <div style={{ flex:1 }} />
            {tab === "words" && (
              <button style={{ ...s.btn, ...s.btnPrimary, fontSize:12, padding:"4px 12px", margin:"6px 0" }}
                onClick={()=>{ setNewSignInput(""); setNewSignError(""); setShowAddSign(true); }}>
                + Add
              </button>
            )}
          </div>

          {tab !== "motion" ? (
            <div style={s.grid}>
              {(tab==="letters" ? LETTERS : WORD_SIGNS).map(label => {
                const count = classStats[label] || 0;
                const pct   = Math.min(count/STATIC_TARGET, 1);
                const isActive = currentLabel === label;
                const isTrimming = trimLabel === label;
                return (
                  <div key={label} style={s.signCell}>
                    <button
                      style={{ ...s.signBtn, ...(isActive ? s.signBtnActive : {}), ...(count>=STATIC_TARGET ? s.signBtnDone : {}) }}
                      onClick={()=>{ setCurrentLabel(label); if(recording) send({ action:"stop_recording" }); setRecording(false); setTrimLabel(""); }}
                      disabled={!connected}>
                      <span style={s.signLabel}>{fmt(label)}</span>
                      <div style={s.bar}><div style={{ ...s.barFill, width:`${pct*100}%`, background:count>=STATIC_TARGET?"#2e7d32":"#1a1a1a" }} /></div>
                      <span style={s.signCount}>{count}</span>
                    </button>
                    {/* Delete button — appears on hover via state */}
                    {!isTrimming
                      ? <button style={s.deleteBtn} title={`Delete all ${fmt(label)} data`}
                          onClick={e=>{ e.stopPropagation(); setTrimLabel(label); }}
                          disabled={!connected||count===0}>✕</button>
                      : <div style={s.confirmDelete}>
                          <span style={{ fontSize:10, color:"#b91c1c" }}>Delete {count} rows?</span>
                          <button style={{ ...s.btn, background:"#b91c1c", color:"#fff", border:"none", fontSize:11, padding:"2px 7px" }}
                            onClick={()=>handleTrimLabel(label)}>Yes</button>
                          <button style={{ ...s.btn, fontSize:11, padding:"2px 7px" }}
                            onClick={()=>setTrimLabel("")}>No</button>
                        </div>
                    }
                  </div>
                );
              })}
            </div>
          ) : (
            <div style={s.grid}>
              {MOTION_SIGNS.map(({ key, label, desc }) => {
                const count   = motionCountMap[key] || 0;
                const pct     = Math.min(count/MOTION_TARGET, 1);
                const isSel   = selectedSign === key;
                const isTrim  = motionTrimLabel === key;
                return (
                  <div key={key} style={s.signCell}>
                    <button title={desc}
                      style={{ ...s.signBtn, ...(isSel ? s.signBtnActive : {}), ...(count>=MOTION_TARGET ? s.signBtnDone : {}) }}
                      onClick={()=>setSelectedSign(isSel ? "" : key)}>
                      <span style={s.signLabel}>{label}</span>
                      <div style={s.bar}><div style={{ ...s.barFill, width:`${pct*100}%`, background:count>=MOTION_TARGET?"#2e7d32":"#1a1a1a" }} /></div>
                      <span style={s.signCount}>{count}/{MOTION_TARGET}</span>
                    </button>
                    {!isTrim
                      ? <button style={s.deleteBtn} title={`Delete all ${label} motion data`}
                          onClick={e=>{ e.stopPropagation(); setMotionTrimLabel(key); }}
                          disabled={!motionConnected||count===0}>✕</button>
                      : <div style={s.confirmDelete}>
                          <span style={{ fontSize:10, color:"#b91c1c" }}>Delete {count} seqs?</span>
                          <button style={{ ...s.btn, background:"#b91c1c", color:"#fff", border:"none", fontSize:11, padding:"2px 7px" }}
                            onClick={()=>trimMotionLabel(key)}>Yes</button>
                          <button style={{ ...s.btn, fontSize:11, padding:"2px 7px" }}
                            onClick={()=>setMotionTrimLabel("")}>No</button>
                        </div>
                    }
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* Add sign modal */}
      {showAddSign && (
        <div style={s.overlay} onClick={()=>setShowAddSign(false)}>
          <div style={s.modal} onClick={e=>e.stopPropagation()}>
            <div style={s.modalHead}>
              <span style={{ fontSize:15, fontWeight:600 }}>Add a new sign</span>
              <button style={{ background:"none", border:"none", fontSize:16, color:"#888880", cursor:"pointer" }} onClick={()=>setShowAddSign(false)}>✕</button>
            </div>
            <div style={{ padding:"16px 18px 14px", display:"flex", flexDirection:"column", gap:10 }}>
              <label style={{ fontSize:12, fontWeight:600, color:"#555550" }}>Word or phrase</label>
              <input style={s.input} type="text" placeholder="e.g. bathroom, thank you"
                value={newSignInput}
                onChange={e=>{ setNewSignInput(e.target.value); setNewSignError(""); }}
                onKeyDown={e=>e.key==="Enter"&&handleAddSign()} autoFocus />
              {newSignInput && <div style={{ fontSize:11, color:"#555550", fontFamily:"'IBM Plex Mono', monospace", background:"#f0f0eb", padding:"4px 8px" }}>Label: <strong>{wordToLabel(newSignInput)}</strong></div>}
              {newSignError && <div style={{ fontSize:12, color:"#b91c1c" }}>{newSignError}</div>}
              <div style={{ display:"flex", gap:8, marginTop:4 }}>
                <button style={{ ...s.btn, ...s.btnPrimary }} onClick={handleAddSign}>Add sign</button>
                <button style={s.btn} onClick={()=>setShowAddSign(false)}>Cancel</button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Small sub-components ──────────────────────────────────────────────────────
function CamView({ frame, active, numHands, children }) {
  return (
    <div style={s.camWrap}>
      {frame
        ? <img src={`data:image/jpeg;base64,${frame}`} alt="feed" style={s.camImg} />
        : <div style={s.camBlank}>
            <span style={{ fontSize:28, color:"#555" }}>◻</span>
            <span style={{ fontSize:13, color:"#888" }}>{active ? "Connecting…" : "Camera inactive"}</span>
          </div>
      }
      {active && (
        <div style={{
          ...s.handTag,
          background: numHands>0 ? "#e8f5e9" : "#fff8e1",
          color: numHands>0 ? "#2e7d32" : "#f57f17",
          border: `1px solid ${numHands>0 ? "#a5d6a7" : "#ffe082"}`,
        }}>
          {numHands>0 ? `${numHands} hand${numHands>1?"s":""}` : "No hands"}
        </div>
      )}
      {children}
    </div>
  );
}
function FpsTag({ fps }) { return <span style={s.fpsTag}>{fps>0 ? `${fps} fps` : "—"}</span>; }
function StatusDot({ on }) { return <span style={{ width:8, height:8, borderRadius:"50%", background:on?"#2e7d32":"#9e9e9e", display:"inline-block" }} />; }
function Stat({ num, label }) {
  return (
    <div style={{ flex:1, display:"flex", flexDirection:"column", gap:2, paddingRight:14, borderRight:`1px solid #d4d4d0`, marginRight:14, lastChild:{ borderRight:"none", marginRight:0 } }}>
      <span style={{ fontSize:22, fontWeight:600, fontFamily:"'IBM Plex Mono', monospace" }}>{num}</span>
      <span style={{ fontSize:11, color:"#888880", textTransform:"uppercase", letterSpacing:"0.06em" }}>{label}</span>
    </div>
  );
}
function Divider() { return <div style={{ borderTop:"1px solid #d4d4d0" }} />; }
function TrainLog({ lines, done, onClose, ref: logRef }) {
  return (
    <div style={{ border:"1px solid #d4d4d0", overflow:"hidden" }}>
      <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", padding:"6px 10px", background:"#f0f0eb", borderBottom:"1px solid #d4d4d0" }}>
        <span style={{ fontSize:11, fontWeight:600, color:"#555550" }}>{done ? "Training complete ✓" : "Training output"}</span>
        <button style={{ background:"none", border:"none", fontSize:13, color:"#888880", cursor:"pointer" }} onClick={onClose}>✕</button>
      </div>
      <div ref={logRef} style={{ maxHeight:160, overflowY:"auto", padding:"8px 10px", background:"#1a1a1a", fontFamily:"'IBM Plex Mono', monospace", fontSize:11, lineHeight:1.6 }}>
        {lines.map((l,i)=><div key={i} style={{ color:"#d4d4d0", whiteSpace:"pre-wrap", wordBreak:"break-all" }}>{l}</div>)}
      </div>
    </div>
  );
}

const C = {
  bg:"#f5f5f0", surface:"#ffffff", border:"#d4d4d0",
  text:"#1a1a1a", textMid:"#555550", textDim:"#888880",
};
const s = {
  root: { minHeight:"100vh", background:C.bg, fontFamily:"'IBM Plex Sans', system-ui, sans-serif", fontSize:14, color:C.text, display:"flex", flexDirection:"column" },
  header: { background:C.surface, borderBottom:`1px solid ${C.border}`, padding:"0 28px", height:52, display:"flex", alignItems:"center", justifyContent:"space-between", gap:16 },
  hLeft: { display:"flex", alignItems:"center", gap:10 },
  hRight: { display:"flex", alignItems:"center", gap:10 },
  logoMark: { width:24, height:24, background:C.text, color:"#fff", display:"flex", alignItems:"center", justifyContent:"center", fontSize:12, fontWeight:700, flexShrink:0 },
  logoText: { fontSize:15, fontWeight:700, letterSpacing:"-0.02em", color:C.text },
  sep: { fontSize:16, color:C.textDim },
  pageTitle: { fontSize:14, fontWeight:500, color:C.textMid },
  fpsTag: { padding:"3px 9px", background:"#f0f0eb", border:`1px solid ${C.border}`, fontSize:12, fontFamily:"'IBM Plex Mono', monospace", color:C.textDim },
  body: { flex:1, display:"grid", gridTemplateColumns:"440px 1fr", gap:20, padding:"20px 28px 28px", maxWidth:1200, width:"100%", margin:"0 auto", alignItems:"start" },
  leftCol: { display:"flex", flexDirection:"column", gap:12 },
  camWrap: { position:"relative", background:"#111", border:`1px solid ${C.border}`, aspectRatio:"4/3", overflow:"hidden" },
  camImg: { width:"100%", height:"100%", objectFit:"cover", display:"block" },
  camBlank: { height:"100%", display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center", gap:8 },
  recBadge: { position:"absolute", bottom:10, left:"50%", transform:"translateX(-50%)", background:"rgba(0,0,0,0.75)", color:"#fff", fontSize:13, fontWeight:500, padding:"5px 14px", display:"flex", alignItems:"center", gap:7, whiteSpace:"nowrap" },
  recDot: { display:"inline-block", width:8, height:8, borderRadius:"50%", background:"#ef4444" },
  handTag: { position:"absolute", top:8, right:8, padding:"3px 8px", fontSize:11, fontWeight:500 },
  motionBar: { position:"absolute", bottom:0, left:0, right:0, height:4, background:"rgba(255,255,255,0.1)" },
  panel: { background:C.surface, border:`1px solid ${C.border}`, padding:"14px 16px", display:"flex", flexDirection:"column", gap:10 },
  recRow: { display:"flex", alignItems:"center", justifyContent:"space-between", gap:10 },
  labelPill: { padding:"3px 10px", background:C.text, color:"#fff", fontSize:13, fontWeight:600, letterSpacing:"0.05em" },
  labelSub: { fontSize:12, color:C.textDim, fontFamily:"'IBM Plex Mono', monospace", marginLeft:8 },
  labelEmpty: { fontSize:13, color:C.textDim, fontStyle:"italic" },
  statsRow: { display:"flex" },
  sectionLabel: { fontSize:11, fontWeight:600, letterSpacing:"0.08em", textTransform:"uppercase", color:C.textDim },
  saveMsg: { fontSize:12, color:"#2e7d32", padding:"6px 10px", background:"#e8f5e9", border:"1px solid #a5d6a7" },
  warnMsg: { fontSize:12, color:"#b45309", padding:"6px 10px", background:"#fffbeb", border:"1px solid #fcd34d" },
  helpText: { fontSize:12, color:C.textDim, lineHeight:1.5 },
  rightCol: { display:"flex", flexDirection:"column" },
  tabs: { display:"flex", borderBottom:`1px solid ${C.border}`, marginBottom:12 },
  tab: { padding:"8px 16px", background:"none", border:"none", borderBottom:"2px solid transparent", fontSize:13, fontWeight:500, color:C.textMid, marginBottom:-1, cursor:"pointer" },
  tabActive: { borderBottomColor:C.text, color:C.text, fontWeight:600 },
  grid: { display:"grid", gridTemplateColumns:"repeat(auto-fill, minmax(100px, 1fr))", gap:4 },
  signCell: { position:"relative", display:"flex", flexDirection:"column" },
  signBtn: { padding:"10px 8px 8px", border:`1px solid ${C.border}`, background:C.surface, display:"flex", flexDirection:"column", alignItems:"center", gap:4, cursor:"pointer", width:"100%" },
  signBtnActive: { border:`1px solid ${C.text}`, background:"#fafaf8" },
  signBtnDone: { border:"1px solid #a5d6a7", background:"#f1f8f1" },
  signLabel: { fontSize:13, fontWeight:600, color:C.text },
  bar: { width:"100%", height:3, background:"#e5e5e0", overflow:"hidden" },
  barFill: { height:"100%", transition:"width 0.3s ease" },
  signCount: { fontSize:11, fontFamily:"'IBM Plex Mono', monospace", color:C.textDim },
  deleteBtn: { position:"absolute", top:2, right:2, background:"none", border:"none", fontSize:10, color:C.textDim, padding:"1px 3px", lineHeight:1 },
  confirmDelete: { display:"flex", alignItems:"center", gap:4, padding:"3px 4px", background:"#fef2f2", border:"1px solid #fca5a5", flexWrap:"wrap" },
  btn: { padding:"7px 14px", border:`1px solid ${C.border}`, fontSize:13, fontWeight:500, background:C.surface, color:C.text },
  btnPrimary: { background:C.text, color:"#fff", border:`1px solid ${C.text}` },
  btnRecord: { background:C.text, color:"#fff", border:`1px solid ${C.text}`, padding:"7px 18px" },
  btnDanger: { background:"#b91c1c", color:"#fff", border:"1px solid #b91c1c", padding:"7px 18px" },
  overlay: { position:"fixed", inset:0, background:"rgba(0,0,0,0.4)", display:"flex", alignItems:"center", justifyContent:"center", zIndex:100 },
  modal: { background:C.surface, border:`1px solid ${C.border}`, width:"100%", maxWidth:400, boxShadow:"0 8px 32px rgba(0,0,0,0.12)" },
  modalHead: { display:"flex", alignItems:"center", justifyContent:"space-between", padding:"13px 18px", borderBottom:`1px solid ${C.border}` },
  input: { width:"100%", padding:"8px 10px", border:`1px solid ${C.border}`, fontSize:13, color:C.text, background:C.bg, outline:"none" },
};