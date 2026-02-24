import { useState, useMemo, useEffect, useRef } from "react";

const C = {
  bg:       "#f5f5f0",
  surface:  "#ffffff",
  border:  "#d4d4d0",
  text:     "#1a1a1a",
  textMid:  "#555550",
  textDim:  "#888880",
  primary:  "#1a1a1a",
  correct:  "#166534",
  wrong:    "#b91c1c",
};

const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");

// Put A.png, B.png, … Z.png in public/practice/ to show sign images. Until then, placeholder is shown.
const PRACTICE_IMAGE_BASE = "/practice";
const TEST_QUESTION_COUNT = 5;
const QUESTION_TIME_SECONDS = 10;

const CIRCLE_TIMER_SIZE = 160;
const CIRCLE_STROKE = 14;
const CIRCLE_R = (CIRCLE_TIMER_SIZE - CIRCLE_STROKE) / 2;
const CIRCLE_CIRCUMFERENCE = 2 * Math.PI * CIRCLE_R;

function letterImageUrl(letter) {
  return `${PRACTICE_IMAGE_BASE}/${letter}.png`;
}

function LetterImage({ letter, style = {} }) {
  const [usePlaceholder, setUsePlaceholder] = useState(false);
  const src = letterImageUrl(letter);
  if (usePlaceholder) {
    return (
      <div style={{ ...s.signImagePlaceholder, ...style }} title="Add public/practice/[letter].png for real images">
        <span style={s.signImagePlaceholderText}>Sign image</span>
        <span style={s.signImagePlaceholderLetter}>{letter}</span>
      </div>
    );
  }
  return (
    <img
      src={src}
      alt={`ASL sign for ${letter}`}
      style={{ ...s.signImage, ...style }}
      onError={() => setUsePlaceholder(true)}
    />
  );
}

// Brief hint for each letter (ASL finger spelling)
const LETTER_HINTS = {
  A: "Fist, thumb to the side", B: "Flat hand, fingers together", C: "Curved C shape",
  D: "Point up, others in", E: "Fingers curved in", F: "OK sign, thumb & index",
  G: "Point sideways", H: "Index & middle, point sideways", I: "Pinky up",
  J: "Pinky traces J", K: "V with thumb", L: "L shape",
  M: "Three fingers down on thumb", N: "Two fingers on thumb", O: "O shape",
  P: "K with finger down", Q: "G pointing down", R: "R fingers crossed",
  S: "Fist, thumb in front", T: "Thumb between index & middle", U: "Two fingers up",
  V: "V shape", W: "Three fingers up", X: "Index bent", Y: "Thumb & pinky out",
  Z: "Z in the air",
};

const s = {
  root: {
    minHeight: "100vh", background: C.bg, fontFamily: "'IBM Plex Sans', system-ui, sans-serif",
    color: C.text, display: "flex", flexDirection: "column",
  },
  header: {
    background: C.surface, borderBottom: `1px solid ${C.border}`,
    padding: "0 20px", height: 52, display: "flex", alignItems: "center", justifyContent: "space-between",
  },
  backBtn: {
    padding: "8px 14px", fontSize: 14, fontWeight: 500, color: C.textMid,
    background: "none", border: `1px solid ${C.border}`, borderRadius: 6, cursor: "pointer",
  },
  headerTitle: { fontSize: 18, fontWeight: 600, color: C.text },
  main: {
    flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
    padding: 24,
  },
  modeGrid: {
    display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 20, maxWidth: 420,
  },
  modeCard: {
    background: C.surface, border: `2px solid ${C.border}`, borderRadius: 12,
    padding: "32px 24px", textAlign: "center", cursor: "pointer",
    transition: "border-color 0.2s, box-shadow 0.2s",
  },
  modeIcon: { fontSize: 40, marginBottom: 12 },
  modeLabel: { fontSize: 18, fontWeight: 600, color: C.text, marginBottom: 6 },
  modeDesc: { fontSize: 13, color: C.textDim },
  // Flashcards
  cardWrap: {
    width: "100%", maxWidth: 340, perspective: 1000,
  },
  cardInner: {
    position: "relative", width: "100%", minHeight: 220,
    transformStyle: "preserve-3d",
  },
  cardFace: {
    position: "absolute", inset: 0,
    background: C.surface, border: `2px solid ${C.border}`, borderRadius: 12,
    display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
    padding: 28, backfaceVisibility: "hidden",
    boxShadow: "0 4px 20px rgba(0,0,0,0.08)",
  },
  cardBack: { transform: "rotateY(180deg)" },
  signImage: {
    width: "100%", maxWidth: 200, height: 160, objectFit: "contain", borderRadius: 8,
    background: "transparent",
  },
  signImageLarge: {
    maxWidth: 320, height: 260, objectFit: "contain",
  },
  signImagePlaceholder: {
    width: "100%", maxWidth: 200, height: 160, background: "transparent", borderRadius: 8,
    display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
    border: `2px dashed ${C.border}`,
  },
  signImagePlaceholderLarge: {
    maxWidth: 320, height: 260,
  },
  signImagePlaceholderText: { fontSize: 12, color: C.textDim },
  signImagePlaceholderLetter: { fontSize: 48, fontWeight: 700, color: C.textDim, marginTop: 4 },
  cardLetter: { fontSize: 72, fontWeight: 700, color: C.primary, lineHeight: 1 },
  cardPrompt: { fontSize: 15, color: C.textMid, marginTop: 12 },
  cardHint: { fontSize: 14, color: C.textDim, marginTop: 12, textAlign: "center" },
  navRow: {
    display: "flex", alignItems: "center", justifyContent: "center", gap: 16, marginTop: 28,
  },
  navBtn: {
    padding: "10px 20px", fontSize: 15, fontWeight: 500, background: C.surface,
    border: `1px solid ${C.border}`, borderRadius: 8, cursor: "pointer", color: C.text,
  },
  progress: { fontSize: 14, color: C.textDim, marginTop: 12 },
  // Test (larger UI)
  questionBox: {
    background: C.surface, border: `2px solid ${C.border}`, borderRadius: 16,
    padding: 36, maxWidth: 520, width: "100%",
  },
  questionImageWrap: { marginBottom: 28, display: "flex", justifyContent: "center" },
  questionText: { fontSize: 22, fontWeight: 600, color: C.text, marginBottom: 24 },
  optionsGrid: { display: "flex", flexDirection: "column", gap: 14 },
  optionBtn: {
    padding: "18px 24px", fontSize: 20, textAlign: "left",
    background: C.surface, border: `2px solid ${C.border}`, borderRadius: 10,
    cursor: "pointer", fontWeight: 500, color: C.text,
    transition: "border-color 0.2s, background 0.2s",
  },
  optionCorrect: { borderColor: C.correct, background: "#f0fdf4" },
  optionWrong: { borderColor: C.wrong, background: "#fef2f2" },
  resultBox: {
    textAlign: "center", padding: 40,
  },
  resultScore: { fontSize: 48, fontWeight: 700, color: C.primary, marginBottom: 12 },
  resultText: { fontSize: 18, color: C.textMid, marginBottom: 24 },
  againBtn: {
    padding: "14px 28px", fontSize: 16, fontWeight: 600, color: C.surface, background: C.primary,
    border: "none", borderRadius: 8, cursor: "pointer",
  },
  // Circle countdown timer
  circleTimerWrap: {
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    marginBottom: 28,
  },
  circleTimerSvg: {
    transform: "rotate(-90deg)",
    width: CIRCLE_TIMER_SIZE,
    height: CIRCLE_TIMER_SIZE,
  },
  circleTimerBg: {
    fill: "none",
    stroke: C.border,
    strokeWidth: CIRCLE_STROKE,
  },
  circleTimerFill: {
    fill: "none",
    stroke: C.primary,
    strokeWidth: CIRCLE_STROKE,
    strokeLinecap: "round",
    transition: "stroke-dashoffset 0.3s ease",
  },
  circleTimerText: {
    position: "absolute",
    left: "50%",
    top: "50%",
    transform: "translate(-50%, -50%)",
    fontSize: 42,
    fontWeight: 700,
    fontFamily: "'IBM Plex Mono', monospace",
    color: C.text,
  },
};

function shuffle(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export default function Practice({ onNavigate }) {
  const [mode, setMode] = useState(null); // null | "flashcards" | "test"
  const [flashcardIndex, setFlashcardIndex] = useState(0);
  const [flipped, setFlipped] = useState(false);
  const [flashcardOrder, setFlashcardOrder] = useState(() => shuffle([...LETTERS]));
  // Test state
  const [testOrder, setTestOrder] = useState([]);
  const [testIndex, setTestIndex] = useState(0);
  const [testCorrect, setTestCorrect] = useState(0);
  const [testAnswered, setTestAnswered] = useState(false);
  const [testPicked, setTestPicked] = useState(null);
  const [testDone, setTestDone] = useState(false);
  const [testElapsed, setTestElapsed] = useState(0);
  const [countdown, setCountdown] = useState(QUESTION_TIME_SECONDS);
  const testStartRef = useRef(null);
  const nextQuestionRef = useRef(null);

  const startFlashcards = () => {
    setFlashcardOrder(shuffle([...LETTERS]));
    setFlashcardIndex(0);
    setFlipped(false);
  };

  const startTest = () => {
    setTestOrder(shuffle([...LETTERS]).slice(0, TEST_QUESTION_COUNT));
    setTestIndex(0);
    setTestCorrect(0);
    setTestAnswered(false);
    setTestPicked(null);
    setTestDone(false);
    setTestElapsed(0);
    setCountdown(QUESTION_TIME_SECONDS);
    testStartRef.current = Date.now();
  };

  // Test elapsed timer (total time)
  useEffect(() => {
    if (mode !== "test" || testDone || testOrder.length === 0) return;
    const tick = () => setTestElapsed(Math.floor((Date.now() - testStartRef.current) / 1000));
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, [mode, testDone, testOrder.length]);

  // Per-question 10s countdown (circle timer); on 0: play fail sound, then auto-advance
  useEffect(() => {
    if (mode !== "test" || testDone || testOrder.length === 0 || testAnswered) return;
    const id = setInterval(() => {
      setCountdown((c) => {
        if (c <= 1) {
          setTestAnswered(true);
          setTestPicked(null);
          setTimeout(() => nextQuestionRef.current?.(), 1500);
          return 0;
        }
        return c - 1;
      });
    }, 1000);
    return () => clearInterval(id);
  }, [mode, testDone, testOrder.length, testAnswered, testIndex]);

  const testQuestion = testOrder[testIndex];
  const testOptions = useMemo(() => {
    if (!testQuestion) return [];
    const others = LETTERS.filter((l) => l !== testQuestion);
    const wrong = shuffle(others).slice(0, 3);
    return shuffle([testQuestion, ...wrong]);
  }, [testQuestion]);

  const handleTestAnswer = (letter) => {
    if (testAnswered) return;
    setTestAnswered(true);
    setTestPicked(letter);
    if (letter === testQuestion) {
      setTestCorrect((c) => c + 1);
    }
  };

  const handleTestNext = () => {
    if (testIndex + 1 >= testOrder.length) {
      setTestElapsed(Math.floor((Date.now() - testStartRef.current) / 1000));
      setTestDone(true);
    } else {
      setTestIndex((i) => i + 1);
      setTestAnswered(false);
      setTestPicked(null);
      setCountdown(QUESTION_TIME_SECONDS);
    }
  };
  nextQuestionRef.current = handleTestNext;

  const handleBack = () => {
    if (mode === null) {
      onNavigate?.("home");
    } else {
      setMode(null);
      setFlipped(false);
      setFlashcardIndex(0);
    }
  };

  if (mode === null) {
    return (
      <div style={s.root}>
        <style>{`
          @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
          body { background: ${C.bg}; }
          button:hover { opacity: 0.9; }
          button:focus-visible { outline: 2px solid ${C.primary}; outline-offset: 2px; }
        `}</style>
        <header style={s.header}>
          <button style={s.backBtn} onClick={handleBack}>← Back</button>
          <span style={s.headerTitle}>Practice A–Z</span>
          <div style={{ width: 70 }} />
        </header>
        <main style={s.main}>
          <p style={{ fontSize: 16, color: C.textMid, marginBottom: 28 }}>
            Choose a way to practice the ASL alphabet
          </p>
          <div style={s.modeGrid}>
            <button
              style={s.modeCard}
              onClick={() => { setMode("flashcards"); startFlashcards(); }}
              onMouseEnter={(e) => { e.currentTarget.style.borderColor = C.primary; e.currentTarget.style.boxShadow = "0 8px 24px rgba(0,0,0,0.1)"; }}
              onMouseLeave={(e) => { e.currentTarget.style.borderColor = C.border; e.currentTarget.style.boxShadow = "none"; }}
            >
              <div style={s.modeIcon}>🃏</div>
              <div style={s.modeLabel}>Flashcards</div>
              <div style={s.modeDesc}>Flip through letters A–Z and see the sign hint</div>
            </button>
            <button
              style={s.modeCard}
              onClick={() => { setMode("test"); startTest(); }}
              onMouseEnter={(e) => { e.currentTarget.style.borderColor = C.primary; e.currentTarget.style.boxShadow = "0 8px 24px rgba(0,0,0,0.1)"; }}
              onMouseLeave={(e) => { e.currentTarget.style.borderColor = C.border; e.currentTarget.style.boxShadow = "none"; }}
            >
              <div style={s.modeIcon}>✏️</div>
              <div style={s.modeLabel}>Test</div>
              <div style={s.modeDesc}>Multiple choice: pick the correct letter (5 questions)</div>
            </button>
          </div>
        </main>
      </div>
    );
  }

  if (mode === "flashcards") {
    const letter = flashcardOrder[flashcardIndex];
    const hint = letter ? LETTER_HINTS[letter] : "";
    const atEnd = flashcardIndex >= 25;
    return (
      <div style={s.root}>
        <header style={s.header}>
          <button style={s.backBtn} onClick={handleBack}>← Back</button>
          <span style={s.headerTitle}>Flashcards</span>
          <span style={{ fontSize: 14, color: C.textDim }}>{flashcardIndex + 1} / 26</span>
        </header>
        <main style={s.main}>
          {letter ? (
            <>
              <div
                style={s.cardWrap}
                onClick={() => setFlipped((f) => !f)}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => e.key === "Enter" && setFlipped((f) => !f)}
              >
                <div style={{ ...s.cardInner, transform: flipped ? "rotateY(180deg)" : "none" }}>
                  <div style={s.cardFace}>
                    <LetterImage letter={letter} />
                    <div style={s.cardPrompt}>What letter? (tap to flip)</div>
                  </div>
                  <div style={{ ...s.cardFace, ...s.cardBack }}>
                    <div style={s.cardLetter}>{letter}</div>
                    <div style={s.cardHint}>{hint}</div>
                  </div>
                </div>
              </div>
              <div style={s.navRow}>
                <button
                  style={s.navBtn}
                  onClick={(e) => { e.stopPropagation(); setFlashcardIndex((i) => Math.max(0, i - 1)); setFlipped(false); }}
                  disabled={flashcardIndex === 0}
                >
                  ← Prev
                </button>
                <span style={s.progress}>{flashcardIndex + 1} / 26</span>
                <button
                  style={s.navBtn}
                  onClick={(e) => { e.stopPropagation(); setFlashcardIndex((i) => Math.min(25, i + 1)); setFlipped(false); }}
                  disabled={atEnd}
                >
                  Next →
                </button>
              </div>
              {atEnd && (
                <button style={{ ...s.againBtn, marginTop: 20 }} onClick={() => startFlashcards()}>
                  Restart (shuffle again)
                </button>
              )}
            </>
          ) : (
            <p>Loading…</p>
          )}
        </main>
      </div>
    );
  }

  // Test mode
  if (testDone) {
    const total = testOrder.length;
    const pct = total ? Math.round((testCorrect / total) * 100) : 0;
    return (
      <div style={s.root}>
        <header style={s.header}>
          <button style={s.backBtn} onClick={handleBack}>← Back</button>
          <span style={s.headerTitle}>Test</span>
          <div style={{ width: 70 }} />
        </header>
        <main style={s.main}>
          <div style={s.resultBox}>
            <div style={s.resultScore}>{testCorrect} / {total}</div>
            <div style={s.resultText}>{pct}% correct</div>
            <div style={{ ...s.resultText, marginBottom: 8 }}>Time: {formatTime(testElapsed)}</div>
            <button style={s.againBtn} onClick={() => startTest()}>
              Try again
            </button>
            <button style={{ ...s.backBtn, marginTop: 12 }} onClick={handleBack}>
              Back to Practice
            </button>
          </div>
        </main>
      </div>
    );
  }

  if (!testQuestion) {
    return (
      <div style={s.root}>
        <header style={s.header}>
          <button style={s.backBtn} onClick={handleBack}>← Back</button>
          <span style={s.headerTitle}>Test</span>
        </header>
        <main style={s.main}><p>Loading…</p></main>
      </div>
    );
  }

  return (
    <div style={s.root}>
      <header style={s.header}>
        <button style={s.backBtn} onClick={handleBack}>← Back</button>
        <span style={s.headerTitle}>Test</span>
        <span style={{ fontSize: 14, color: C.textDim, fontFamily: "'IBM Plex Mono', monospace" }}>
          {testIndex + 1} / {TEST_QUESTION_COUNT} · {formatTime(testElapsed)}
        </span>
      </header>
      <main style={s.main}>
        <div style={s.questionBox}>
          <div style={{ ...s.circleTimerWrap, position: "relative" }}>
            <svg style={s.circleTimerSvg} viewBox={`0 0 ${CIRCLE_TIMER_SIZE} ${CIRCLE_TIMER_SIZE}`}>
              <circle
                cx={CIRCLE_TIMER_SIZE / 2}
                cy={CIRCLE_TIMER_SIZE / 2}
                r={CIRCLE_R}
                style={s.circleTimerBg}
              />
              <circle
                cx={CIRCLE_TIMER_SIZE / 2}
                cy={CIRCLE_TIMER_SIZE / 2}
                r={CIRCLE_R}
                style={{
                  ...s.circleTimerFill,
                  strokeDasharray: CIRCLE_CIRCUMFERENCE,
                  strokeDashoffset: CIRCLE_CIRCUMFERENCE * (1 - countdown / QUESTION_TIME_SECONDS),
                }}
              />
            </svg>
            <span style={s.circleTimerText}>{countdown}</span>
          </div>
          <div style={s.questionImageWrap}>
            <LetterImage letter={testQuestion} style={s.signImageLarge} />
          </div>
          <div style={s.questionText}>Which letter is this sign for?</div>
          <div style={s.optionsGrid}>
            {testOptions.map((letter) => {
              let btnStyle = s.optionBtn;
              if (testAnswered) {
                if (letter === testQuestion) btnStyle = { ...s.optionBtn, ...s.optionCorrect };
                else if (letter === testPicked && letter !== testQuestion) btnStyle = { ...s.optionBtn, ...s.optionWrong };
              }
              return (
                <button
                  key={letter}
                  style={btnStyle}
                  onClick={() => handleTestAnswer(letter)}
                  disabled={testAnswered}
                >
                  {letter}
                </button>
              );
            })}
          </div>
        </div>
        {testAnswered && (
          <button style={{ ...s.againBtn, marginTop: 24 }} onClick={handleTestNext}>
            {testIndex + 1 >= testOrder.length ? "See results" : "Next →"}
          </button>
        )}
      </main>
    </div>
  );
}
