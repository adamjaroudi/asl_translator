import { useState, useRef, useEffect, useCallback } from "react";

// Shared word list key — also read by CollectData
export const CUSTOM_WORDS_KEY = "asl_custom_words";

const YT_SEARCH_URL = (word) =>
  `https://www.youtube.com/results?search_query=${encodeURIComponent("ASL sign " + word)}`;

const YT_EMBED_URL = (videoId) =>
  `https://www.youtube-nocookie.com/embed/${videoId}?autoplay=0&rel=0`;

// YouTube's oEmbed endpoint can't give us search results without an API key.
// Instead we use the public YouTube search page URL and let the user click a result,
// OR the user pastes a YouTube URL/ID to pin a specific video to that word.
// We also provide a direct "Search YouTube" button that opens in a new tab.

const STARTER_WORDS = [
  "hello", "thank you", "please", "sorry", "help",
  "yes", "no", "good", "love", "friend",
  "water", "food", "bathroom", "more", "stop",
  "learn", "understand", "work", "finish", "want",
];

function parseYouTubeId(input) {
  if (!input) return null;
  // Direct video ID (11 chars)
  if (/^[a-zA-Z0-9_-]{11}$/.test(input.trim())) return input.trim();
  // Full URL
  try {
    const url = new URL(input);
    if (url.hostname.includes("youtu.be")) return url.pathname.slice(1).split("?")[0];
    if (url.hostname.includes("youtube.com")) return url.searchParams.get("v");
  } catch {}
  // youtu.be shortlink without protocol
  const match = input.match(/(?:youtu\.be\/|v=)([a-zA-Z0-9_-]{11})/);
  return match ? match[1] : null;
}

function loadCustomWords() {
  try { return JSON.parse(localStorage.getItem(CUSTOM_WORDS_KEY) || "[]"); }
  catch { return []; }
}

function saveCustomWords(words) {
  try { localStorage.setItem(CUSTOM_WORDS_KEY, JSON.stringify(words)); }
  catch {}
}

const C = {
  bg:      "#f5f5f0",
  surface: "#ffffff",
  border:  "#d4d4d0",
  text:    "#1a1a1a",
  textMid: "#555550",
  textDim: "#888880",
};

export default function Dictionary({ onNavigate }) {
  const [query, setQuery]           = useState("");
  const [activeWord, setActiveWord] = useState(null);   // currently viewed word object
  const [savedWords, setSavedWords] = useState(STARTER_WORDS.map(w => ({ word: w, videoId: null })));
  const [customWords, setCustomWords] = useState(loadCustomWords);

  // Add word modal
  const [showAddModal, setShowAddModal] = useState(false);
  const [addWordInput, setAddWordInput] = useState("");
  const [addVideoInput, setAddVideoInput] = useState("");
  const [addError, setAddError]         = useState("");

  // Pin video modal (for existing words)
  const [showPinModal, setShowPinModal] = useState(false);
  const [pinInput, setPinInput]         = useState("");
  const [pinError, setPinError]         = useState("");

  const inputRef = useRef(null);

  // Persist custom words
  useEffect(() => saveCustomWords(customWords), [customWords]);

  // All words combined: saved + custom
  const allWords = [
    ...customWords,
    ...savedWords.filter(s => !customWords.find(c => c.word === s.word)),
  ];

  const handleSelectWord = (wordObj) => {
    setActiveWord(wordObj);
    setQuery(wordObj.word);
  };

  const handleSearch = useCallback(() => {
    const w = query.trim().toLowerCase();
    if (!w) return;
    const existing = allWords.find(x => x.word === w);
    if (existing) {
      setActiveWord(existing);
    } else {
      setActiveWord({ word: w, videoId: null });
    }
  }, [query, allWords]);

  const handleKeyDown = (e) => { if (e.key === "Enter") handleSearch(); };

  // Add new word (from modal)
  const handleAddWord = () => {
    const word = addWordInput.trim().toLowerCase();
    if (!word) { setAddError("Word cannot be empty."); return; }
    if (allWords.find(x => x.word === word)) { setAddError("Word already exists."); return; }
    const videoId = addVideoInput.trim() ? parseYouTubeId(addVideoInput.trim()) : null;
    if (addVideoInput.trim() && !videoId) { setAddError("Invalid YouTube URL or ID."); return; }
    const newWord = { word, videoId };
    setCustomWords(prev => [newWord, ...prev]);
    setActiveWord(newWord);
    setQuery(word);
    setAddWordInput(""); setAddVideoInput(""); setAddError("");
    setShowAddModal(false);
  };

  // Remove a custom word
  const handleRemoveCustom = (word) => {
    setCustomWords(prev => prev.filter(w => w.word !== word));
    if (activeWord?.word === word) setActiveWord(null);
  };

  // Pin a YouTube video to the active word
  const handlePinVideo = () => {
    const videoId = parseYouTubeId(pinInput.trim());
    if (!videoId) { setPinError("Invalid YouTube URL or ID."); return; }
    const updated = { ...activeWord, videoId };
    setActiveWord(updated);
    // Update in whichever list it belongs to
    setCustomWords(prev => {
      const idx = prev.findIndex(w => w.word === activeWord.word);
      if (idx >= 0) {
        const next = [...prev]; next[idx] = updated; return next;
      }
      // Not in custom — add it
      return [updated, ...prev];
    });
    setSavedWords(prev => prev.map(w => w.word === activeWord.word ? updated : w));
    setPinInput(""); setPinError(""); setShowPinModal(false);
  };

  const isCustom = (word) => customWords.some(c => c.word === word);

  return (
    <div style={s.root}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #f5f5f0; }
        button { font-family: 'IBM Plex Sans', sans-serif; cursor: pointer; }
        input  { font-family: 'IBM Plex Sans', sans-serif; }
        .chip-btn:hover { background: #1a1a1a !important; color: #fff !important; border-color: #1a1a1a !important; }
      `}</style>

      {/* ── Header ── */}
      <header style={s.header}>
        <div style={s.headerLeft}>
          <button style={s.backBtn} onClick={() => onNavigate("translator")}>← Translator</button>
          <span style={s.divV} />
          <span style={s.headerTitle}>ASL Dictionary</span>
        </div>
        <div style={s.headerRight}>
          <button style={{ ...s.btn, ...s.btnPrimary }} onClick={() => setShowAddModal(true)}>
            + Add Word
          </button>
          <button style={s.headerNavBtn} onClick={() => onNavigate("collect")}>Collect Data →</button>
        </div>
      </header>

      <div style={s.body}>

        {/* ── Left sidebar ── */}
        <div style={s.sidebar}>

          {/* Search */}
          <div style={s.searchRow}>
            <input
              ref={inputRef}
              style={s.searchInput}
              type="text"
              placeholder="Search a word…"
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              autoFocus
            />
            <button style={{ ...s.btn, ...s.btnPrimary }} onClick={handleSearch}>Go</button>
          </div>

          {/* Custom words */}
          {customWords.length > 0 && (
            <div style={s.wordGroup}>
              <div style={s.groupHeader}>
                <span style={s.groupLabel}>My Words</span>
                <span style={s.groupCount}>{customWords.length}</span>
              </div>
              <div style={s.wordList}>
                {customWords.map(w => (
                  <div key={w.word} style={s.wordRow}>
                    <button
                      className="chip-btn"
                      style={{
                        ...s.wordBtn,
                        ...(activeWord?.word === w.word ? s.wordBtnActive : {}),
                      }}
                      onClick={() => handleSelectWord(w)}
                    >
                      <span style={s.wordBtnText}>{w.word}</span>
                      {w.videoId && <span style={s.videoPin}>▶</span>}
                    </button>
                    <button style={s.removeBtn} onClick={() => handleRemoveCustom(w.word)} title="Remove">✕</button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Starter words */}
          <div style={s.wordGroup}>
            <div style={s.groupHeader}>
              <span style={s.groupLabel}>Common Signs</span>
              <span style={s.groupCount}>{savedWords.length}</span>
            </div>
            <div style={s.wordList}>
              {savedWords.map(w => (
                <button
                  key={w.word}
                  className="chip-btn"
                  style={{
                    ...s.wordBtn,
                    ...(activeWord?.word === w.word ? s.wordBtnActive : {}),
                  }}
                  onClick={() => handleSelectWord(w)}
                >
                  <span style={s.wordBtnText}>{w.word}</span>
                  {w.videoId && <span style={s.videoPin}>▶</span>}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* ── Main content ── */}
        <div style={s.main}>
          {!activeWord ? (
            <div style={s.emptyState}>
              <span style={s.emptyIcon}>🤟</span>
              <span style={s.emptyTitle}>Look up any ASL sign</span>
              <span style={s.emptyText}>
                Select a word from the list or search above.
                You can pin a YouTube video to any word so it always shows the right sign.
              </span>
              <button style={{ ...s.btn, ...s.btnPrimary, marginTop: 8 }} onClick={() => setShowAddModal(true)}>
                + Add a word
              </button>
            </div>
          ) : (
            <div style={s.wordView}>
              {/* Word header */}
              <div style={s.wordViewHeader}>
                <div>
                  <div style={s.wordViewTitle}>{activeWord.word}</div>
                  {isCustom(activeWord.word) && (
                    <span style={s.customBadge}>custom</span>
                  )}
                </div>
                <div style={s.wordViewActions}>
                  <button style={{ ...s.btn, ...s.btnSecondary }} onClick={() => { setPinInput(""); setPinError(""); setShowPinModal(true); }}>
                    {activeWord.videoId ? "✎ Change video" : "📌 Pin a video"}
                  </button>
                  <a
                    href={YT_SEARCH_URL(activeWord.word)}
                    target="_blank"
                    rel="noreferrer"
                    style={s.ytLink}
                  >
                    Search YouTube ↗
                  </a>
                </div>
              </div>

              {/* Video area */}
              {activeWord.videoId ? (
                <div style={s.iframeWrap}>
                  <iframe
                    key={activeWord.videoId}
                    src={YT_EMBED_URL(activeWord.videoId)}
                    style={s.iframe}
                    title={`ASL sign for ${activeWord.word}`}
                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                    allowFullScreen
                  />
                </div>
              ) : (
                <div style={s.noPinnedVideo}>
                  <div style={s.noPinnedIcon}>▶</div>
                  <div style={s.noPinnedTitle}>No video pinned yet</div>
                  <div style={s.noPinnedText}>
                    Find the sign on YouTube, then paste the URL here to pin it permanently.
                  </div>
                  <div style={s.noPinnedBtns}>
                    <a
                      href={YT_SEARCH_URL(activeWord.word)}
                      target="_blank"
                      rel="noreferrer"
                      style={{ ...s.btn, ...s.btnPrimary, textDecoration: "none", display: "inline-block" }}
                    >
                      Search "{activeWord.word}" on YouTube ↗
                    </a>
                    <button style={{ ...s.btn, ...s.btnSecondary }} onClick={() => { setPinInput(""); setPinError(""); setShowPinModal(true); }}>
                      📌 Paste YouTube URL
                    </button>
                  </div>
                </div>
              )}

              {/* How to pin instructions */}
              {!activeWord.videoId && (
                <div style={s.instructBox}>
                  <div style={s.instructTitle}>How to pin a video</div>
                  <ol style={s.instructList}>
                    <li>Click "Search on YouTube" above</li>
                    <li>Find a clear ASL tutorial for this sign</li>
                    <li>Copy the video URL from your browser</li>
                    <li>Click "Paste YouTube URL" and paste it in</li>
                  </ol>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* ── Add Word Modal ── */}
      {showAddModal && (
        <div style={s.overlay} onClick={() => setShowAddModal(false)}>
          <div style={s.modal} onClick={e => e.stopPropagation()}>
            <div style={s.modalHeader}>
              <span style={s.modalTitle}>Add a word</span>
              <button style={s.modalClose} onClick={() => setShowAddModal(false)}>✕</button>
            </div>
            <div style={s.modalBody}>
              <label style={s.fieldLabel}>Word or phrase</label>
              <input
                style={s.fieldInput}
                type="text"
                placeholder="e.g. bathroom, i love you"
                value={addWordInput}
                onChange={e => { setAddWordInput(e.target.value); setAddError(""); }}
                onKeyDown={e => e.key === "Enter" && handleAddWord()}
                autoFocus
              />
              <label style={{ ...s.fieldLabel, marginTop: 14 }}>
                YouTube URL <span style={s.fieldOptional}>(optional — you can add later)</span>
              </label>
              <input
                style={s.fieldInput}
                type="text"
                placeholder="https://youtube.com/watch?v=..."
                value={addVideoInput}
                onChange={e => { setAddVideoInput(e.target.value); setAddError(""); }}
                onKeyDown={e => e.key === "Enter" && handleAddWord()}
              />
              {addError && <div style={s.fieldError}>{addError}</div>}

              <div style={s.modalFooter}>
                <button style={{ ...s.btn, ...s.btnPrimary }} onClick={handleAddWord}>Add word</button>
                <button style={s.btn} onClick={() => { setShowAddModal(false); setAddError(""); }}>Cancel</button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Pin Video Modal ── */}
      {showPinModal && (
        <div style={s.overlay} onClick={() => setShowPinModal(false)}>
          <div style={s.modal} onClick={e => e.stopPropagation()}>
            <div style={s.modalHeader}>
              <span style={s.modalTitle}>Pin a YouTube video for "{activeWord?.word}"</span>
              <button style={s.modalClose} onClick={() => setShowPinModal(false)}>✕</button>
            </div>
            <div style={s.modalBody}>
              <label style={s.fieldLabel}>YouTube URL or video ID</label>
              <input
                style={s.fieldInput}
                type="text"
                placeholder="https://youtube.com/watch?v=dQw4w9WgXcQ"
                value={pinInput}
                onChange={e => { setPinInput(e.target.value); setPinError(""); }}
                onKeyDown={e => e.key === "Enter" && handlePinVideo()}
                autoFocus
              />
              {pinError && <div style={s.fieldError}>{pinError}</div>}
              <div style={s.modalFooter}>
                <button style={{ ...s.btn, ...s.btnPrimary }} onClick={handlePinVideo}>Pin video</button>
                {activeWord?.videoId && (
                  <button
                    style={{ ...s.btn, color: "#b91c1c", borderColor: "#fca5a5" }}
                    onClick={() => {
                      const updated = { ...activeWord, videoId: null };
                      setActiveWord(updated);
                      setCustomWords(prev => prev.map(w => w.word === activeWord.word ? updated : w));
                      setSavedWords(prev => prev.map(w => w.word === activeWord.word ? updated : w));
                      setShowPinModal(false);
                    }}
                  >
                    Remove video
                  </button>
                )}
                <button style={s.btn} onClick={() => setShowPinModal(false)}>Cancel</button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────
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
  headerLeft:  { display: "flex", alignItems: "center", gap: 14 },
  headerRight: { display: "flex", alignItems: "center", gap: 8 },
  backBtn: {
    background: "none", border: "none", fontSize: 13,
    fontWeight: 500, color: C.textMid, padding: 0, cursor: "pointer",
  },
  divV: { width: 1, height: 18, background: C.border, display: "block" },
  headerTitle:  { fontSize: 15, fontWeight: 600 },
  headerNavBtn: {
    padding: "5px 13px", background: "none", border: `1px solid ${C.border}`,
    fontSize: 13, fontWeight: 500, color: C.textMid, cursor: "pointer",
  },

  body: {
    flex: 1, display: "grid", gridTemplateColumns: "260px 1fr",
    gap: 0, maxWidth: 1200, width: "100%", margin: "0 auto",
    alignItems: "start",
  },

  // Sidebar
  sidebar: {
    borderRight: `1px solid ${C.border}`, padding: "16px 0",
    display: "flex", flexDirection: "column", gap: 0,
    minHeight: "calc(100vh - 52px)",
    background: C.surface,
  },
  searchRow: { display: "flex", gap: 6, padding: "0 14px 14px", borderBottom: `1px solid ${C.border}` },
  searchInput: {
    flex: 1, padding: "7px 10px", border: `1px solid ${C.border}`,
    fontSize: 13, color: C.text, background: C.bg, outline: "none",
  },
  wordGroup: { padding: "14px 0" },
  groupHeader: {
    display: "flex", alignItems: "center", justifyContent: "space-between",
    padding: "0 14px 8px", borderBottom: `1px solid ${C.border}`, marginBottom: 4,
  },
  groupLabel: {
    fontSize: 10, fontWeight: 600, letterSpacing: "0.1em",
    textTransform: "uppercase", color: C.textDim,
  },
  groupCount: {
    fontSize: 11, fontFamily: "'IBM Plex Mono', monospace", color: C.textDim,
    background: "#f0f0eb", padding: "1px 6px", border: `1px solid ${C.border}`,
  },
  wordList: { display: "flex", flexDirection: "column" },
  wordRow:  { display: "flex", alignItems: "stretch" },
  wordBtn: {
    flex: 1, padding: "7px 14px", background: "none", border: "none",
    textAlign: "left", fontSize: 13, color: C.textMid, cursor: "pointer",
    display: "flex", alignItems: "center", justifyContent: "space-between",
    transition: "background 0.08s",
  },
  wordBtnActive: { background: "#1a1a1a", color: "#fff" },
  wordBtnText: { flex: 1 },
  videoPin: { fontSize: 10, color: "#2e7d32", marginLeft: 6 },
  removeBtn: {
    padding: "0 10px", background: "none", border: "none",
    fontSize: 11, color: C.textDim, cursor: "pointer",
    borderLeft: `1px solid ${C.border}`,
  },

  // Main
  main: {
    padding: "24px 28px",
    minHeight: "calc(100vh - 52px)",
  },
  emptyState: {
    display: "flex", flexDirection: "column", alignItems: "center",
    justifyContent: "center", gap: 12, padding: "60px 40px",
    maxWidth: 400, margin: "0 auto", textAlign: "center",
  },
  emptyIcon:  { fontSize: 52 },
  emptyTitle: { fontSize: 20, fontWeight: 600, color: C.textMid },
  emptyText:  { fontSize: 13, color: C.textDim, lineHeight: 1.7 },

  wordView: { display: "flex", flexDirection: "column", gap: 16, maxWidth: 820 },
  wordViewHeader: {
    display: "flex", alignItems: "center", justifyContent: "space-between",
    flexWrap: "wrap", gap: 10,
    paddingBottom: 14, borderBottom: `1px solid ${C.border}`,
  },
  wordViewTitle: { fontSize: 26, fontWeight: 600, color: C.text },
  customBadge: {
    display: "inline-block", marginTop: 4, padding: "2px 8px",
    fontSize: 11, fontWeight: 600, letterSpacing: "0.06em",
    background: "#fffbeb", border: "1px solid #fcd34d", color: "#b45309",
  },
  wordViewActions: { display: "flex", alignItems: "center", gap: 10 },
  ytLink: {
    fontSize: 13, color: C.textMid, textDecoration: "none",
    borderBottom: `1px solid ${C.border}`, paddingBottom: 1,
  },

  iframeWrap: {
    width: "100%", aspectRatio: "16/9",
    border: `1px solid ${C.border}`, overflow: "hidden",
    background: "#000",
  },
  iframe: { width: "100%", height: "100%", border: "none", display: "block" },

  noPinnedVideo: {
    display: "flex", flexDirection: "column", alignItems: "center",
    justifyContent: "center", gap: 12, padding: "48px 40px",
    border: `1px solid ${C.border}`, background: "#fafaf8",
    textAlign: "center",
  },
  noPinnedIcon:  { fontSize: 36, color: C.textDim },
  noPinnedTitle: { fontSize: 17, fontWeight: 600, color: C.textMid },
  noPinnedText:  { fontSize: 13, color: C.textDim, lineHeight: 1.7, maxWidth: 360 },
  noPinnedBtns:  { display: "flex", gap: 10, marginTop: 4, flexWrap: "wrap", justifyContent: "center" },

  instructBox: {
    padding: "14px 16px", border: `1px solid ${C.border}`,
    background: "#f5f5f0",
  },
  instructTitle: {
    fontSize: 11, fontWeight: 600, textTransform: "uppercase",
    letterSpacing: "0.08em", color: C.textDim, marginBottom: 8,
  },
  instructList: {
    paddingLeft: 18, display: "flex", flexDirection: "column", gap: 5,
    fontSize: 13, color: C.textMid, lineHeight: 1.6,
  },

  // Modals
  overlay: {
    position: "fixed", inset: 0, background: "rgba(0,0,0,0.4)",
    display: "flex", alignItems: "center", justifyContent: "center", zIndex: 100,
  },
  modal: {
    background: C.surface, border: `1px solid ${C.border}`,
    width: "100%", maxWidth: 440, boxShadow: "0 8px 32px rgba(0,0,0,0.12)",
  },
  modalHeader: {
    display: "flex", alignItems: "center", justifyContent: "space-between",
    padding: "14px 18px", borderBottom: `1px solid ${C.border}`,
  },
  modalTitle: { fontSize: 15, fontWeight: 600 },
  modalClose: {
    background: "none", border: "none", fontSize: 16,
    color: C.textDim, cursor: "pointer", padding: "0 2px",
  },
  modalBody:   { padding: "18px 18px 14px" },
  modalFooter: { display: "flex", gap: 8, marginTop: 18 },
  fieldLabel:  { display: "block", fontSize: 12, fontWeight: 600, color: C.textMid, marginBottom: 5 },
  fieldOptional: { fontWeight: 400, color: C.textDim },
  fieldInput: {
    width: "100%", padding: "8px 10px", border: `1px solid ${C.border}`,
    fontSize: 13, color: C.text, background: C.bg, outline: "none",
  },
  fieldError: {
    marginTop: 7, fontSize: 12, color: "#b91c1c",
    padding: "5px 8px", background: "#fef2f2", border: "1px solid #fca5a5",
  },

  // Buttons
  btn: {
    padding: "7px 14px", border: `1px solid ${C.border}`,
    fontSize: 13, fontWeight: 500, background: C.surface, color: C.text, cursor: "pointer",
  },
  btnPrimary:   { background: "#1a1a1a", color: "#fff", border: "1px solid #1a1a1a" },
  btnSecondary: { background: C.surface, color: C.text, border: `1px solid ${C.border}` },
};