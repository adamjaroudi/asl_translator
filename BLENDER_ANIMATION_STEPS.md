# What to Do to Get Blender Animation Playback Working

This is a step-by-step checklist so the translator can send the current sentence to Blender and Blender can play the corresponding sign animations.

---

## Your scope: A–Z only, map and animate

Right now you only care about **letters A–Z**:

1. User spells with the camera (e.g. H, E, L, L, O). The sentence might be `"HELLO"` or `"H E L L O"` if they use Space.
2. You send that **sentence string** to Blender (e.g. `"HELLO"`).
3. In Blender you **map each character** to an animation: A → `anim_A`, B → `anim_B`, … Z → `anim_Z`. Skip spaces (or treat space as a short pause).
4. **Animate**: play the sequence `anim_H`, `anim_E`, `anim_L`, `anim_L`, `anim_O` in order.

So you need **26 letter animations** in Blender (plus optional “space”/pause). No word signs or phrase mapping for now.

---

## 1. What Blender receives (for A–Z)

- Send the **sentence string** from the app (e.g. `"HELLO"` or `"H E L L O"`).
- In Blender: loop over each character; for each letter A–Z, play the matching animation; ignore or pause on spaces. That’s it.

---

## 2. Add a “Play in 3D” trigger in the app

- In the **frontend** (`ASLTranslator.jsx`): add a button (e.g. “Play in 3D” or “▶ 3D”) and/or a keyboard shortcut (e.g. **P**).
- When the user triggers it:
  - Either send a message to the **existing WebSocket** (translator on port 8765) asking for “play in 3D” with the current sentence, **or**
  - Send the current sentence to a **separate** Blender endpoint (see step 3).
- The translator WebSocket already has the sentence; the frontend also has `data.sentence`. So the frontend can send e.g. `{ "action": "play_3d", "sentence": data.sentence }` to the backend, and the backend forwards it to Blender (see step 3).

---

## 3. Connect the backend to Blender

- **Option A — Backend forwards to Blender**  
  - In `server.py`: when the translator client sends `play_3d` with the sentence, the server sends that sentence to Blender (e.g. over a **second WebSocket** that Blender opens to `localhost:8766`).
  - Start a small **WebSocket server** in `server.py` on port 8766 that only Blender connects to. When the app says “play”, the server pushes the sentence to that Blender client.

- **Option B — Frontend talks to Blender directly**  
  - Blender runs a WebSocket **server** (inside Blender, via a script). The frontend opens a connection to e.g. `ws://localhost:8766` and sends the sentence when “Play in 3D” is clicked.  
  - No change to `server.py`; Blender must be listening before you click.

**Recommendation:** Use **Option A** (backend forwards) so a single connection from the frontend (to the translator) handles both translation and 3D playback.

---

## 4. Blender side — listener script (A–Z)

- In Blender, you need a script that:
  1. Opens a **WebSocket client** to your server (e.g. `ws://localhost:8766`) and stays connected.
  2. On message `{ "action": "play", "sentence": "HELLO" }`:
     - Normalize: `sentence = sentence.upper().replace(" ", "")` (or keep spaces and treat as pause).
     - For each character `c` in `sentence`:
       - If `c` is A–Z: map to action name (e.g. `"anim_A"` … `"anim_Z"`).
       - If you kept spaces: optional short pause between letters.
     - Play the list of actions **in sequence** (one after another).
- You need a **Blender scene** with:
  - A character (or hands) with an **armature**.
  - **26 actions** named e.g. `anim_A`, `anim_B`, … `anim_Z` (or `A`, `B`, … `Z`). Placeholder poses are fine at first.
- Use Blender’s API to set the current action and advance the timeline (or NLA) for each letter, then the next.

---

## 5. Run order (once everything is wired)

1. Start **Python backend**: `python server.py` (translator on 8765; optional Blender relay on 8766).
2. Start **Blender**, open your scene, run the **listener script** so it connects to the backend (or so the frontend can connect to Blender).
3. Start the **frontend**: `cd asl_front && npm start`, open the translator.
4. Build a sentence with the camera, then click **“Play in 3D”** (or press **P**). The sentence should be sent to Blender and the character should play the sign sequence.

---

## 6. Optional improvements later

- **Sign sequence (Option B):** In `server.py`, maintain a list of “signs added” (e.g. `["Hello", " ", "World"]` or raw labels like `["[hello]", " ", "[world]"]`) and send that list to Blender instead of (or in addition to) the sentence string. Blender then maps each sign label to one animation with no ambiguity.
- **Documentation:** Add a short section in the main README or a separate doc: “3D playback: open Blender, run the listener, then use ‘Play in 3D’ in the app.”

---

## Summary checklist (A–Z only)

| Step | What to do |
|------|------------|
| 1 | Send sentence string (letters only for now). Blender receives e.g. `"HELLO"`. |
| 2 | Add “Play in 3D” button/key (e.g. P) in `ASLTranslator.jsx`; send `play_3d` + sentence to backend. |
| 3 | In `server.py`, add a Blender WebSocket (e.g. port 8766); when frontend sends `play_3d`, forward sentence to Blender. |
| 4 | In Blender: listener script (WebSocket client), armature + 26 actions (A–Z), loop over each char and play `anim_X` in sequence. |
| 5 | Run: server → Blender (listener) → frontend → spell with camera → Play in 3D. |

Once this is in place, you can refine the 26 letter animations and add word signs later if you want.
