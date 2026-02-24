# Motion-Based ASL Sign Recognition

This extends the existing ASL translator with a **motion-based classifier** that handles signs requiring hand movement — like J, Z (letters) and words such as PLEASE, THANK YOU, WHERE, etc.

## Architecture Overview

```
Static signs (A-Z minus J/Z, word poses)  →  LinearSVC on single-frame landmarks
Motion signs (J, Z, PLEASE, THANK_YOU...) →  LSTM/CNN on 30-frame sequences
```

The motion system works by:
1. **Buffering** 30 consecutive frames of hand landmarks into a sliding window
2. **Motion gating** — only classifying when the hand is actually moving
3. **Temporal feature engineering** — adding frame-to-frame deltas + velocity magnitude
4. **Classification** via 1D-CNN + LSTM (or Random Forest fallback if PyTorch unavailable)

---

## Motion Signs Supported

| Sign | Category | Motion |
|------|----------|--------|
| J | Letter | I-hand traces J curve downward then hooks left |
| Z | Letter | Index finger traces Z shape in air |
| [PLEASE] | Word | Flat hand circles clockwise on chest |
| [THANK_YOU] | Word | Flat hand from chin extending outward |
| [WHERE] | Word | Index finger wags side to side |
| [HOW] | Word | Curved hands roll forward |
| [COME] | Word | Index finger curls/beckons inward |
| [GO_AWAY] | Word | Wrist flick outward |
| [NAME] | Word | H-hand taps twice |

---

## Quick Start

### Option A — Synthetic data (fast)
```bash
python generate_motion_dataset.py
python train_motion_model.py
python motion_translator.py
```

### Option B — Collect your own data (best accuracy)
```bash
python collect_motion_data.py
python train_motion_model.py
python motion_translator.py
```

---

## New Files

```
asl-translator/
├── collect_motion_data.py     # Webcam collector for motion sign sequences
├── generate_motion_dataset.py # Synthetic motion sequence generator
├── train_motion_model.py      # LSTM + CNN trainer for motion sequences
├── motion_translator.py       # Real-time motion sign translator
└── data/
    └── motion_landmarks.csv   # Motion sequence data (label, frame, features...)
    └── model/
        └── motion_classifier.pkl  # Trained motion model
```

---

## How to Collect Your Own Motion Data

Run `collect_motion_data.py`:

```
Controls:
  SPACE       — Start recording
  SPACE again — Stop recording (auto-stops at 30 frames)
  J / Z       — Label as letter motion sign
  1           — Label as [PLEASE]
  2           — Label as [THANK_YOU]
  3           — Label as [WHERE]
  4           — Label as [HOW]
  5           — Label as [WHAT]
  6           — Label as [COME]
  7           — Label as [GO_AWAY]
  8           — Label as [NAME]
  BACKSPACE   — Discard last recording
  ESC         — Save and quit
```

Aim for **100+ sequences per sign**. Vary your speed and position.

---

## Model Details

### Feature Engineering
Each frame produces a **126-dim vector** (63 landmarks × xyz × 2 hands, zeros for absent hand).

The training pipeline adds:
- **Frame deltas** (+126 dim) — captures velocity
- **Velocity magnitude** (+1 dim) — captures motion intensity
- Final: **(N, 30, 253)** tensor per sequence

### Model Architecture (PyTorch)
```
Input (batch, 30, 253)
  → Conv1D(253→128, k=3) + ReLU
  → Conv1D(128→64, k=3) + ReLU
  → Dropout(0.3)
  → LSTM(64 → 128, 2 layers)
  → Dense(128 → num_classes)
  → Softmax
```

### Fallback (no PyTorch)
If PyTorch is not installed, a **RandomForest** classifier runs on flattened sequences.

Install PyTorch for best results:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## Motion Translator Controls

| Key | Action |
|-----|--------|
| SPACE | Add space |
| BACKSPACE | Delete last word |
| C | Clear sentence |
| ESC | Quit |

The translator shows:
- **Motion sign** — from the motion model (shown when hand is moving)
- **Static sign** — from the existing static model (shown always, for reference)
- **Buffer bar** — how full the 30-frame sequence window is
- **Motion bar** — current hand velocity

---

## Tips for Best Results

1. **Make deliberate, complete motions** — start and finish the sign cleanly
2. **Hold still before and after** — the motion gate needs a clear start/stop
3. **Keep hand in frame** the whole time
4. **Vary speed** when collecting data — some people sign faster than others
5. **Collect 200+ sequences** per sign for production accuracy
