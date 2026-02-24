"""
collect_motion_data.py
======================
Webcam data collector for MOTION-BASED ASL signs.

Unlike static signs (single frame), motion signs require a sequence of frames
to capture the trajectory of the hand over time.

Usage:
    python collect_motion_data.py

Controls:
    SPACE       — Start / stop recording a sequence
    S           — Save the last recorded sequence (label it with a key below)
    A-Z         — Label the sign after recording
    1-9         — Label word signs after recording
    BACKSPACE   — Discard last recording
    ESC         — Save all data and quit

Motion Signs supported:
    J   — I-hand makes a J curve downward
    Z   — Index finger traces Z in the air
    1 → [PLEASE]    — Flat hand circles on chest
    2 → [THANK_YOU] — Fingers from chin outward
    3 → [WHERE]     — Index finger wags side to side
    4 → [HOW]       — Curved hands roll forward
    5 → [WHAT]      — Hands spread open/out
    6 → [WANT]      — Clawed hands pull toward body
    7 → [COME]      — Index fingers curl toward body
    8 → [GO_AWAY]   — Wrist flick outward
    9 → [NAME]      — H-hand taps twice
"""

import os
import csv
import time
import json
import pickle
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Paths ────────────────────────────────────────────────────────────────────
MODEL_DIR   = "model"
DATA_DIR    = "data"
TASK_FILE   = os.path.join(MODEL_DIR, "hand_landmarker.task")
DATA_FILE   = os.path.join(DATA_DIR,  "motion_landmarks.csv")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
SEQ_LENGTH    = 30      # frames per sequence
NUM_LANDMARKS = 21
FEATURE_DIM   = 63      # 21 landmarks × 3 (x,y,z) per hand, normalized

WORD_SIGNS = {
    "1": "[PLEASE]",
    "2": "[THANK_YOU]",
    "3": "[WHERE]",
    "4": "[HOW]",
    "5": "[WHAT]",
    "6": "[WANT]",
    "7": "[COME]",
    "8": "[GO_AWAY]",
    "9": "[NAME]",
}

TARGET_SAMPLES = 100  # sequences per sign

# ── Hand landmarker setup ─────────────────────────────────────────────────────
def download_task_file():
    """Download MediaPipe hand landmarker if needed."""
    if os.path.exists(TASK_FILE):
        return
    print("Downloading hand_landmarker.task...")
    import urllib.request
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, TASK_FILE)
    print("Downloaded.")

def make_landmarker():
    download_task_file()
    opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=TASK_FILE),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(opts)

# ── Feature extraction ────────────────────────────────────────────────────────
def extract_frame_features(hand_landmarks_list):
    """
    Extract a flat 63-dim (or 126-dim for 2 hands) normalized feature vector
    from one frame's hand landmark results.
    Returns a 126-dim vector (2 hands worth, zeros if only 1 or 0 hands).
    """
    def normalize_hand(lms):
        pts = np.array([[l.x, l.y, l.z] for l in lms])
        # Normalize relative to wrist
        wrist = pts[0].copy()
        pts -= wrist
        # Scale by hand size (distance wrist → middle finger MCP)
        scale = np.linalg.norm(pts[9]) + 1e-6
        pts /= scale
        return pts.flatten()  # 63 values

    h1 = np.zeros(63)
    h2 = np.zeros(63)

    if hand_landmarks_list:
        h1 = normalize_hand(hand_landmarks_list[0])
        if len(hand_landmarks_list) > 1:
            h2 = normalize_hand(hand_landmarks_list[1])

    return np.concatenate([h1, h2])  # 126-dim


# ── CSV I/O ───────────────────────────────────────────────────────────────────
def load_existing():
    """Load existing motion sequences. Returns list of (label, seq_array)."""
    sequences = []
    if not os.path.exists(DATA_FILE):
        return sequences
    with open(DATA_FILE, "r") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            label = row[0]
            frame_idx = int(row[1])
            values = list(map(float, row[2:]))
            # rebuild sequences
            if not sequences or sequences[-1][0] != label or frame_idx == 0:
                sequences.append((label, []))
            sequences[-1][1].append(values)
    return sequences


def count_per_label(sequences):
    counts = {}
    for label, _ in sequences:
        counts[label] = counts.get(label, 0) + 1
    return counts


def append_sequence(label, sequence):
    """Append a single sequence (list of 126-dim arrays) to the CSV."""
    file_exists = os.path.exists(DATA_FILE)
    with open(DATA_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            # Write header
            feat_cols = [f"f{i}" for i in range(126)]
            writer.writerow(["label", "frame"] + feat_cols)
        for fi, frame_feats in enumerate(sequence):
            writer.writerow([label, fi] + list(frame_feats))


# ── Display helpers ───────────────────────────────────────────────────────────
COLORS = {
    "idle":       (100, 100, 100),
    "recording":  (0,   0,   220),
    "done":       (0,   200, 50),
    "saved":      (0,   180, 255),
}

def draw_ui(frame, state, label, count_map, pending_seq, status_msg):
    h, w = frame.shape[:2]

    # Header bar
    bar_color = COLORS.get(state, (80, 80, 80))
    cv2.rectangle(frame, (0, 0), (w, 70), bar_color, -1)

    state_text = {
        "idle":      "IDLE — Press SPACE to start recording",
        "recording": "● RECORDING — Press SPACE to stop",
        "done":      "DONE — Press a letter/number key to label & save",
        "saved":     f"SAVED as '{label}' — Keep going!",
    }.get(state, state)
    cv2.putText(frame, state_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    if status_msg:
        cv2.putText(frame, status_msg, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220,220,220), 1)

    # Side panel
    panel_x = w - 230
    cv2.rectangle(frame, (panel_x, 0), (w, h), (30, 30, 30), -1)
    cv2.putText(frame, "MOTION SIGNS", (panel_x+10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
    cv2.putText(frame, f"Target: {TARGET_SAMPLES}/sign", (panel_x+10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,150,150), 1)

    # Letter signs
    y = 75
    for letter in "JZ":
        cnt = count_map.get(letter, 0)
        pct = min(cnt / TARGET_SAMPLES, 1.0)
        col = (0, 200, 80) if pct >= 1 else (200, 200, 200)
        cv2.putText(frame, f"{letter}: {cnt}", (panel_x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        cv2.rectangle(frame, (panel_x+80, y-12), (panel_x+210, y-2), (60,60,60), -1)
        cv2.rectangle(frame, (panel_x+80, y-12), (panel_x+80+int(130*pct), y-2), col, -1)
        y += 22

    y += 5
    cv2.putText(frame, "Word signs:", (panel_x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)
    y += 20
    for key, wsign in WORD_SIGNS.items():
        cnt = count_map.get(wsign, 0)
        pct = min(cnt / TARGET_SAMPLES, 1.0)
        col = (0, 200, 80) if pct >= 1 else (200, 200, 200)
        short = wsign[1:-1]  # strip brackets
        cv2.putText(frame, f"{key}:{short[:8]:8s}{cnt}", (panel_x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)
        y += 18

    # Pending indicator
    if pending_seq is not None:
        cv2.putText(frame, f"Pending: {len(pending_seq)} frames", (10, h-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,220,220), 1)

    return frame


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    landmarker = make_landmarker()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Load existing data counts
    existing = load_existing()
    count_map = count_per_label(existing)
    all_sequences = list(existing)

    state        = "idle"      # idle | recording | done
    recording    = []          # list of 126-dim arrays (current recording)
    pending_seq  = None        # completed sequence waiting to be labelled
    last_label   = ""
    status_msg   = ""
    frame_ts     = 0

    print("\n=== Motion Sign Data Collector ===")
    print("  SPACE     — Start/stop recording")
    print("  A-Z       — Label sign after recording")
    print("  1-9       — Label word sign after recording")
    print("  BACKSPACE — Discard pending sequence")
    print("  ESC       — Quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.flip(frame, 1, dst=frame)
        frame_ts += 33

        # Run detection
        small = cv2.resize(frame, (640, 480))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = landmarker.detect_for_video(mp_img, frame_ts)

        feats = extract_frame_features(results.hand_landmarks if results.hand_landmarks else [])

        # Draw landmarks
        if results.hand_landmarks:
            mp_draw = mp.solutions.drawing_utils
            mp_hands_style = mp.solutions.hands
            h_fr, w_fr = frame.shape[:2]
            # Scale landmarks back to full frame
            for lms in results.hand_landmarks:
                pts = [(int(l.x * w_fr), int(l.y * h_fr)) for l in lms]
                for i, j in mp.solutions.hands.HAND_CONNECTIONS:
                    cv2.line(frame, pts[i], pts[j], (0, 255, 120), 2)
                for pt in pts:
                    cv2.circle(frame, pt, 4, (255, 255, 255), -1)

        if state == "recording":
            recording.append(feats)
            # Draw progress
            prog = len(recording) / SEQ_LENGTH
            cv2.rectangle(frame, (10, 80), (int(10 + 300*prog), 100), (0, 0, 220), -1)
            cv2.rectangle(frame, (10, 80), (310, 100), (200, 200, 200), 2)
            cv2.putText(frame, f"Frame {len(recording)}/{SEQ_LENGTH}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

            if len(recording) >= SEQ_LENGTH:
                # Auto-stop when full
                pending_seq = list(recording)
                recording   = []
                state       = "done"
                status_msg  = "Full! Press a key to label it."

        ui = draw_ui(frame, state, last_label, count_map, pending_seq, status_msg)
        cv2.imshow("Motion Sign Collector", ui)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key == ord(' '):
            if state == "idle":
                recording  = []
                state      = "recording"
                status_msg = ""
                print("Recording started...")
            elif state == "recording":
                if len(recording) >= 10:
                    pending_seq = list(recording)
                    recording   = []
                    state       = "done"
                    status_msg  = "Press a key to label this sequence."
                    print(f"Recording stopped ({len(pending_seq)} frames).")
                else:
                    recording   = []
                    state       = "idle"
                    status_msg  = "Too short — try again."
        elif key == 8:  # BACKSPACE
            pending_seq = None
            state       = "idle"
            status_msg  = "Discarded."
            print("Sequence discarded.")
        elif state == "done" and pending_seq is not None:
            # Determine label
            label = None
            if 65 <= key <= 90 or 97 <= key <= 122:  # a-z / A-Z
                char = chr(key).upper()
                if char in ("J", "Z"):
                    label = char
                else:
                    status_msg = f"'{char}' is not a motion letter (only J, Z). Try again."
            elif 49 <= key <= 57:  # 1-9
                digit = chr(key)
                if digit in WORD_SIGNS:
                    label = WORD_SIGNS[digit]
                else:
                    status_msg = f"Key '{digit}' has no motion sign. Try again."

            if label is not None:
                # Pad or trim to SEQ_LENGTH
                seq = pending_seq[:SEQ_LENGTH]
                while len(seq) < SEQ_LENGTH:
                    seq.append(seq[-1])  # repeat last frame

                append_sequence(label, seq)
                count_map[label] = count_map.get(label, 0) + 1
                all_sequences.append((label, seq))

                last_label  = label
                pending_seq = None
                state       = "idle"
                status_msg  = f"Saved '{label}' ({count_map[label]} total)"
                print(f"  Saved: {label:16s} | Total: {count_map[label]}")

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

    print("\n=== Collection complete ===")
    for label, cnt in sorted(count_map.items()):
        print(f"  {label:16s}: {cnt} sequences")
    print(f"\nData saved to {DATA_FILE}")
    print("Next step: python train_motion_model.py")


if __name__ == "__main__":
    main()
