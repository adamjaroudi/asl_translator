"""
motion_translator.py
====================
Real-time translator for MOTION-BASED ASL signs.

Maintains a sliding window of SEQ_LENGTH frames. Once the window is full
and motion is detected (hand is moving), the sequence is classified.

Architecture:
  - MediaPipe Hand Landmarker (VIDEO mode, 60 fps)
  - Sliding frame buffer (SEQ_LENGTH = 30 frames)
  - Motion threshold gate (only classify when hand is moving)
  - LSTM / RandomForest motion classifier
  - Optional static classifier overlay (loads asl_classifier.pkl)

Controls:
  SPACE     — Add space to sentence
  BACKSPACE — Delete last character/word
  C         — Clear sentence
  ESC       — Quit
"""

import os
import sys
import pickle
import time
import collections
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_DIR        = "model"
TASK_FILE        = os.path.join(MODEL_DIR, "hand_landmarker.task")
MOTION_MODEL_FILE = os.path.join(MODEL_DIR, "motion_classifier.pkl")
STATIC_MODEL_FILE = os.path.join(MODEL_DIR, "asl_classifier.pkl")

# ── Constants ──────────────────────────────────────────────────────────────────
SEQ_LENGTH          = 30
FEATURE_DIM_RAW     = 126   # raw per-frame features (63 × 2 hands)
CONFIDENCE_THRESHOLD = 0.65
MOTION_THRESHOLD    = 0.015  # min mean delta magnitude to trigger classification
COOLDOWN_SECONDS    = 1.2   # seconds between consecutive sign additions
STABLE_FRAMES       = 8     # frames a sign must be stable before adding

ML_W, ML_H = 640, 480      # detection resolution

# ── Model loading ──────────────────────────────────────────────────────────────
def load_motion_model():
    if not os.path.exists(MOTION_MODEL_FILE):
        print(f"Motion model not found at {MOTION_MODEL_FILE}")
        print("Run: python generate_motion_dataset.py && python train_motion_model.py")
        sys.exit(1)
    with open(MOTION_MODEL_FILE, "rb") as f:
        payload = pickle.load(f)
    print(f"Loaded motion model ({payload['model_type']}) | acc={payload.get('accuracy', '?'):.2%}")
    print(f"  Signs: {payload['classes']}")
    return payload


def load_static_model():
    if not os.path.exists(STATIC_MODEL_FILE):
        return None
    with open(STATIC_MODEL_FILE, "rb") as f:
        clf = pickle.load(f)
    print(f"Loaded static model | classes: {list(clf.classes_[:5])}...")
    return clf


# ── MediaPipe setup ────────────────────────────────────────────────────────────
def download_task_file():
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
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    return mp_vision.HandLandmarker.create_from_options(opts)


# ── Feature extraction ─────────────────────────────────────────────────────────
def normalize_hand(lms):
    pts = np.array([[l.x, l.y, l.z] for l in lms], dtype=np.float32)
    wrist = pts[0].copy()
    pts  -= wrist
    scale = np.linalg.norm(pts[9]) + 1e-6
    pts  /= scale
    return pts.flatten()  # 63


def extract_frame_features(hand_landmarks_list):
    h1 = np.zeros(63, dtype=np.float32)
    h2 = np.zeros(63, dtype=np.float32)
    if hand_landmarks_list:
        h1 = normalize_hand(hand_landmarks_list[0])
        if len(hand_landmarks_list) > 1:
            h2 = normalize_hand(hand_landmarks_list[1])
    return np.concatenate([h1, h2])  # 126


def engineer_sequence(seq_arr):
    """
    seq_arr: (SEQ_LENGTH, 126)
    Returns: (1, SEQ_LENGTH, 253) engineered tensor.
    """
    X = seq_arr[np.newaxis]        # (1, T, 126)
    deltas = np.diff(X, axis=1, prepend=X[:, :1, :])  # (1, T, 126)
    mags   = np.linalg.norm(deltas, axis=2, keepdims=True)  # (1, T, 1)
    return np.concatenate([X, deltas, mags], axis=2).astype(np.float32)  # (1, T, 253)


# ── Motion detection ───────────────────────────────────────────────────────────
def compute_motion(buffer):
    """Return mean magnitude of frame-to-frame differences in buffer."""
    if len(buffer) < 2:
        return 0.0
    arr = np.array(list(buffer))     # (T, 126)
    deltas = np.diff(arr, axis=0)    # (T-1, 126)
    return float(np.mean(np.linalg.norm(deltas, axis=1)))


# ── Static feature extraction (for static model) ──────────────────────────────
def extract_static_features(hand_landmarks_list, handedness_list=None):
    """Mirror of the static translator's extract_features."""
    NUM_LM = 21
    def norm_hand(lms):
        pts = np.array([[l.x, l.y, l.z] for l in lms], dtype=np.float32)
        pts -= pts[0]
        scale = np.linalg.norm(pts[9]) + 1e-6
        pts  /= scale
        return pts.flatten()

    h1_feats = np.zeros(63, dtype=np.float32)
    h2_feats = np.zeros(63, dtype=np.float32)
    rel_pos  = np.zeros(3,  dtype=np.float32)
    num_hands = 0

    if hand_landmarks_list:
        num_hands = len(hand_landmarks_list)
        h1_feats  = norm_hand(hand_landmarks_list[0])
        if num_hands > 1:
            h2_feats = norm_hand(hand_landmarks_list[1])
            w1 = np.array([hand_landmarks_list[0][0].x,
                            hand_landmarks_list[0][0].y,
                            hand_landmarks_list[0][0].z])
            w2 = np.array([hand_landmarks_list[1][0].x,
                            hand_landmarks_list[1][0].y,
                            hand_landmarks_list[1][0].z])
            rel_pos = w2 - w1

    return np.concatenate([h1_feats, h2_feats, rel_pos, [num_hands]]).reshape(1, -1)


# ── Prediction ─────────────────────────────────────────────────────────────────
def classify_motion(seq_buffer, payload):
    """Classify a full motion sequence. Returns (label, confidence)."""
    arr   = np.array(list(seq_buffer), dtype=np.float32)  # (30, 126)
    X_eng = engineer_sequence(arr)                          # (1, 30, 253)

    model      = payload["model"]
    model_type = payload["model_type"]

    if model_type == "LSTM":
        proba = model.predict_proba(X_eng)[0]
    else:
        # RF: flatten
        X_flat = X_eng.reshape(1, -1)
        proba  = model.predict_proba(X_flat)[0]

    classes = payload["classes"]
    idx     = int(np.argmax(proba))
    return classes[idx], float(proba[idx])


# ── Display helpers ────────────────────────────────────────────────────────────
def draw_landmarks(frame, hand_landmarks_list):
    if not hand_landmarks_list:
        return
    h, w = frame.shape[:2]
    for lms in hand_landmarks_list:
        pts = [(int(l.x * w), int(l.y * h)) for l in lms]
        for i, j in mp.solutions.hands.HAND_CONNECTIONS:
            cv2.line(frame, pts[i], pts[j], (0, 255, 100), 2)
        for pt in pts:
            cv2.circle(frame, pt, 4, (255, 255, 255), -1)


def draw_buffer_bar(frame, buffer_len, motion, is_classifying):
    h, w = frame.shape[:2]
    pct = buffer_len / SEQ_LENGTH
    bar_w = int(300 * pct)
    col   = (0, 200, 50) if not is_classifying else (0, 100, 255)
    cv2.rectangle(frame, (10, h-40), (310, h-20), (50, 50, 50), -1)
    cv2.rectangle(frame, (10, h-40), (10 + bar_w, h-20), col, -1)
    cv2.rectangle(frame, (10, h-40), (310, h-20), (200, 200, 200), 1)
    motion_pct = min(motion / (MOTION_THRESHOLD * 3), 1.0)
    motion_bar = int(100 * motion_pct)
    m_col = (0, 255, 255) if motion > MOTION_THRESHOLD else (100, 100, 100)
    cv2.rectangle(frame, (320, h-40), (420, h-20), (50, 50, 50), -1)
    cv2.rectangle(frame, (320, h-40), (320 + motion_bar, h-20), m_col, -1)
    cv2.putText(frame, f"Buffer {buffer_len}/{SEQ_LENGTH}", (10, h-45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
    cv2.putText(frame, "Motion", (320, h-45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)


def draw_hud(frame, sentence, motion_sign, motion_conf, static_sign, static_conf, fps, mode):
    h, w = frame.shape[:2]

    # Sentence bar at bottom
    cv2.rectangle(frame, (0, h-90), (w, h-50), (20, 20, 20), -1)
    cv2.putText(frame, sentence if sentence else "[ Start signing ]",
                (10, h-62), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Sign prediction overlay
    cv2.rectangle(frame, (0, 0), (w, 60), (20, 20, 40), -1)

    if motion_sign:
        conf_col = (0, 220, 80) if motion_conf >= CONFIDENCE_THRESHOLD else (50, 150, 255)
        txt = f"Motion: {motion_sign}  ({motion_conf:.0%})"
        cv2.putText(frame, txt, (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, conf_col, 2)
    else:
        cv2.putText(frame, "Waiting for motion...", (10, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 1)

    if static_sign:
        cv2.putText(frame, f"Static: {static_sign} ({static_conf:.0%})",
                    (w-280, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 255), 1)

    cv2.putText(frame, f"FPS:{fps:.0f}", (w-80, h-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
    cv2.putText(frame, f"Mode: {mode}", (10, h-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)


def format_label(label):
    """Convert [THANK_YOU] → 'THANK YOU', J → 'J'."""
    if label.startswith("[") and label.endswith("]"):
        return label[1:-1].replace("_", " ")
    return label


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    payload      = load_motion_model()
    static_clf   = load_static_model()
    landmarker   = make_landmarker()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Sliding window buffer
    frame_buffer = collections.deque(maxlen=SEQ_LENGTH)

    sentence       = ""
    motion_sign    = ""
    motion_conf    = 0.0
    static_sign    = ""
    static_conf    = 0.0
    stable_count   = 0
    last_sign      = ""
    last_add_time  = 0.0
    prev_motion    = ""

    fps = 0.0
    fps_count = 0
    fps_t0 = time.time()
    frame_ts = 0
    frame_idx = 0
    mode = "MOTION"
    DETECTION_INTERVAL = 2

    print("\n=== Motion ASL Translator ===")
    print("  SPACE     — Add space")
    print("  BACKSPACE — Delete last")
    print("  C         — Clear")
    print("  ESC       — Quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.flip(frame, 1, dst=frame)
        frame_idx += 1
        fps_count += 1

        # FPS
        if time.time() - fps_t0 >= 1.0:
            fps = fps_count / (time.time() - fps_t0)
            fps_count = 0
            fps_t0 = time.time()

        # Detection
        if frame_idx % DETECTION_INTERVAL == 0:
            frame_ts += DETECTION_INTERVAL * 16
            small = cv2.resize(frame, (ML_W, ML_H))
            rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = landmarker.detect_for_video(mp_img, frame_ts)

            lms_list = results.hand_landmarks if results.hand_landmarks else []

            feats = extract_frame_features(lms_list)
            frame_buffer.append(feats)

            # Static sign prediction (always on)
            if lms_list and static_clf is not None:
                sf = extract_static_features(lms_list)
                sp = static_clf.predict_proba(sf)[0]
                si = int(np.argmax(sp))
                static_sign = static_clf.classes_[si]
                static_conf = float(sp[si])

            # Motion classification (only when buffer full and motion detected)
            motion_val = compute_motion(frame_buffer)
            is_classifying = False

            if len(frame_buffer) == SEQ_LENGTH and motion_val > MOTION_THRESHOLD:
                is_classifying = True
                pred, conf = classify_motion(frame_buffer, payload)
                motion_sign = pred
                motion_conf = conf

                if pred == prev_motion and conf >= CONFIDENCE_THRESHOLD:
                    stable_count += 1
                else:
                    stable_count = 0

                now = time.time()
                if (
                    stable_count >= STABLE_FRAMES
                    and pred != last_sign
                    and (now - last_add_time) >= COOLDOWN_SECONDS
                    and conf >= CONFIDENCE_THRESHOLD
                ):
                    word = format_label(pred)
                    sentence    += word + " "
                    last_sign    = pred
                    last_add_time = now
                    stable_count = 0
                    print(f"  Added: {word}")

                prev_motion = pred
            elif motion_val <= MOTION_THRESHOLD:
                # Hand is still — reset stability
                stable_count = 0
                prev_motion  = ""

            draw_landmarks(frame, lms_list)
            draw_buffer_bar(frame, len(frame_buffer), motion_val, is_classifying)

        draw_hud(frame, sentence.strip(), motion_sign, motion_conf,
                 static_sign, static_conf, fps, mode)

        cv2.imshow("Motion ASL Translator", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key == ord(' '):
            sentence += " "
        elif key == 8:  # BACKSPACE
            parts = sentence.rstrip().rsplit(" ", 1)
            sentence = parts[0] + " " if len(parts) > 1 else ""
        elif key == ord('c') or key == ord('C'):
            sentence = ""

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print("\nBye!")


if __name__ == "__main__":
    main()
