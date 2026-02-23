"""
Real-time ASL Translator — Letters + Words, Both Hands.

Uses your webcam + MediaPipe hand tracking + a trained LinearSVC classifier.
Inference is <0.1 ms per frame so everything runs on the main thread cleanly.

Controls:
    SPACE      Add a space to the sentence
    BACKSPACE  Delete last character/word
    C          Clear the entire sentence
    ESC        Quit
"""

import argparse
import os
import pickle
import time
import cv2
import mediapipe as mp
import numpy as np

from features import engineer_features_from_raw
from camera_util import open_camera

MODEL_FILE = os.path.join("model", "asl_classifier.pkl")
HAND_MODEL = os.path.join("model", "hand_landmarker.task")

NUM_LANDMARKS     = 21
FEATURES_PER_HAND = NUM_LANDMARKS * 3

CONFIDENCE_THRESHOLD = 0.72   # Only add sign when model is fairly confident
CONFIDENCE_MARGIN    = 0.10   # Top prediction must beat second-best by this much (reduces confusion)
STABLE_FRAMES        = 14     # Hold sign steady a bit longer before adding
COOLDOWN_SECONDS     = 1.2

# Only run detection every N frames (2 = good balance of speed vs responsiveness)
DETECTION_INTERVAL = 2

# Resolution sent to MediaPipe — smaller = faster, landmarks are 0-1 normalised
# so drawing still works on full-res display frame
ML_W, ML_H = 480, 360

# UI Colors (BGR)
BG_DARK    = (30, 30, 30)
BG_PANEL   = (45, 45, 45)
ACCENT     = (255, 160, 50)
ACCENT2    = (50, 200, 255)
TEXT_WHITE = (255, 255, 255)
TEXT_DIM   = (160, 160, 160)
GREEN      = (100, 220, 100)
RED        = (80, 80, 255)
YELLOW     = (60, 220, 255)

HAND_CONNECTIONS  = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
FINGERTIP_INDICES = {4, 8, 12, 16, 20}
HAND_COLORS       = [ACCENT, ACCENT2]


# ── Feature helpers ────────────────────────────────────────────────────────────

def normalize_landmarks(landmarks):
    coords  = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    coords -= coords[0]
    md = np.max(np.linalg.norm(coords, axis=1))
    if md > 0:
        coords /= md
    return coords.flatten().tolist()


def extract_features(all_hand_landmarks):
    hands = sorted(all_hand_landmarks, key=lambda lm: lm[0].x)
    h1_lm = hands[0]
    h1f   = normalize_landmarks(h1_lm)
    h1w   = np.array([h1_lm[0].x, h1_lm[0].y, h1_lm[0].z])

    if len(hands) >= 2:
        h2_lm = hands[1]
        h2f   = normalize_landmarks(h2_lm)
        h2w   = np.array([h2_lm[0].x, h2_lm[0].y, h2_lm[0].z])
        rel   = (h2w - h1w).tolist()
        nh    = 1.0
    else:
        h2f = [0.0] * FEATURES_PER_HAND
        rel = [0.0, 0.0, 0.0]
        nh  = 0.0

    raw = h1f + h2f + rel + [nh]
    return np.array(engineer_features_from_raw(raw), dtype=np.float32).reshape(1, -1)


# ── Label formatting ───────────────────────────────────────────────────────────

def format_prediction(label):
    if label.startswith("[") and label.endswith("]"):
        return label[1:-1].replace("-", " ").title()
    return label


def prediction_for_sentence(label):
    if label.startswith("[") and label.endswith("]"):
        return f" {label[1:-1].replace('-', ' ').lower()} "
    return label


# ── Drawing ────────────────────────────────────────────────────────────────────

def draw_hand_landmarks(img, landmarks, color):
    h, w = img.shape[:2]
    for conn in HAND_CONNECTIONS:
        s, e = conn.start, conn.end
        cv2.line(img,
                 (int(landmarks[s].x * w), int(landmarks[s].y * h)),
                 (int(landmarks[e].x * w), int(landmarks[e].y * h)),
                 color, 2, cv2.LINE_AA)
    for i, lm in enumerate(landmarks):
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 5, GREEN if i in FINGERTIP_INDICES else TEXT_WHITE, -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 5, BG_DARK, 1, cv2.LINE_AA)


def create_ui(frame, prediction, confidence, sentence, num_hands, fps, top_alternatives=None):
    h, w = frame.shape[:2]

    cv2.rectangle(frame, (0, 0), (w, 70), BG_DARK, -1)
    cv2.putText(frame, "ASL TRANSLATOR", (15, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, ACCENT, 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 95, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_DIM, 1, cv2.LINE_AA)

    hand_text  = f"{num_hands} Hand{'s' if num_hands != 1 else ''}" if num_hands else "Show Hands"
    hand_color = GREEN if num_hands else YELLOW
    cv2.circle(frame, (w - 180, 52), 6, hand_color, -1)
    cv2.putText(frame, hand_text, (w - 165, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, hand_color, 1, cv2.LINE_AA)

    panel_y = h - 170
    cv2.rectangle(frame, (0, panel_y), (w, h), BG_DARK, -1)

    if prediction and confidence > 0:
        top_alternatives = top_alternatives or []
        display     = format_prediction(prediction)
        is_word     = prediction.startswith("[")
        badge_color = ACCENT2 if is_word else ACCENT
        cv2.putText(frame, "WORD" if is_word else "LETTER",
                    (15, panel_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, badge_color, 1, cv2.LINE_AA)
        fs = 1.2 if len(display) > 3 else 1.8
        cv2.putText(frame, display, (15, panel_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, badge_color, 3, cv2.LINE_AA)
        # Show "or: N? S?" when model is unsure between letters (helps you see confused letters)
        if not is_word and top_alternatives and top_alternatives[0][1] >= 0.08:
            or_letters = " or ".join(format_prediction(l) for l, p in top_alternatives[:2] if p >= 0.08 and not l.startswith("["))
            if or_letters:
                cv2.putText(frame, f"or: {or_letters}?", (15, panel_y + 78),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_DIM, 1, cv2.LINE_AA)
        tw  = cv2.getTextSize(display, cv2.FONT_HERSHEY_SIMPLEX, fs, 3)[0][0]
        bx  = tw + 35
        by  = panel_y + 44
        if bx + 180 < w:
            cv2.rectangle(frame, (bx, by), (bx + 120, by + 16), BG_PANEL, -1)
            cv2.rectangle(frame, (bx, by), (bx + int(120 * confidence), by + 16),
                          GREEN if confidence >= CONFIDENCE_THRESHOLD else RED, -1)
            cv2.putText(frame, f"{confidence:.0%}", (bx + 128, by + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_DIM, 1, cv2.LINE_AA)

    cv2.putText(frame, "Sentence:", (15, panel_y + 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_DIM, 1, cv2.LINE_AA)
    disp_s = sentence.strip() or "..."
    if len(disp_s) > 55:
        disp_s = "..." + disp_s[-52:]
    cv2.putText(frame, disp_s, (15, panel_y + 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_WHITE, 2, cv2.LINE_AA)
    cv2.putText(frame, "SPACE:space | BACKSPACE:delete | C:clear | ESC:quit",
                (15, panel_y + 155), cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_DIM, 1, cv2.LINE_AA)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Real-time ASL translator.")
    parser.add_argument("--camera", type=int, default=None, help="Camera index (0, 1, 2...). Or set CAMERA_INDEX env.")
    args = parser.parse_args()

    if not os.path.exists(MODEL_FILE):
        print(f"Error: Model not found at {MODEL_FILE}")
        print("Run: 1. python generate_dataset.py  2. python train_model.py")
        return

    if not os.path.exists(HAND_MODEL):
        print("Downloading hand landmarker model...")
        import urllib.request
        os.makedirs("model", exist_ok=True)
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/latest/hand_landmarker.task",
            HAND_MODEL,
        )

    print("Loading models...")
    with open(MODEL_FILE, "rb") as f:
        clf = pickle.load(f)

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
    print("Models loaded.\n")

    cap, ok = open_camera(args.camera)
    if not ok or cap is None:
        print("Error: Cannot open webcam. Check Windows Settings → Privacy → Camera.")
        landmarker.close()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS,          60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    sentence        = ""
    predicted_sign  = ""
    confidence      = 0.0
    top_alternatives = []   # [(label, prob), ...] for 2nd/3rd best (so you can see confused letters)
    stable_count    = 0
    last_sign       = ""
    last_add_time   = 0.0
    prev_prediction = ""
    cached_lm       = None   # last known landmarks (for drawing between detections)

    fps = 0.0
    fps_count = 0
    fps_timer = time.time()
    frame_ts  = 0
    frame_idx = 0

    # Pre-allocate small frame buffer
    small = np.empty((ML_H, ML_W, 3), dtype=np.uint8)

    print("=== ASL Translator Running ===")
    print("Controls: SPACE=space, BACKSPACE=delete, C=clear, ESC=quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.flip(frame, 1, dst=frame)
        frame_idx += 1

        # ── Run detection every DETECTION_INTERVAL frames ──────────────
        if frame_idx % DETECTION_INTERVAL == 0:
            cv2.resize(frame, (ML_W, ML_H), dst=small, interpolation=cv2.INTER_NEAREST)
            rgb      = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            frame_ts += 33
            mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results  = landmarker.detect_for_video(mp_img, frame_ts)

            if results.hand_landmarks:
                cached_lm = results.hand_landmarks

                features     = extract_features(cached_lm)
                proba        = clf.predict_proba(features)[0]
                idx          = int(np.argmax(proba))
                current_pred = clf.classes_[idx]
                current_conf = float(proba[idx])
                # Require clear winner (top prediction ahead of second-best)
                sorted_proba = np.sort(proba)[::-1]
                margin_ok = len(sorted_proba) < 2 or (sorted_proba[0] - sorted_proba[1]) >= CONFIDENCE_MARGIN

                predicted_sign = current_pred
                confidence     = current_conf
                # Top 2 alternatives (so you can see e.g. "model is split between M and N")
                order = np.argsort(proba)[::-1]
                top_alternatives = [(clf.classes_[i], float(proba[i])) for i in order[1:4] if proba[i] >= 0.05]

                if current_pred == prev_prediction and current_conf >= CONFIDENCE_THRESHOLD and margin_ok:
                    stable_count += 1
                else:
                    stable_count = 0

                now = time.time()
                if (
                    stable_count >= STABLE_FRAMES
                    and current_pred != last_sign
                    and (now - last_add_time) >= COOLDOWN_SECONDS
                    and current_conf >= CONFIDENCE_THRESHOLD
                    and margin_ok
                ):
                    sentence     += prediction_for_sentence(current_pred)
                    last_sign     = current_pred
                    last_add_time = now
                    stable_count  = 0
                    alt_str = ""
                    if top_alternatives:
                        alt_str = "  (also: " + ", ".join(f"{format_prediction(l)} {p:.0%}" for l, p in top_alternatives[:3]) + ")"
                    print(f"  Added: {format_prediction(current_pred):18s} |  {sentence.strip()}{alt_str}")

                prev_prediction = current_pred
            else:
                cached_lm       = None
                predicted_sign  = ""
                confidence      = 0.0
                top_alternatives = []
                stable_count    = 0
                prev_prediction = ""
                last_sign       = ""

        # ── Draw skeleton from cached landmarks ─────────────────────────
        if cached_lm:
            for i, lm in enumerate(cached_lm):
                draw_hand_landmarks(frame, lm, HAND_COLORS[min(i, 1)])

        num_hands = len(cached_lm) if cached_lm else 0

        # ── FPS ──────────────────────────────────────────────────────────
        fps_count += 1
        now = time.time()
        if now - fps_timer >= 1.0:
            fps       = fps_count / (now - fps_timer)
            fps_count = 0
            fps_timer = now

        create_ui(frame, predicted_sign, confidence, sentence, num_hands, fps, top_alternatives)
        cv2.imshow("ASL Translator", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == 32:
            sentence += " "
            last_sign = ""
            print(f"  [space]  |  {sentence.strip()}")
        elif key in (8, 127):
            sentence = sentence.rstrip()
            if sentence and sentence[-1] == " ":
                sentence = sentence[:-1].rstrip()
                while sentence and sentence[-1] != " ":
                    sentence = sentence[:-1]
            elif sentence:
                sentence = sentence[:-1]
            last_sign = ""
            print(f"  [delete] |  {sentence.strip()}")
        elif key in (ord("c"), ord("C")):
            sentence  = ""
            last_sign = ""
            print("  [cleared]")

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print(f"\nFinal sentence: {sentence.strip()}")


if __name__ == "__main__":
    main()