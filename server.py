"""
ASL WebSocket Server

Two endpoints:
  ws://localhost:8765  — translator (prediction + sentence building)
  ws://localhost:8766  — data collector (record + save landmarks)

Install:
    pip install websockets

Usage:
    python server.py
"""

import asyncio
import json
import pickle
import time
import os
import csv
import base64

import cv2
import mediapipe as mp
import numpy as np
import websockets

from features import engineer_features_from_raw
from camera_util import open_camera as open_camera_impl

MODEL_FILE  = os.path.join("model", "asl_classifier.pkl")
HAND_MODEL  = os.path.join("model", "hand_landmarker.task")
DATA_DIR    = "data"
DATA_FILE   = os.path.join(DATA_DIR, "landmarks.csv")

NUM_LANDMARKS     = 21
FEATURES_PER_HAND = NUM_LANDMARKS * 3
CONFIDENCE_THRESHOLD = 0.6
STABLE_FRAMES        = 12
COOLDOWN_SECONDS     = 1.0
ML_W, ML_H = 480, 360
DETECTION_INTERVAL = 2

HAND_CONNECTIONS  = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
FINGERTIP_INDICES = {4, 8, 12, 16, 20}
HAND_COLORS = [(255, 160, 50), (50, 200, 255)]
GREEN       = (100, 220, 100)
TEXT_WHITE  = (255, 255, 255)
BG_DARK     = (30, 30, 30)


# ── Shared helpers ────────────────────────────────────────────────────────────

def ensure_hand_model():
    if not os.path.exists(HAND_MODEL):
        import urllib.request
        os.makedirs("model", exist_ok=True)
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/latest/hand_landmarker.task",
            HAND_MODEL,
        )


def make_landmarker():
    ensure_hand_model()
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def normalize_landmarks(landmarks):
    coords  = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    coords -= coords[0]
    md = np.max(np.linalg.norm(coords, axis=1))
    if md > 0:
        coords /= md
    return coords.flatten().tolist()


def raw_features_from_landmarks(all_hand_landmarks):
    """Return the 130-dim raw feature vector (no engineering)."""
    hands = sorted(all_hand_landmarks, key=lambda lm: lm[0].x)
    h1_lm = hands[0]
    h1f   = normalize_landmarks(h1_lm)
    h1w   = np.array([h1_lm[0].x, h1_lm[0].y, h1_lm[0].z])
    if len(hands) >= 2:
        h2_lm = hands[1]
        h2f   = normalize_landmarks(h2_lm)
        h2w   = np.array([h2_lm[0].x, h2_lm[0].y, h2_lm[0].z])
        rel, nh = (h2w - h1w).tolist(), 1.0
    else:
        h2f = [0.0] * FEATURES_PER_HAND
        rel, nh = [0.0, 0.0, 0.0], 0.0
    return h1f + h2f + rel + [nh]


def engineered_features_from_landmarks(all_hand_landmarks):
    raw = raw_features_from_landmarks(all_hand_landmarks)
    return np.array(engineer_features_from_raw(raw), dtype=np.float32).reshape(1, -1)


def draw_skeleton(frame, landmarks, color):
    h, w = frame.shape[:2]
    for conn in HAND_CONNECTIONS:
        s, e = conn.start, conn.end
        cv2.line(frame,
                 (int(landmarks[s].x * w), int(landmarks[s].y * h)),
                 (int(landmarks[e].x * w), int(landmarks[e].y * h)),
                 color, 2, cv2.LINE_AA)
    for i, lm in enumerate(landmarks):
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 5,
                   GREEN if i in FINGERTIP_INDICES else TEXT_WHITE, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 5, BG_DARK, 1, cv2.LINE_AA)


def encode_frame(frame, quality=55):
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("ascii")


def format_label(label):
    if label.startswith("[") and label.endswith("]"):
        return label[1:-1].replace("-", " ").title()
    return label


def sentence_token(label):
    if label.startswith("[") and label.endswith("]"):
        return f" {label[1:-1].replace('-', ' ').lower()} "
    return label


def open_camera():
    cap, ok = open_camera_impl(0)
    if not ok or cap is None:
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS,          60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    return cap


def make_csv_header():
    cols = ["label"]
    for prefix in ("h1", "h2"):
        for i in range(NUM_LANDMARKS):
            for axis in ("x", "y", "z"):
                cols.append(f"{prefix}_{axis}{i}")
    cols += ["rel_x", "rel_y", "rel_z", "num_hands"]
    return cols


# ── Translator handler (port 8765) ────────────────────────────────────────────

async def translator_handler(websocket):
    print(f"[translator] Client connected: {websocket.remote_address}")

    if not os.path.exists(MODEL_FILE):
        await websocket.send(json.dumps({"error": "Model not found. Run train_model.py first."}))
        return

    with open(MODEL_FILE, "rb") as f:
        clf = pickle.load(f)

    landmarker = make_landmarker()
    cap = open_camera()
    if cap is None:
        await websocket.send(json.dumps({"error": "Cannot open webcam."}))
        landmarker.close()
        return

    sentence = ""
    predicted_sign = prev_prediction = last_sign = ""
    confidence = stable_count = 0.0
    last_add_time = 0.0
    top_alts = []
    cached_lm = None
    frame_ts = frame_idx = 0
    fps = fps_count = 0.0
    fps_timer = time.time()
    small = np.empty((ML_H, ML_W, 3), dtype=np.uint8)

    async def listen():
        nonlocal sentence, last_sign
        async for msg in websocket:
            try:
                cmd = json.loads(msg)
                if cmd.get("action") == "space":
                    sentence += " "; last_sign = ""
                elif cmd.get("action") == "backspace":
                    sentence = sentence.rstrip()
                    if sentence and sentence[-1] == " ":
                        sentence = sentence[:-1].rstrip()
                        while sentence and sentence[-1] != " ":
                            sentence = sentence[:-1]
                    elif sentence:
                        sentence = sentence[:-1]
                    last_sign = ""
                elif cmd.get("action") == "clear":
                    sentence = ""; last_sign = ""
            except Exception:
                pass

    asyncio.ensure_future(listen())

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.flip(frame, 1, dst=frame)
            frame_idx += 1

            if frame_idx % DETECTION_INTERVAL == 0:
                cv2.resize(frame, (ML_W, ML_H), dst=small, interpolation=cv2.INTER_NEAREST)
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                frame_ts += 33
                results = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), frame_ts)

                if results.hand_landmarks:
                    cached_lm = results.hand_landmarks
                    features  = engineered_features_from_landmarks(cached_lm)
                    proba     = clf.predict_proba(features)[0]
                    idx       = int(np.argmax(proba))
                    cur_pred  = clf.classes_[idx]
                    cur_conf  = float(proba[idx])
                    predicted_sign = cur_pred
                    confidence     = cur_conf

                    # Top 5 alternatives for advanced panel
                    top_idx  = np.argsort(proba)[::-1][:6]
                    top_alts = [[clf.classes_[i], round(float(proba[i]), 3)] for i in top_idx if clf.classes_[i] != cur_pred][:5]

                    if cur_pred == prev_prediction and cur_conf >= CONFIDENCE_THRESHOLD:
                        stable_count += 1
                    else:
                        stable_count = 0

                    now = time.time()
                    if (stable_count >= STABLE_FRAMES and cur_pred != last_sign
                            and (now - last_add_time) >= COOLDOWN_SECONDS
                            and cur_conf >= CONFIDENCE_THRESHOLD):
                        sentence     += sentence_token(cur_pred)
                        last_sign     = cur_pred
                        last_add_time = now
                        stable_count  = 0
                    prev_prediction = cur_pred
                else:
                    cached_lm = None
                    predicted_sign = prev_prediction = last_sign = ""
                    confidence = stable_count = 0.0

            if cached_lm:
                for i, lm in enumerate(cached_lm):
                    draw_skeleton(frame, lm, HAND_COLORS[min(i, 1)])

            fps_count += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                fps = fps_count / (now - fps_timer)
                fps_count = 0
                fps_timer = now

            await websocket.send(json.dumps({
                "frame":      encode_frame(frame),
                "prediction": format_label(predicted_sign) if predicted_sign else "",
                "raw_label":  predicted_sign,
                "confidence": round(confidence, 3),
                "is_word":    predicted_sign.startswith("["),
                "sentence":   sentence.strip(),
                "num_hands":  len(cached_lm) if cached_lm else 0,
                "stable_pct": round(min(stable_count / STABLE_FRAMES, 1.0), 2),
                "fps":        round(fps, 1),
                "top_alts":   top_alts if predicted_sign else [],
            }))
            await asyncio.sleep(0.001)

    except websockets.exceptions.ConnectionClosed:
        print("[translator] Client disconnected.")
    finally:
        cap.release()
        landmarker.close()


# ── Data collection handler (port 8766) ───────────────────────────────────────

async def collector_handler(websocket):
    print(f"[collector] Client connected: {websocket.remote_address}")

    landmarker = make_landmarker()
    cap = open_camera()
    if cap is None:
        await websocket.send(json.dumps({"error": "Cannot open webcam."}))
        landmarker.close()
        return

    # Load existing data
    os.makedirs(DATA_DIR, exist_ok=True)
    existing_rows = []
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            existing_rows = list(csv.reader(f))

    # Build class stats from existing data
    class_stats = {}
    for row in existing_rows[1:]:  # skip header
        if row:
            class_stats[row[0]] = class_stats.get(row[0], 0) + 1

    new_samples  = []   # rows collected this session
    recording    = False
    current_label = ""
    session_count = 0
    frame_ts = frame_idx = 0
    cached_lm = None
    small = np.empty((ML_H, ML_W, 3), dtype=np.uint8)
    fps = fps_count = 0.0
    fps_timer = time.time()

    async def listen():
        nonlocal recording, current_label, session_count
        async for msg in websocket:
            try:
                cmd = json.loads(msg)
                action = cmd.get("action")

                if action == "start_recording":
                    current_label = cmd.get("label", "")
                    session_count = 0
                    recording     = True
                    print(f"[collector] Recording '{current_label}'")

                elif action == "stop_recording":
                    recording = False
                    print(f"[collector] Stopped. Session samples: {session_count}")

                elif action == "save":
                    header = make_csv_header()
                    all_rows = existing_rows if existing_rows else [header]
                    if not existing_rows:
                        all_rows = [header]
                    all_rows = all_rows + new_samples
                    with open(DATA_FILE, "w", newline="") as f:
                        csv.writer(f).writerows(all_rows)
                    total = len(all_rows) - 1
                    print(f"[collector] Saved {len(new_samples)} new samples ({total} total)")
                    await websocket.send(json.dumps({
                        "saved": len(new_samples),
                        "path":  DATA_FILE,
                        "total_samples": total,
                    }))

            except Exception as ex:
                print(f"[collector] Command error: {ex}")

    asyncio.ensure_future(listen())

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.flip(frame, 1, dst=frame)
            frame_idx += 1

            if frame_idx % DETECTION_INTERVAL == 0:
                cv2.resize(frame, (ML_W, ML_H), dst=small, interpolation=cv2.INTER_NEAREST)
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                frame_ts += 33
                results = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), frame_ts)

                if results.hand_landmarks:
                    cached_lm = results.hand_landmarks
                    if recording and current_label:
                        raw = raw_features_from_landmarks(cached_lm)
                        new_samples.append([current_label] + raw)
                        session_count += 1
                        class_stats[current_label] = class_stats.get(current_label, 0) + 1
                else:
                    cached_lm = None

            if cached_lm:
                for i, lm in enumerate(cached_lm):
                    draw_skeleton(frame, lm, HAND_COLORS[min(i, 1)])

            fps_count += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                fps = fps_count / (now - fps_timer)
                fps_count = 0
                fps_timer = now

            total_samples = (len(existing_rows) - 1 if existing_rows else 0) + len(new_samples)

            await websocket.send(json.dumps({
                "frame":         encode_frame(frame),
                "num_hands":     len(cached_lm) if cached_lm else 0,
                "recording":     recording,
                "current_label": current_label,
                "sample_count":  session_count,
                "total_samples": total_samples,
                "class_stats":   class_stats,
                "fps":           round(fps, 1),
            }))
            await asyncio.sleep(0.001)

    except websockets.exceptions.ConnectionClosed:
        print("[collector] Client disconnected.")
    finally:
        cap.release()
        landmarker.close()


# ── Entry point ───────────────────────────────────────────────────────────────

async def main():
    print("ASL server starting…")
    print("  Translator : ws://localhost:8765")
    print("  Collector  : ws://localhost:8766")

    async with (
        websockets.serve(translator_handler, "localhost", 8765),
        websockets.serve(collector_handler,  "localhost", 8766),
    ):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())