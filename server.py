"""
ASL WebSocket Server — Static + Motion Integrated

Endpoints:
  ws://localhost:8765  — translator     (static classifier, single-frame)
  ws://localhost:8766  — data collector (record + save landmarks)
  ws://localhost:8767  — restart trigger
  ws://localhost:8768  — motion translator (sequence-based, sliding window)
  ws://localhost:8769  — motion data collector (record sequences)

Install:
    pip install websockets

Usage:
    python server.py
"""

import asyncio
import collections
import csv
import json
import os
import pickle
import time
import base64

import cv2
import mediapipe as mp
import numpy as np
import websockets

from features import engineer_features_from_raw
from camera_util import open_camera as open_camera_impl

# ── Model paths ───────────────────────────────────────────────────────────────
MODEL_FILE        = os.path.join("model", "asl_classifier.pkl")
MOTION_MODEL_FILE = os.path.join("model", "motion_classifier.pkl")
HAND_MODEL        = os.path.join("model", "hand_landmarker.task")
DATA_DIR          = "data"
DATA_FILE         = os.path.join(DATA_DIR, "landmarks.csv")
MOTION_DATA_FILE  = os.path.join(DATA_DIR, "motion_landmarks.csv")

# ── Shared constants ──────────────────────────────────────────────────────────
NUM_LANDMARKS        = 21
FEATURES_PER_HAND    = NUM_LANDMARKS * 3
CONFIDENCE_THRESHOLD = 0.60
STABLE_FRAMES        = 12
COOLDOWN_SECONDS     = 1.0
ML_W, ML_H           = 480, 360
DETECTION_INTERVAL   = 2

# Motion constants
SEQ_LENGTH           = 30
MOTION_CONFIDENCE    = 0.65
MOTION_THRESHOLD     = 0.015
MOTION_STABLE_FRAMES = 8
MOTION_COOLDOWN      = 1.2

HAND_CONNECTIONS  = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
FINGERTIP_INDICES = {4, 8, 12, 16, 20}
HAND_COLORS       = [(255, 160, 50), (50, 200, 255)]
GREEN             = (100, 220, 100)
TEXT_WHITE        = (255, 255, 255)
BG_DARK           = (30, 30, 30)


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


def make_landmarker(detection_conf=0.7):
    ensure_hand_model()
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=detection_conf,
        min_hand_presence_confidence=detection_conf,
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


def extract_motion_frame_features(hand_landmarks_list):
    """Extract 126-dim motion features from one frame."""
    def norm_hand(lms):
        pts   = np.array([[l.x, l.y, l.z] for l in lms], dtype=np.float32)
        pts  -= pts[0]
        scale = np.linalg.norm(pts[9]) + 1e-6
        pts  /= scale
        return pts.flatten()

    h1 = np.zeros(63, dtype=np.float32)
    h2 = np.zeros(63, dtype=np.float32)
    if hand_landmarks_list:
        h1 = norm_hand(hand_landmarks_list[0])
        if len(hand_landmarks_list) > 1:
            h2 = norm_hand(hand_landmarks_list[1])
    return np.concatenate([h1, h2])


def engineer_sequence(seq_arr):
    """(SEQ_LENGTH, 126) → (1, SEQ_LENGTH, 253)"""
    X      = seq_arr[np.newaxis]
    deltas = np.diff(X, axis=1, prepend=X[:, :1, :])
    mags   = np.linalg.norm(deltas, axis=2, keepdims=True)
    return np.concatenate([X, deltas, mags], axis=2).astype(np.float32)


def compute_motion(buffer):
    if len(buffer) < 2:
        return 0.0
    arr    = np.array(list(buffer))
    deltas = np.diff(arr, axis=0)
    return float(np.mean(np.linalg.norm(deltas, axis=1)))


def classify_motion_seq(buffer, payload):
    arr   = np.array(list(buffer), dtype=np.float32)
    X_eng = engineer_sequence(arr)
    model      = payload["model"]
    model_type = payload["model_type"]
    if model_type == "LSTM":
        proba = model.predict_proba(X_eng)[0]
    else:
        proba = model.predict_proba(X_eng.reshape(1, -1))[0]
    classes = payload["classes"]
    idx     = int(np.argmax(proba))
    top     = np.argsort(proba)[::-1][:4]
    top_alts = [[classes[i], round(float(proba[i]), 3)] for i in top if classes[i] != classes[idx]][:3]
    return classes[idx], float(proba[idx]), top_alts


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
        return label[1:-1].replace("-", " ").replace("_", " ").title()
    return label


def sentence_token(label):
    if label.startswith("[") and label.endswith("]"):
        return f" {label[1:-1].replace('-', ' ').replace('_', ' ').lower()} "
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


def make_motion_csv_header():
    return ["label", "frame"] + [f"f{i}" for i in range(126)]


# ── Static translator handler (port 8765) ─────────────────────────────────────

async def translator_handler(websocket):
    print(f"[translator] Client connected: {websocket.remote_address}")

    if not os.path.exists(MODEL_FILE):
        await websocket.send(json.dumps({"error": "Static model not found. Run train_model.py first."}))
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


# ── Motion translator handler (port 8768) ─────────────────────────────────────

async def motion_translator_handler(websocket):
    print(f"[motion] Client connected: {websocket.remote_address}")

    if not os.path.exists(MOTION_MODEL_FILE):
        await websocket.send(json.dumps({
            "error": "Motion model not found. Run generate_motion_dataset.py then train_motion_model.py."
        }))
        return

    with open(MOTION_MODEL_FILE, "rb") as f:
        payload = pickle.load(f)
    print(f"[motion] Model loaded: {payload['model_type']} | classes: {payload['classes']}")

    # Also load static model for overlay
    static_clf = None
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            static_clf = pickle.load(f)

    landmarker = make_landmarker(detection_conf=0.5)
    cap = open_camera()
    if cap is None:
        await websocket.send(json.dumps({"error": "Cannot open webcam."}))
        landmarker.close()
        return

    frame_buffer  = collections.deque(maxlen=SEQ_LENGTH)
    sentence      = ""
    motion_sign   = ""
    motion_conf   = 0.0
    motion_alts   = []
    static_sign   = ""
    static_conf   = 0.0
    stable_count  = 0
    last_sign     = ""
    last_add_time = 0.0
    prev_motion   = ""
    cached_lm     = None
    frame_ts = frame_idx = 0
    fps = fps_count = 0.0
    fps_timer = time.time()
    small = np.empty((ML_H, ML_W, 3), dtype=np.uint8)

    async def listen():
        nonlocal sentence, last_sign, frame_buffer
        async for msg in websocket:
            try:
                cmd = json.loads(msg)
                if cmd.get("action") == "space":
                    sentence += " "; last_sign = ""
                elif cmd.get("action") == "backspace":
                    parts = sentence.rstrip().rsplit(" ", 1)
                    sentence = parts[0] + " " if len(parts) > 1 else ""
                    last_sign = ""
                elif cmd.get("action") == "clear":
                    sentence = ""; last_sign = ""; frame_buffer.clear()
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

                lms_list = results.hand_landmarks if results.hand_landmarks else []
                cached_lm = lms_list if lms_list else None

                # Always push frame (even zero-hand frames fill the buffer with zeros)
                feats = extract_motion_frame_features(lms_list)
                frame_buffer.append(feats)

                # Static overlay
                if lms_list and static_clf is not None:
                    sf   = engineered_features_from_landmarks(lms_list)
                    sp   = static_clf.predict_proba(sf)[0]
                    si   = int(np.argmax(sp))
                    static_sign = static_clf.classes_[si]
                    static_conf = float(sp[si])
                elif not lms_list:
                    static_sign = ""
                    static_conf = 0.0

                # Motion classification
                motion_val = compute_motion(frame_buffer)
                is_classifying = False

                if len(frame_buffer) == SEQ_LENGTH and motion_val > MOTION_THRESHOLD:
                    is_classifying = True
                    pred, conf, alts = classify_motion_seq(frame_buffer, payload)
                    motion_sign = pred
                    motion_conf = conf
                    motion_alts = alts

                    if pred == prev_motion and conf >= MOTION_CONFIDENCE:
                        stable_count += 1
                    else:
                        stable_count = 0

                    now = time.time()
                    if (stable_count >= MOTION_STABLE_FRAMES
                            and pred != last_sign
                            and (now - last_add_time) >= MOTION_COOLDOWN
                            and conf >= MOTION_CONFIDENCE):
                        sentence     += sentence_token(pred)
                        last_sign     = pred
                        last_add_time = now
                        stable_count  = 0
                        print(f"[motion] Added: {format_label(pred)}")

                    prev_motion = pred
                elif motion_val <= MOTION_THRESHOLD:
                    stable_count = 0
                    prev_motion  = ""

            if cached_lm:
                for i, lm in enumerate(cached_lm):
                    draw_skeleton(frame, lm, HAND_COLORS[min(i, 1)])

            fps_count += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                fps = fps_count / (now - fps_timer)
                fps_count = 0
                fps_timer = now

            motion_val = compute_motion(frame_buffer)

            await websocket.send(json.dumps({
                "frame":          encode_frame(frame),
                "motion_sign":    format_label(motion_sign) if motion_sign else "",
                "motion_raw":     motion_sign,
                "motion_conf":    round(motion_conf, 3),
                "motion_alts":    motion_alts,
                "static_sign":    format_label(static_sign) if static_sign else "",
                "static_conf":    round(static_conf, 3),
                "is_word":        motion_sign.startswith("[") if motion_sign else False,
                "sentence":       sentence.strip(),
                "num_hands":      len(cached_lm) if cached_lm else 0,
                "buffer_pct":     round(len(frame_buffer) / SEQ_LENGTH, 2),
                "motion_val":     round(float(motion_val), 4),
                "motion_active":  motion_val > MOTION_THRESHOLD,
                "stable_pct":     round(min(stable_count / MOTION_STABLE_FRAMES, 1.0), 2),
                "fps":            round(fps, 1),
                "model_classes":  payload["classes"],
            }))
            await asyncio.sleep(0.001)

    except websockets.exceptions.ConnectionClosed:
        print("[motion] Client disconnected.")
    finally:
        cap.release()
        landmarker.close()


# ── Static data collector handler (port 8766) ─────────────────────────────────

async def collector_handler(websocket):
    print(f"[collector] Client connected: {websocket.remote_address}")

    landmarker = make_landmarker()
    cap = open_camera()
    if cap is None:
        await websocket.send(json.dumps({"error": "Cannot open webcam."}))
        landmarker.close()
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    existing_rows = []
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            existing_rows = list(csv.reader(f))

    class_stats = {}
    for row in existing_rows[1:]:
        if row:
            class_stats[row[0]] = class_stats.get(row[0], 0) + 1

    new_samples   = []
    recording     = False
    current_label = ""
    session_count = 0
    frame_ts = frame_idx = 0
    cached_lm = None
    small = np.empty((ML_H, ML_W, 3), dtype=np.uint8)
    fps = fps_count = 0.0
    fps_timer = time.time()

    async def listen():
        nonlocal recording, current_label, session_count, existing_rows, new_samples, class_stats
        async for msg in websocket:
            try:
                cmd    = json.loads(msg)
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
                    header   = make_csv_header()
                    all_rows = (existing_rows if existing_rows else [header]) + new_samples
                    with open(DATA_FILE, "w", newline="") as f:
                        csv.writer(f).writerows(all_rows)
                    total = len(all_rows) - 1
                    print(f"[collector] Saved {len(new_samples)} new samples ({total} total)")
                    await websocket.send(json.dumps({
                        "saved": len(new_samples),
                        "path":  DATA_FILE,
                        "total_samples": total,
                    }))

                elif action == "trim_label":
                    label = cmd.get("label", "").strip()
                    if label and os.path.exists(DATA_FILE):
                        with open(DATA_FILE, "r") as f:
                            rows = list(csv.reader(f))
                        header = rows[0] if rows else []
                        kept   = [r for r in rows[1:] if r and r[0] != label]
                        with open(DATA_FILE, "w", newline="") as f:
                            csv.writer(f).writerows([header] + kept)
                        # update in-memory state
                        class_stats.pop(label, None)
                        existing_rows[:] = [header] + kept
                        new_samples[:] = [r for r in new_samples if r and r[0] != label]
                        print(f"[collector] Deleted all '{label}' samples")
                        await websocket.send(json.dumps({"trimmed": True, "label": label}))

                elif action == "train":
                    print("[collector] Starting static model training...")
                    await websocket.send(json.dumps({"train_start": True}))
                    import subprocess, sys
                    proc = await asyncio.create_subprocess_exec(
                        sys.executable, "train_model.py",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.STDOUT,
                    )

                    async def stream_output():
                        while True:
                            line = await proc.stdout.readline()
                            if not line:
                                break
                            text = line.decode("utf-8", errors="replace").rstrip()
                            try:
                                await websocket.send(json.dumps({"train_log": text}))
                            except Exception:
                                break
                        await proc.wait()
                        try:
                            await websocket.send(json.dumps({
                                "train_done": True,
                                "train_log": f"--- Done (exit code {proc.returncode}) ---",
                            }))
                        except Exception:
                            pass

                    asyncio.ensure_future(stream_output())

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


# ── Motion data collector handler (port 8769) ─────────────────────────────────

async def motion_collector_handler(websocket):
    print(f"[motion-collector] Client connected: {websocket.remote_address}")

    landmarker = make_landmarker(detection_conf=0.5)
    cap = open_camera()
    if cap is None:
        await websocket.send(json.dumps({"error": "Cannot open webcam."}))
        landmarker.close()
        return

    os.makedirs(DATA_DIR, exist_ok=True)

    # Count existing sequences
    count_map = {}
    if os.path.exists(MOTION_DATA_FILE):
        with open(MOTION_DATA_FILE, "r") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            prev_label = None
            for row in reader:
                if not row:
                    continue
                lbl   = row[0]
                frame = int(row[1]) if len(row) > 1 else 0
                if frame == 0 and lbl != prev_label:
                    count_map[lbl] = count_map.get(lbl, 0) + 1
                prev_label = lbl

    state         = "idle"    # idle | recording | done
    recording_buf = []        # current in-progress sequence frames
    pending_seq   = None      # completed sequence waiting for label
    frame_ts = frame_idx = 0
    cached_lm = None
    small = np.empty((ML_H, ML_W, 3), dtype=np.uint8)
    fps = fps_count = 0.0
    fps_timer = time.time()

    async def listen():
        nonlocal state, recording_buf, pending_seq, count_map
        async for msg in websocket:
            try:
                cmd    = json.loads(msg)
                action = cmd.get("action")

                if action == "start_recording":
                    if state == "idle":
                        recording_buf = []
                        state         = "recording"
                        print("[motion-collector] Recording started")

                elif action == "stop_recording":
                    if state == "recording" and len(recording_buf) >= 10:
                        pending_seq = list(recording_buf)
                        recording_buf = []
                        state = "done"
                        print(f"[motion-collector] Recording stopped ({len(pending_seq)} frames)")
                    else:
                        recording_buf = []
                        state = "idle"

                elif action == "label_sequence":
                    label = cmd.get("label", "").strip()
                    if state == "done" and pending_seq and label:
                        # Pad / trim to SEQ_LENGTH
                        seq = pending_seq[:SEQ_LENGTH]
                        while len(seq) < SEQ_LENGTH:
                            seq.append(seq[-1])

                        # Append to CSV
                        file_exists = os.path.exists(MOTION_DATA_FILE)
                        with open(MOTION_DATA_FILE, "a", newline="") as f:
                            writer = csv.writer(f)
                            if not file_exists:
                                writer.writerow(make_motion_csv_header())
                            for fi, frame_feats in enumerate(seq):
                                writer.writerow([label, fi] + list(frame_feats))

                        count_map[label] = count_map.get(label, 0) + 1
                        total = count_map[label]
                        print(f"[motion-collector] Saved '{label}' ({total} total)")

                        await websocket.send(json.dumps({
                            "saved_sequence": True,
                            "label":          label,
                            "count":          total,
                            "count_map":      count_map,
                        }))
                        pending_seq = None
                        state       = "idle"

                elif action == "discard":
                    pending_seq   = None
                    recording_buf = []
                    state         = "idle"

                elif action == "trim_motion_label":
                    label = cmd.get("label", "").strip()
                    if label and os.path.exists(MOTION_DATA_FILE):
                        with open(MOTION_DATA_FILE, "r") as f:
                            rows = list(csv.reader(f))
                        header = rows[0] if rows else []
                        kept   = [r for r in rows[1:] if r and r[0] != label]
                        with open(MOTION_DATA_FILE, "w", newline="") as f:
                            csv.writer(f).writerows([header] + kept)
                        count_map.pop(label, None)
                        print(f"[motion-collector] Deleted all '{label}' motion sequences")
                        await websocket.send(json.dumps({
                            "trimmed_motion": True,
                            "label": label,
                            "count_map": count_map,
                        }))

                elif action == "train_motion":
                    print("[motion-collector] Starting motion model training...")
                    await websocket.send(json.dumps({"train_start": True}))
                    import subprocess, sys
                    proc = await asyncio.create_subprocess_exec(
                        sys.executable, "train_motion_model.py",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.STDOUT,
                    )

                    async def stream_output():
                        while True:
                            line = await proc.stdout.readline()
                            if not line:
                                break
                            text = line.decode("utf-8", errors="replace").rstrip()
                            try:
                                await websocket.send(json.dumps({"train_log": text}))
                            except Exception:
                                break
                        await proc.wait()
                        try:
                            await websocket.send(json.dumps({
                                "train_done": True,
                                "train_log": f"--- Done (exit {proc.returncode}) ---",
                            }))
                        except Exception:
                            pass

                    asyncio.ensure_future(stream_output())

            except Exception as ex:
                print(f"[motion-collector] Command error: {ex}")

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

                lms_list  = results.hand_landmarks if results.hand_landmarks else []
                cached_lm = lms_list if lms_list else None

                if state == "recording":
                    feats = extract_motion_frame_features(lms_list)
                    recording_buf.append(feats.tolist())
                    if len(recording_buf) >= SEQ_LENGTH:
                        pending_seq   = list(recording_buf)
                        recording_buf = []
                        state         = "done"
                        print(f"[motion-collector] Auto-stopped at {SEQ_LENGTH} frames")

            if cached_lm:
                for i, lm in enumerate(cached_lm):
                    draw_skeleton(frame, lm, HAND_COLORS[min(i, 1)])

            fps_count += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                fps = fps_count / (now - fps_timer)
                fps_count = 0
                fps_timer = now

            progress = len(recording_buf) / SEQ_LENGTH if state == "recording" else (1.0 if state == "done" else 0.0)

            await websocket.send(json.dumps({
                "frame":       encode_frame(frame),
                "state":       state,
                "num_hands":   len(cached_lm) if cached_lm else 0,
                "progress":    round(progress, 2),
                "frames_recorded": len(recording_buf) if state == "recording" else (SEQ_LENGTH if state == "done" else 0),
                "count_map":   count_map,
                "fps":         round(fps, 1),
                "has_pending": pending_seq is not None,
            }))
            await asyncio.sleep(0.001)

    except websockets.exceptions.ConnectionClosed:
        print("[motion-collector] Client disconnected.")
    finally:
        cap.release()
        landmarker.close()


# ── Entry point ───────────────────────────────────────────────────────────────

def restart_server():
    import subprocess, sys
    print("[server] Restarting...")
    subprocess.Popen([sys.executable] + sys.argv)
    os._exit(0)


async def main():
    print("ASL Server starting…")
    print("  Static translator  : ws://localhost:8765")
    print("  Static collector   : ws://localhost:8766")
    print("  Restart            : ws://localhost:8767")
    print("  Motion translator  : ws://localhost:8768")
    print("  Motion collector   : ws://localhost:8769")

    async def restart_handler(websocket):
        async for msg in websocket:
            try:
                cmd = json.loads(msg)
                if cmd.get("action") == "restart":
                    await websocket.send(json.dumps({"restarting": True}))
                    await asyncio.sleep(0.2)
                    asyncio.get_event_loop().call_later(0.1, restart_server)
            except Exception:
                pass

    async with (
        websockets.serve(translator_handler,        "localhost", 8765),
        websockets.serve(collector_handler,         "localhost", 8766),
        websockets.serve(restart_handler,           "localhost", 8767),
        websockets.serve(motion_translator_handler, "localhost", 8768),
        websockets.serve(motion_collector_handler,  "localhost", 8769),
    ):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())