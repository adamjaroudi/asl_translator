"""
ASL WebSocket Server

Streams real-time ASL predictions to a React frontend over WebSocket.

Install extra dependency:
    pip install websockets

Usage:
    python server.py

Then open the React app at http://localhost:5173 (or wherever Vite serves it).
"""

import asyncio
import json
import pickle
import time
import os
import base64

import cv2
import mediapipe as mp
import numpy as np
import websockets

from features import engineer_features_from_raw

MODEL_FILE = os.path.join("model", "asl_classifier.pkl")
HAND_MODEL = os.path.join("model", "hand_landmarker.task")

NUM_LANDMARKS     = 21
FEATURES_PER_HAND = NUM_LANDMARKS * 3
CONFIDENCE_THRESHOLD = 0.6
STABLE_FRAMES        = 12
COOLDOWN_SECONDS     = 1.0
ML_W, ML_H = 480, 360
DETECTION_INTERVAL = 2

HAND_CONNECTIONS  = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
FINGERTIP_INDICES = {4, 8, 12, 16, 20}

# Colors for skeleton (BGR)
HAND_COLORS   = [(255, 160, 50), (50, 200, 255)]
GREEN         = (100, 220, 100)
TEXT_WHITE    = (255, 255, 255)
BG_DARK       = (30, 30, 30)


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
        rel, nh = (h2w - h1w).tolist(), 1.0
    else:
        h2f = [0.0] * FEATURES_PER_HAND
        rel, nh = [0.0, 0.0, 0.0], 0.0
    raw = h1f + h2f + rel + [nh]
    return np.array(engineer_features_from_raw(raw), dtype=np.float32).reshape(1, -1)


def format_label(label):
    if label.startswith("[") and label.endswith("]"):
        return label[1:-1].replace("-", " ").title()
    return label


def sentence_token(label):
    if label.startswith("[") and label.endswith("]"):
        return f" {label[1:-1].replace('-', ' ').lower()} "
    return label


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


def frame_to_jpeg_b64(frame, quality=60):
    """Encode frame as base64 JPEG for sending over WebSocket."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("ascii")


async def asl_handler(websocket):
    print(f"Client connected: {websocket.remote_address}")

    # ── Load models ──────────────────────────────────────────────────────
    if not os.path.exists(MODEL_FILE):
        await websocket.send(json.dumps({"error": "Model not found. Run train_model.py first."}))
        return

    if not os.path.exists(HAND_MODEL):
        await websocket.send(json.dumps({"status": "Downloading hand model..."}))
        import urllib.request
        os.makedirs("model", exist_ok=True)
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/latest/hand_landmarker.task",
            HAND_MODEL,
        )

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

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        await websocket.send(json.dumps({"error": "Cannot open webcam."}))
        landmarker.close()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS,          60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    sentence        = ""
    predicted_sign  = ""
    confidence      = 0.0
    stable_count    = 0
    last_sign       = ""
    last_add_time   = 0.0
    prev_prediction = ""
    cached_lm       = None
    frame_ts        = 0
    frame_idx       = 0
    small           = np.empty((ML_H, ML_W, 3), dtype=np.uint8)

    fps = 0.0
    fps_count = 0
    fps_timer = time.time()

    print("Streaming started.")
    try:
        # Listen for commands from the frontend in a separate task
        async def listen_commands():
            nonlocal sentence, last_sign
            async for msg in websocket:
                try:
                    cmd = json.loads(msg)
                    if cmd.get("action") == "space":
                        sentence += " "
                        last_sign = ""
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
                        sentence  = ""
                        last_sign = ""
                except Exception:
                    pass

        asyncio.ensure_future(listen_commands())

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.flip(frame, 1, dst=frame)
            frame_idx += 1

            if frame_idx % DETECTION_INTERVAL == 0:
                cv2.resize(frame, (ML_W, ML_H), dst=small, interpolation=cv2.INTER_NEAREST)
                rgb      = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                frame_ts += 33
                mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                results  = landmarker.detect_for_video(mp_img, frame_ts)

                if results.hand_landmarks:
                    cached_lm    = results.hand_landmarks
                    features     = extract_features(cached_lm)
                    proba        = clf.predict_proba(features)[0]
                    idx          = int(np.argmax(proba))
                    current_pred = clf.classes_[idx]
                    current_conf = float(proba[idx])

                    predicted_sign = current_pred
                    confidence     = current_conf

                    if current_pred == prev_prediction and current_conf >= CONFIDENCE_THRESHOLD:
                        stable_count += 1
                    else:
                        stable_count = 0

                    now = time.time()
                    if (
                        stable_count >= STABLE_FRAMES
                        and current_pred != last_sign
                        and (now - last_add_time) >= COOLDOWN_SECONDS
                        and current_conf >= CONFIDENCE_THRESHOLD
                    ):
                        sentence     += sentence_token(current_pred)
                        last_sign     = current_pred
                        last_add_time = now
                        stable_count  = 0

                    prev_prediction = current_pred
                else:
                    cached_lm       = None
                    predicted_sign  = ""
                    confidence      = 0.0
                    stable_count    = 0
                    prev_prediction = ""
                    last_sign       = ""

            # Draw skeleton
            if cached_lm:
                for i, lm in enumerate(cached_lm):
                    draw_skeleton(frame, lm, HAND_COLORS[min(i, 1)])

            # FPS
            fps_count += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                fps       = fps_count / (now - fps_timer)
                fps_count = 0
                fps_timer = now

            # Encode frame and send JSON payload
            frame_b64 = frame_to_jpeg_b64(frame, quality=55)

            payload = {
                "frame":       frame_b64,
                "prediction":  format_label(predicted_sign) if predicted_sign else "",
                "raw_label":   predicted_sign,
                "confidence":  round(confidence, 3),
                "is_word":     predicted_sign.startswith("["),
                "sentence":    sentence.strip(),
                "num_hands":   len(cached_lm) if cached_lm else 0,
                "stable_pct":  round(min(stable_count / STABLE_FRAMES, 1.0), 2),
                "fps":         round(fps, 1),
            }
            await websocket.send(json.dumps(payload))

            # ~30fps cap — yield to event loop
            await asyncio.sleep(0.001)

    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected.")
    finally:
        cap.release()
        landmarker.close()
        print("Camera released.")


async def main():
    print("ASL WebSocket server starting on ws://localhost:8765")
    print("Open the React app and click 'Start Camera'")
    async with websockets.serve(asl_handler, "localhost", 8765):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())