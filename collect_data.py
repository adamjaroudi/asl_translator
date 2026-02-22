"""
Collect hand landmark data from your webcam for training the ASL classifier.
Supports both one-hand and two-hand signs.

Usage:
    python collect_data.py

Controls:
    A-Z        Start recording that letter sign
    0-9        Start recording word sign (page 1)
    SPACE      Stop recording
    BACKSPACE  Remove last sign from this session (e.g. delete bad "E" before saving)
    ESC        Quit and save
    Q          Quit without saving

Number keys:  1=I-LOVE-YOU  2=GOOD  3=MORE  4=HELP  5=BOOK
              6=STOP  7=PLAY  8=WANT  9=WITH  0=SAME
Extra keys:   -=NO  ==YES  [=FRIEND  ]=WORK  ;=FINISH
              '=GO  ,=SIT  .=BIG  /=SMALL  \\=LOVE  `=EAT  TAB=DRINK
"""

import argparse
import os
import csv
import cv2
import mediapipe as mp
import numpy as np

from camera_util import open_camera

DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "landmarks.csv")
NUM_LANDMARKS = 21
FEATURES_PER_HAND = NUM_LANDMARKS * 3
TOTAL_FEATURES = FEATURES_PER_HAND * 2 + 3 + 1  # 130
HAND_MODEL = os.path.join("model", "hand_landmarker.task")

HAND_CONNECTIONS = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

WORD_KEYS = {
    ord("1"): "[I-LOVE-YOU]",
    ord("2"): "[GOOD]",
    ord("3"): "[MORE]",
    ord("4"): "[HELP]",
    ord("5"): "[BOOK]",
    ord("6"): "[STOP]",
    ord("7"): "[PLAY]",
    ord("8"): "[WANT]",
    ord("9"): "[WITH]",
    ord("0"): "[SAME]",
    ord("-"): "[NO]",
    ord("="): "[YES]",
    ord("["): "[FRIEND]",
    ord("]"): "[WORK]",
    ord(";"): "[FINISH]",
    ord("'"): "[GO]",
    ord(","): "[SIT]",
    ord("."): "[BIG]",
    ord("/"): "[SMALL]",
    ord("\\"): "[LOVE]",
    ord("`"): "[EAT]",
    9: "[DRINK]",  # TAB key
}

HAND_COLORS = [(255, 160, 50), (50, 200, 255)]  # orange for hand1, cyan for hand2


def normalize_landmarks(landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    wrist = coords[0]
    coords = coords - wrist
    max_dist = np.max(np.linalg.norm(coords, axis=1))
    if max_dist > 0:
        coords = coords / max_dist
    return coords.flatten().tolist()


def extract_features(all_hand_landmarks):
    """Build the 130-dim feature vector from detected hands (1 or 2)."""
    hands_with_x = []
    for landmarks in all_hand_landmarks:
        wrist_x = landmarks[0].x
        hands_with_x.append((wrist_x, landmarks))

    hands_with_x.sort(key=lambda t: t[0])

    h1_lm = hands_with_x[0][1]
    h1_features = normalize_landmarks(h1_lm)
    h1_wrist = np.array([h1_lm[0].x, h1_lm[0].y, h1_lm[0].z])

    if len(hands_with_x) >= 2:
        h2_lm = hands_with_x[1][1]
        h2_features = normalize_landmarks(h2_lm)
        h2_wrist = np.array([h2_lm[0].x, h2_lm[0].y, h2_lm[0].z])
        rel = (h2_wrist - h1_wrist).tolist()
        num_hands = 1.0
    else:
        h2_features = [0.0] * FEATURES_PER_HAND
        rel = [0.0, 0.0, 0.0]
        num_hands = 0.0

    return h1_features + h2_features + rel + [num_hands]


def draw_hand(img, landmarks, color):
    h, w, _ = img.shape
    for conn in HAND_CONNECTIONS:
        s, e = conn.start, conn.end
        pt1 = (int(landmarks[s].x * w), int(landmarks[s].y * h))
        pt2 = (int(landmarks[e].x * w), int(landmarks[e].y * h))
        cv2.line(img, pt1, pt2, color, 2, cv2.LINE_AA)
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 4, (255, 255, 255), -1, cv2.LINE_AA)


def make_header():
    cols = ["label"]
    for prefix in ("h1", "h2"):
        for i in range(NUM_LANDMARKS):
            for axis in ("x", "y", "z"):
                cols.append(f"{prefix}_{axis}{i}")
    cols += ["rel_x", "rel_y", "rel_z", "num_hands"]
    return cols


def main():
    parser = argparse.ArgumentParser(description="Collect ASL hand data from webcam.")
    parser.add_argument("--camera", type=int, default=None, help="Camera index (0, 1, 2...). Or set CAMERA_INDEX env.")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(HAND_MODEL):
        print(f"Downloading hand landmarker model...")
        import urllib.request
        os.makedirs("model", exist_ok=True)
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/latest/hand_landmarker.task",
            HAND_MODEL,
        )
        print("Downloaded.")

    existing_rows = []
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            reader = csv.reader(f)
            existing_rows = list(reader)
        print(f"Loaded {len(existing_rows) - 1} existing samples.")

    cap, ok = open_camera(args.camera)
    if not ok or cap is None:
        print("Error: Cannot open webcam. Check Windows Settings → Privacy → Camera.")
        return

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    hand_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    recording = False
    current_label = ""
    sample_count = 0
    new_samples = []  # This session only; only saved if you press ESC
    save_on_exit = True  # False if user pressed Q
    frame_timestamp = 0

    total_in_file = len(existing_rows) - 1 if existing_rows else 0

    print("\n=== ASL Data Collector (Letters + Words, Both Hands) ===")
    print("Press a letter key (A-Z) to record that letter sign.")
    print("Press a number key for word signs:")
    for k, v in sorted(WORD_KEYS.items()):
        print(f"  {chr(k)} = {v}")
    print("SPACE: stop recording | BACKSPACE: remove last sign from session | ESC: save | Q: quit without save\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        frame_timestamp += 33
        results = hand_landmarker.detect_for_video(mp_image, frame_timestamp)

        num_hands = len(results.hand_landmarks) if results.hand_landmarks else 0

        if results.hand_landmarks:
            for idx, landmarks in enumerate(results.hand_landmarks):
                color = HAND_COLORS[min(idx, 1)]
                draw_hand(frame, landmarks, color)

            if recording:
                features = extract_features(results.hand_landmarks)
                new_samples.append([current_label] + features)
                sample_count += 1

        status_color = (0, 0, 255) if recording else (200, 200, 200)
        status_text = (
            f"Recording: {current_label} ({sample_count} samples)"
            if recording
            else "Ready - press a key to record"
        )
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame, f"Hands: {num_hands} | This session: {len(new_samples)} | In file: {total_in_file}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.putText(frame, "BACKSPACE: remove last sign | ESC: save | Q: quit no save", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        cv2.imshow("ASL Data Collector", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC — quit and save
            save_on_exit = True
            break
        elif key in (ord("q"), ord("Q")):  # Q — quit without saving
            save_on_exit = False
            print("\nQuit without saving. This session's samples discarded.")
            break
        elif key == 32:  # SPACE
            if recording:
                recording = False
                print(f"  Stopped recording '{current_label}'. Got {sample_count} samples.")
        elif key in (8, 127):  # BACKSPACE — remove last sign from this session (e.g. wrong E)
            if not recording and new_samples:
                last_label = new_samples[-1][0]
                n_removed = sum(1 for row in new_samples if row[0] == last_label)
                new_samples[:] = [row for row in new_samples if row[0] != last_label]
                print(f"  Removed {n_removed} samples for '{last_label}' (not saved when you press ESC).")
            elif not new_samples:
                print("  No samples in this session to remove.")
        elif key in WORD_KEYS:
            current_label = WORD_KEYS[key]
            recording = True
            sample_count = 0
            print(f"  Recording word sign '{current_label}'...")
        elif 65 <= key <= 90 or 97 <= key <= 122:
            current_label = chr(key).upper()
            recording = True
            sample_count = 0
            print(f"  Recording letter '{current_label}'...")

    cap.release()
    cv2.destroyAllWindows()
    hand_landmarker.close()

    if save_on_exit and (existing_rows or new_samples):
        header = make_header()
        rows_to_save = existing_rows if existing_rows else [header]
        rows_to_save = rows_to_save + new_samples
        with open(DATA_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows_to_save)
        print(f"\nSaved {len(new_samples)} new samples ({len(rows_to_save) - 1} total in file).")


if __name__ == "__main__":
    main()
