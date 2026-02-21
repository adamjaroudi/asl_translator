"""
Import a real ASL dataset and combine it with synthetic word sign data.

This script handles two sources:
  1. Pre-extracted MediaPipe landmark CSV (fastest — from Kaggle)
  2. Folder of ASL alphabet images (processes through MediaPipe)

The imported letter data is combined with synthetic two-hand word signs
to produce a complete training dataset.

=== QUICKSTART (recommended) ===

  1. Go to: https://www.kaggle.com/datasets/jaisuryaprabu/sign-language-landmarks
     (Free Kaggle account required — just sign in with Google)
  2. Click "Download" to get the ZIP file
  3. Extract and place the CSV file in this project's "downloads/" folder
  4. Run: python import_dataset.py
  5. Run: python train_model.py

=== ALTERNATIVE: Image dataset ===

  1. Download any ASL alphabet image dataset
  2. Organize images as: downloads/images/A/*.jpg, downloads/images/B/*.jpg, etc.
  3. Run: python import_dataset.py
  4. Run: python train_model.py
"""

import os
import sys
import csv
import glob
import numpy as np

DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "landmarks.csv")
DOWNLOAD_DIR = "downloads"
NUM_LANDMARKS = 21
FEATURES_PER_HAND = NUM_LANDMARKS * 3  # 63
MAX_SAMPLES_PER_CLASS = 800

# Labels to skip (non-letter classes sometimes found in datasets)
SKIP_LABELS = {"del", "delete", "space", "nothing", "blank", ""}


def make_header():
    cols = ["label"]
    for prefix in ("h1", "h2"):
        for i in range(NUM_LANDMARKS):
            for axis in ("x", "y", "z"):
                cols.append(f"{prefix}_{axis}{i}")
    cols += ["rel_x", "rel_y", "rel_z", "num_hands"]
    return cols


def single_hand_to_130(features_63):
    """Convert a 63-dim single-hand vector to our 130-dim two-hand format."""
    h2_zeros = [0.0] * FEATURES_PER_HAND
    rel_zeros = [0.0, 0.0, 0.0]
    return features_63 + h2_zeros + rel_zeros + [0.0]


def normalize_63(raw_features):
    """Normalize 63 raw landmark features (x,y,z for 21 points) to wrist-relative."""
    coords = np.array(raw_features).reshape(21, 3)
    wrist = coords[0].copy()
    coords = coords - wrist
    max_dist = np.max(np.linalg.norm(coords, axis=1))
    if max_dist > 0:
        coords = coords / max_dist
    return coords.flatten().tolist()


def find_landmark_csv():
    """Search downloads/ for any CSV that looks like an ASL landmark dataset."""
    if not os.path.exists(DOWNLOAD_DIR):
        return None

    csv_files = glob.glob(os.path.join(DOWNLOAD_DIR, "**", "*.csv"), recursive=True)
    for f in csv_files:
        try:
            with open(f, "r") as fh:
                reader = csv.reader(fh)
                header = next(reader)
                first_row = next(reader)
                # Check if it has ~63+ numeric columns and a label column
                num_cols = len(header)
                if num_cols >= 60:
                    return f
        except (StopIteration, UnicodeDecodeError):
            continue
    return None


def import_landmark_csv(csv_path):
    """Import pre-extracted landmark CSV and convert to our 130-dim format."""
    print(f"Reading landmarks from: {csv_path}")

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        raw_rows = list(reader)

    print(f"  Found {len(raw_rows)} rows, {len(header)} columns")

    # Detect label column (usually first or last)
    label_col = None
    for i, col_name in enumerate(header):
        if col_name.lower() in ("label", "class", "letter", "sign", "target", "category"):
            label_col = i
            break

    if label_col is None:
        # Try first column: check if values are letters
        test_vals = set(row[0].strip().upper() for row in raw_rows[:100])
        if test_vals.issubset(set("ABCDEFGHIJKLMNOPQRSTUVWXYZ") | SKIP_LABELS):
            label_col = 0
        else:
            # Try last column
            test_vals = set(row[-1].strip().upper() for row in raw_rows[:100])
            if test_vals.issubset(set("ABCDEFGHIJKLMNOPQRSTUVWXYZ") | SKIP_LABELS):
                label_col = len(header) - 1
            else:
                print("Error: Cannot detect which column contains the labels.")
                return []

    print(f"  Label column: '{header[label_col]}' (index {label_col})")

    feature_cols = [i for i in range(len(header)) if i != label_col]

    # If more than 63 feature columns, take only the first 63 (x,y,z landmarks)
    if len(feature_cols) > FEATURES_PER_HAND:
        feature_cols = feature_cols[:FEATURES_PER_HAND]

    rows_by_label = {}
    skipped = 0

    for row in raw_rows:
        label = row[label_col].strip().upper()
        if label.lower() in SKIP_LABELS or len(label) != 1 or not label.isalpha():
            skipped += 1
            continue

        try:
            features = [float(row[c]) for c in feature_cols]
        except (ValueError, IndexError):
            skipped += 1
            continue

        if len(features) != FEATURES_PER_HAND:
            skipped += 1
            continue

        if label not in rows_by_label:
            rows_by_label[label] = []
        rows_by_label[label].append(features)

    if skipped > 0:
        print(f"  Skipped {skipped} rows (non-letter labels or bad data)")

    converted = []
    for label in sorted(rows_by_label.keys()):
        samples = rows_by_label[label]
        if len(samples) > MAX_SAMPLES_PER_CLASS:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(samples), MAX_SAMPLES_PER_CLASS, replace=False)
            samples = [samples[i] for i in indices]

        for features_63 in samples:
            normalized = normalize_63(features_63)
            features_130 = single_hand_to_130(normalized)
            converted.append([label] + features_130)

        print(f"  {label}: {len(samples)} samples")

    return converted


def find_image_folder():
    """Search for organized image folders in downloads/."""
    image_dir = os.path.join(DOWNLOAD_DIR, "images")
    if not os.path.isdir(image_dir):
        # Also check for common Kaggle extraction patterns
        for candidate in ["asl_alphabet_train", "asl-alphabet", "train", "Train"]:
            alt = os.path.join(DOWNLOAD_DIR, candidate)
            if os.path.isdir(alt):
                image_dir = alt
                break

    if not os.path.isdir(image_dir):
        return None

    # Check it has letter subfolders
    subdirs = [d for d in os.listdir(image_dir)
               if os.path.isdir(os.path.join(image_dir, d)) and len(d) == 1 and d.isalpha()]
    if len(subdirs) >= 10:
        return image_dir
    return None


def import_images(image_dir):
    """Process ASL images through MediaPipe to extract landmarks."""
    try:
        import cv2
        import mediapipe as mp
    except ImportError:
        print("Error: opencv-python and mediapipe are required for image processing.")
        print("Run: pip install -r requirements.txt")
        return []

    hand_model = os.path.join("model", "hand_landmarker.task")
    if not os.path.exists(hand_model):
        print("Downloading hand landmarker model...")
        import urllib.request
        os.makedirs("model", exist_ok=True)
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/latest/hand_landmarker.task",
            hand_model,
        )

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=hand_model),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )
    landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    subdirs = sorted([d for d in os.listdir(image_dir)
                      if os.path.isdir(os.path.join(image_dir, d))
                      and len(d) <= 2 and d.upper().isalpha()])

    converted = []
    total_processed = 0

    for subdir in subdirs:
        label = subdir.upper()
        folder = os.path.join(image_dir, subdir)
        image_files = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            image_files.extend(glob.glob(os.path.join(folder, ext)))
            image_files.extend(glob.glob(os.path.join(folder, ext.upper())))

        if not image_files:
            continue

        if len(image_files) > MAX_SAMPLES_PER_CLASS:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(image_files), MAX_SAMPLES_PER_CLASS, replace=False)
            image_files = [image_files[i] for i in indices]

        count = 0
        for img_path in image_files:
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                results = landmarker.detect(mp_image)

                if not results.hand_landmarks:
                    continue

                hands = results.hand_landmarks
                hands_sorted = sorted(hands, key=lambda lms: lms[0].x)

                h1 = hands_sorted[0]
                h1_coords = np.array([[lm.x, lm.y, lm.z] for lm in h1])
                h1_coords = h1_coords - h1_coords[0]
                md = np.max(np.linalg.norm(h1_coords, axis=1))
                if md > 0:
                    h1_coords /= md
                h1_features = h1_coords.flatten().tolist()

                if len(hands_sorted) >= 2:
                    h2 = hands_sorted[1]
                    h2_coords = np.array([[lm.x, lm.y, lm.z] for lm in h2])
                    h2_coords = h2_coords - h2_coords[0]
                    md2 = np.max(np.linalg.norm(h2_coords, axis=1))
                    if md2 > 0:
                        h2_coords /= md2
                    h2_features = h2_coords.flatten().tolist()
                    rel = [hands_sorted[1][0].x - hands_sorted[0][0].x,
                           hands_sorted[1][0].y - hands_sorted[0][0].y,
                           hands_sorted[1][0].z - hands_sorted[0][0].z]
                    num_hands = 1.0
                else:
                    h2_features = [0.0] * FEATURES_PER_HAND
                    rel = [0.0, 0.0, 0.0]
                    num_hands = 0.0

                features_130 = h1_features + h2_features + rel + [num_hands]
                converted.append([label] + features_130)
                count += 1
            except Exception:
                continue

        total_processed += count
        print(f"  {label}: {count}/{len(image_files)} images processed")

    landmarker.close()
    print(f"  Total: {total_processed} landmarks extracted from images")
    return converted


def generate_word_signs(n_samples=500):
    """Generate synthetic samples for two-hand word signs."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from generate_dataset import ASL_SIGNS, generate_sample, make_header

    rng = np.random.default_rng(42)
    word_labels = sorted(l for l in ASL_SIGNS if l.startswith("["))

    rows = []
    print(f"\nGenerating {n_samples} synthetic samples for {len(word_labels)} word signs...")
    for label in word_labels:
        for _ in range(n_samples):
            features = generate_sample(label, rng)
            rows.append([label] + features)
        print(f"  {label:18s}: {n_samples} samples")

    return rows


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    print("=" * 60)
    print("  ASL Dataset Importer")
    print("=" * 60)

    letter_rows = []

    # Try landmark CSV first (fastest)
    csv_path = find_landmark_csv()
    if csv_path:
        print(f"\n[1/3] Found landmark CSV")
        letter_rows = import_landmark_csv(csv_path)
    else:
        # Try image folder
        img_dir = find_image_folder()
        if img_dir:
            print(f"\n[1/3] Found image folder: {img_dir}")
            letter_rows = import_images(img_dir)
        else:
            print(f"\n[1/3] No dataset found in '{DOWNLOAD_DIR}/' folder.")
            print()
            print("  OPTION A (recommended — fast, small download):")
            print("    1. Go to: https://www.kaggle.com/datasets/jaisuryaprabu/sign-language-landmarks")
            print("    2. Sign in (free Google account)")
            print("    3. Click 'Download' (2.7 MB ZIP)")
            print("    4. Extract the CSV into the 'downloads/' folder")
            print("    5. Re-run: python import_dataset.py")
            print()
            print("  OPTION B (large image dataset — slower but richer):")
            print("    1. Download any ASL alphabet image dataset")
            print("    2. Organize as: downloads/images/A/*.jpg, downloads/images/B/*.jpg, etc.")
            print("    3. Re-run: python import_dataset.py")
            print()

            # Fall back to synthetic letters
            print("  For now, generating synthetic letter data as fallback...")
            from generate_dataset import ASL_SIGNS, generate_sample
            rng = np.random.default_rng(42)
            alpha_labels = sorted(l for l in ASL_SIGNS if not l.startswith("["))
            for label in alpha_labels:
                for _ in range(300):
                    features = generate_sample(label, rng)
                    letter_rows.append([label] + features)
                print(f"  {label}: 300 synthetic samples")

    if not letter_rows:
        print("Error: No letter data could be loaded.")
        return

    n_letters = len(set(row[0] for row in letter_rows))
    print(f"\n  Loaded {len(letter_rows)} letter samples across {n_letters} classes")

    # Generate word sign data
    print(f"\n[2/3] Generating word sign training data")
    word_rows = generate_word_signs(n_samples=500)
    n_words = len(set(row[0] for row in word_rows))

    # Combine and save
    print(f"\n[3/3] Combining and saving dataset")
    header = make_header()
    all_rows = [header] + letter_rows + word_rows

    with open(DATA_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(all_rows)

    total = len(all_rows) - 1
    print(f"\n{'=' * 60}")
    print(f"  Dataset saved to {DATA_FILE}")
    print(f"  {len(letter_rows)} letter samples ({n_letters} classes)")
    print(f"  {len(word_rows)} word samples ({n_words} classes)")
    print(f"  {total} total samples")
    print(f"{'=' * 60}")
    print(f"\nNext step: python train_model.py")


if __name__ == "__main__":
    main()
