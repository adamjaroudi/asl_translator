"""
Train an ASL classifier optimised for real-time inference.

Uses a LinearSVC wrapped in CalibratedClassifierCV so we still get
predict_proba(), but inference is ~13,000x faster than HistGradientBoosting
(0.04 ms vs 525 ms per single-sample call).

Usage:
    python train_model.py
"""

import os
import csv
import pickle
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from features import engineer_features_from_raw, augment_sample

DATA_FILE = os.path.join("data", "landmarks.csv")
MODEL_DIR  = "model"
MODEL_FILE = os.path.join(MODEL_DIR, "asl_classifier.pkl")

TARGET_SAMPLES_PER_CLASS = 500
AUG_MULTIPLIER           = 8


def load_data():
    with open(DATA_FILE, "r") as f:
        reader = csv.reader(f)
        rows   = list(reader)
    data     = rows[1:]
    labels   = [row[0] for row in data]
    features = [list(map(float, row[1:])) for row in data]
    return features, labels


def augment_and_balance(X_raw, y, rng):
    by_class = {}
    for feat, label in zip(X_raw, y):
        by_class.setdefault(label, []).append(feat)

    X_aug, y_aug = [], []
    for label, samples in by_class.items():
        X_aug.extend(samples)
        y_aug.extend([label] * len(samples))

        n_aug = max(
            TARGET_SAMPLES_PER_CLASS - len(samples),
            len(samples) * AUG_MULTIPLIER,
        )
        for _ in range(n_aug):
            src = samples[rng.integers(len(samples))]
            X_aug.append(augment_sample(src, rng))
            y_aug.append(label)

    return X_aug, y_aug


def main():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        print("Run 'python import_dataset.py' or 'python generate_dataset.py' first.")
        return

    print("Loading data...")
    X_raw, y = load_data()
    classes  = sorted(set(y))
    n_letters = sum(1 for c in classes if not c.startswith("["))
    n_words   = sum(1 for c in classes if c.startswith("["))
    print(f"Loaded {len(X_raw)} samples, {len(classes)} classes "
          f"({n_letters} letters + {n_words} words)")

    unique, counts = np.unique(y, return_counts=True)
    print("\nSamples per class:")
    for label, count in zip(unique, counts):
        kind = "word" if label.startswith("[") else "letter"
        print(f"  {label:18s} ({kind}): {count}")

    print("\nSplitting data...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Augmenting training data...")
    rng = np.random.default_rng(42)
    X_train_aug, y_train_aug = augment_and_balance(X_train_raw, y_train, rng)
    print(f"  Training samples: {len(y_train)} -> {len(y_train_aug)}")

    print("Computing engineered features...")
    X_train = np.array([engineer_features_from_raw(x) for x in X_train_aug], dtype=np.float32)
    X_test  = np.array([engineer_features_from_raw(x) for x in X_test_raw],  dtype=np.float32)
    print(f"  Feature dimensions: {X_train.shape[1]}")

    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set:     {len(X_test)} samples")

    # ── LinearSVC + Platt scaling for calibrated probabilities ──
    # LinearSVC.decision_function = 0.04 ms per call (vs 525 ms for HistGB)
    # CalibratedClassifierCV adds a tiny logistic regression on top — still <0.1 ms
    print("\nTraining LinearSVC classifier (fast inference)...")
    base = LinearSVC(
        C=0.5,
        max_iter=3000,
        random_state=42,
        dual="auto",
    )
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    CalibratedClassifierCV(base, cv=3, method="sigmoid")),
    ])
    clf.fit(X_train, y_train_aug)

    y_pred   = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nTest Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # ── Inference speed check ──
    import time
    dummy = X_test[:1]
    t = time.perf_counter()
    for _ in range(1000):
        clf.predict_proba(dummy)
    ms = (time.perf_counter() - t) / 1000 * 1000
    print(f"Inference speed: {ms:.3f} ms per call")

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(clf, f)

    print(f"\nModel saved to {MODEL_FILE}")
    print("Now run: python asl_translator.py")


if __name__ == "__main__":
    main()