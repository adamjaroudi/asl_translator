"""
Train an ASL classifier with feature engineering and data augmentation.

Uses LinearSVC wrapped in CalibratedClassifierCV for fast inference.

Optimizations vs original:
  - AUG_MULTIPLIER reduced 8 -> 4  (half the augmented data)
  - cv=2 instead of 3              (3 fits instead of 4)
  - n_jobs=-1                      (uses all CPU cores)
  - Batch feature engineering with progress reporting

Usage:
    python train_model.py
"""

import argparse
import os
import csv
import time
import pickle
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from features import engineer_features_from_raw, augment_sample

DATA_FILE  = os.path.join("data", "landmarks.csv")
MODEL_DIR  = "model"
MODEL_FILE = os.path.join(MODEL_DIR, "asl_classifier.pkl")

TARGET_SAMPLES_PER_CLASS = 500
AUG_MULTIPLIER           = 8
# Stronger SVM (C=1.0) can improve accuracy when using --accuracy
ACCURACY_SVC_C           = 1.0


def load_data():
    with open(DATA_FILE, "r") as f:
        rows = list(csv.reader(f))
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

        n_needed = max(0, TARGET_SAMPLES_PER_CLASS - len(samples))
        n_aug    = max(n_needed, len(samples) * AUG_MULTIPLIER)

        for _ in range(n_aug):
            src = samples[rng.integers(len(samples))]
            X_aug.append(augment_sample(src, rng))
            y_aug.append(label)

    return X_aug, y_aug


def batch_engineer(X_list, label="samples"):
    """Engineer features with progress updates every 10%."""
    n = len(X_list)
    out = []
    report_every = max(1, n // 10)
    t0 = time.time()
    for i, x in enumerate(X_list):
        out.append(engineer_features_from_raw(x))
        if (i + 1) % report_every == 0 or i == n - 1:
            pct     = (i + 1) / n * 100
            elapsed = time.time() - t0
            eta     = (elapsed / (i + 1)) * (n - i - 1)
            print(f"  {pct:5.1f}%  ({i+1}/{n})  "
                  f"elapsed {elapsed:.0f}s  eta {eta:.0f}s", flush=True)
    return np.array(out, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Train ASL classifier.")
    parser.add_argument("--accuracy", action="store_true",
                        help="Use stronger SVM (C=1.0) for better recognition accuracy")
    args = parser.parse_args()

    svc_c = ACCURACY_SVC_C if args.accuracy else 0.5
    if args.accuracy:
        print("Accuracy mode: training with C=1.0 for better recognition.\n")

    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        print("Run 'python import_dataset.py' or 'python generate_dataset.py' first.")
        return

    t_start = time.time()

    print("Loading data...")
    X_raw, y = load_data()
    classes   = sorted(set(y))
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

    print(f"\nEngineering features for {len(X_train_aug)} training samples...")
    X_train = batch_engineer(X_train_aug)

    print(f"\nEngineering features for {len(X_test_raw)} test samples...")
    X_test = batch_engineer(X_test_raw)

    print(f"\nFeature dimensions: 130 -> {X_train.shape[1]}")
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set:     {len(X_test)} samples")

    print("\nTraining LinearSVC classifier (cv=2, n_jobs=-1)...")
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", CalibratedClassifierCV(
            LinearSVC(C=svc_c, max_iter=3000, random_state=42, dual="auto"),
            cv=2,
            n_jobs=-1,
        )),
    ])
    clf.fit(X_train, y_train_aug)

    y_pred   = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nTest Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Confusion matrix — only show meaningful mistakes
    from sklearn.metrics import confusion_matrix
    classes_list = sorted(set(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=classes_list)
    errors = []
    for i, true_cls in enumerate(classes_list):
        for j, pred_cls in enumerate(classes_list):
            if i != j and cm[i][j] > 0:
                errors.append((cm[i][j], true_cls, pred_cls))
    if errors:
        errors.sort(reverse=True)
        print("\nTop confusions (true -> predicted):")
        for count, t, p in errors[:10]:
            print(f"  {t:<20} -> {p:<20} ({count}x)")

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(clf, f)

    # Emit structured result for frontend confusion matrix display
    import json as _json
    per_class_acc = {}
    for cls in classes_list:
        mask = y_test == cls
        if mask.sum() > 0:
            per_class_acc[cls] = float((y_pred[mask] == cls).mean())
    top_confusions = [
        {"true": t, "pred": p, "count": int(c)}
        for c, t, p in sorted(errors, reverse=True)[:15]
    ] if errors else []
    print(f"TRAIN_RESULT:{_json.dumps({'accuracy': float(accuracy), 'per_class': per_class_acc, 'confusions': top_confusions})}")

    elapsed = time.time() - t_start
    print(f"\nTotal training time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Model saved to {MODEL_FILE}")
    print("Restart server.py to load the new model.")
    print("Tip: For best accuracy, collect your own data then retrain with --accuracy")


if __name__ == "__main__":
    main()