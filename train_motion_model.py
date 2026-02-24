"""
train_motion_model.py
─────────────────────
Trains a motion sign classifier from data/motion_landmarks.csv and saves
the model to model/motion_classifier.pkl.

Usage:
    python train_motion_model.py

Data format (motion_landmarks.csv):
    label, frame, f0, f1, ..., f125
    Each sign produces SEQ_LENGTH (30) rows — one per frame.

Model format saved:
    {
        "model":      sklearn estimator with predict_proba(),
        "model_type": "RF",
        "classes":    list of class labels,
        "seq_length": 30,
        "feat_dim":   126,
        "accuracy":   float,
    }
"""

import os
import sys
import pickle
import collections

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

# ── Constants (must match server.py) ─────────────────────────────────────────
SEQ_LENGTH = 30
FEAT_DIM   = 126     # 63 per hand × 2
DATA_FILE  = os.path.join("data", "motion_landmarks.csv")
MODEL_FILE = os.path.join("model", "motion_classifier.pkl")


# ── Feature engineering (must match server.py engineer_sequence) ──────────────
def engineer_sequence_flat(seq_arr):
    """(SEQ_LENGTH, 126) → flat vector for RF"""
    X      = seq_arr[np.newaxis]                                     # (1, T, 126)
    deltas = np.diff(X, axis=1, prepend=X[:, :1, :])                # (1, T, 126)
    mags   = np.linalg.norm(deltas, axis=2, keepdims=True)          # (1, T, 1)
    engineered = np.concatenate([X, deltas, mags], axis=2)          # (1, T, 253)
    return engineered.reshape(1, -1)                                 # (1, T*253)


# ── Load CSV ───────────────────────────────────────────────────────────────────
def load_data(path):
    if not os.path.exists(path):
        print(f"[error] Data file not found: {path}")
        print("        Collect motion data first using the 'Motion Signs' tab in Collect Data.")
        sys.exit(1)

    sequences   = collections.defaultdict(list)   # label → list of frames
    current_seq = collections.defaultdict(list)   # label → current building sequence
    row_counts  = collections.Counter()

    with open(path, "r") as f:
        header = f.readline().strip().split(",")
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 2 + FEAT_DIM:
                continue
            label     = parts[0]
            # parts[1] is frame index
            features  = [float(v) for v in parts[2:2 + FEAT_DIM]]
            current_seq[label].append(features)
            row_counts[label] += 1

            if len(current_seq[label]) == SEQ_LENGTH:
                sequences[label].append(np.array(current_seq[label], dtype=np.float32))
                current_seq[label] = []

    if not sequences:
        print("[error] No complete sequences found in data file.")
        print(f"        Make sure each sign has at least {SEQ_LENGTH} rows.")
        sys.exit(1)

    # Report
    for label, seqs in sorted(sequences.items()):
        display = label.strip("[]").replace("_", " ").replace("-", " ").title()
        print(f"  {display:<20} {len(seqs):>4} sequences")

    return sequences


# ── Build X, y ────────────────────────────────────────────────────────────────
def build_dataset(sequences):
    X_list, y_list = [], []
    for label, seqs in sequences.items():
        for seq in seqs:
            X_list.append(engineer_sequence_flat(seq).flatten())
            y_list.append(label)
    return np.array(X_list, dtype=np.float32), np.array(y_list)


# ── Train ─────────────────────────────────────────────────────────────────────
def train(X, y):
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    classes = sorted(set(y))
    n_classes = len(classes)
    n_samples = len(y)

    print(f"\n[train] {n_samples} samples · {n_classes} classes · {X.shape[1]} features")

    # Split
    if n_samples >= 10 and n_classes >= 2:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(sss.split(X, y))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    else:
        X_train, X_test = X, X
        y_train, y_test = y, y
        print("[warn]  Not enough data for a proper train/test split — using all data.")

    # Use RF — fast, no GPU needed, good for small datasets
    print("[train] Fitting RandomForest...")
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test)
    print(f"[train] Test accuracy: {acc * 100:.1f}%")

    if acc < 0.5 and n_samples > 20:
        print("[warn]  Low accuracy — collect more sequences (aim for 50+ per sign).")

    # Per-class accuracy
    from sklearn.metrics import classification_report, confusion_matrix
    y_pred = clf.predict(X_test)
    print("\n[train] Per-sign accuracy:")
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    for cls in classes:
        display = cls.strip("[]").replace("_", " ").replace("-", " ").title()
        r = report.get(cls, {})
        prec = r.get("precision", 0) * 100
        rec  = r.get("recall",    0) * 100
        sup  = int(r.get("support", 0))
        bar  = "#" * int(rec / 10)
        flag = " <-- low" if rec < 60 and sup > 0 else ""
        print(f"  {display:<14} precision {prec:4.0f}%  recall {rec:4.0f}%  [{bar:<10}]  n={sup}{flag}")

    # Confusion matrix (only show off-diagonal mistakes)
    cm     = confusion_matrix(y_test, y_pred, labels=classes)
    errors = []
    for i, true_cls in enumerate(classes):
        for j, pred_cls in enumerate(classes):
            if i != j and cm[i][j] > 0:
                t = true_cls.strip("[]").replace("_"," ").title()
                p = pred_cls.strip("[]").replace("_"," ").title()
                errors.append((cm[i][j], t, p))
    if errors:
        errors.sort(reverse=True)
        print("\n[train] Top confusions (true -> predicted):")
        for count, t, p in errors[:8]:
            print(f"  {t:<14} -> {p:<14} ({count}x)")

    return clf, classes, acc, report, errors


# ── Save ──────────────────────────────────────────────────────────────────────
def save_model(clf, classes, acc):
    os.makedirs("model", exist_ok=True)
    payload = {
        "model":      clf,
        "model_type": "RF",
        "classes":    classes,
        "seq_length": SEQ_LENGTH,
        "feat_dim":   FEAT_DIM,
        "accuracy":   acc,
    }
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(payload, f)
    print(f"[train] Model saved -> {MODEL_FILE}")
    print(f"[train] Classes: {', '.join(c.strip('[]').replace('_',' ') for c in classes)}")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  Motion Sign Classifier — Training")
    print("=" * 50)
    print(f"\n[data]  Loading {DATA_FILE}...")

    sequences = load_data(DATA_FILE)
    X, y      = build_dataset(sequences)
    clf, classes, acc, report, errors = train(X, y)
    save_model(clf, classes, acc)

    # Emit structured result for frontend display
    import json as _json
    per_class_acc = {
        cls: float(report.get(cls, {}).get("recall", 0))
        for cls in classes
    }
    top_confusions = [
        {"true": t, "pred": p, "count": int(c)}
        for c, t, p in sorted(errors, reverse=True)[:15]
    ] if errors else []
    print(f"TRAIN_RESULT:{_json.dumps({'accuracy': float(acc), 'per_class': per_class_acc, 'confusions': top_confusions})}")

    print("\n[done]  Restart the server to load the new model.")
    print("=" * 50)
