"""
generate_motion_dataset.py
==========================
Generates synthetic training sequences for motion-based ASL signs.

Each sign is defined as a trajectory — a function that returns N keyframes
of hand landmark positions. Gaussian noise and augmentation is applied to
produce diverse training samples.

Motion Signs:
    J         — I-hand traces a J curve (down + left hook)
    Z         — Index finger traces Z (right, diagonal, right)
    [PLEASE]  — Flat hand circles on chest
    [THANK_YOU] — Fingers from chin moving outward
    [WHERE]   — Index finger wags side to side
    [HOW]     — Curved hands roll forward
    [COME]    — Index fingers curl inward
    [GO_AWAY] — Wrist flick outward
    [NAME]    — H-hand taps twice (repeated motion)
"""

import os
import csv
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = "data"
DATA_FILE = os.path.join(DATA_DIR, "motion_landmarks.csv")

os.makedirs(DATA_DIR, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
SEQ_LENGTH     = 30    # frames per sequence
NUM_LANDMARKS  = 21
FEATURE_DIM    = 126   # 63 * 2 hands (zeros for absent hand)
SAMPLES_PER_SIGN = 300
NOISE_SCALE    = 0.015

# ── Base hand (open, palm facing camera) ──────────────────────────────────────
# Simplified 21-landmark hand in normalized coords
# Index 0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky
def make_base_hand():
    h = np.zeros((21, 3))
    # Wrist
    h[0]  = [0.0,  0.0,  0.0]
    # Thumb (abducted)
    h[1]  = [-0.04,  -0.03, 0.0]
    h[2]  = [-0.07,  -0.06, 0.0]
    h[3]  = [-0.09,  -0.09, 0.0]
    h[4]  = [-0.10,  -0.12, 0.0]
    # Index
    h[5]  = [-0.02,  -0.07, 0.0]
    h[6]  = [-0.02,  -0.12, 0.0]
    h[7]  = [-0.02,  -0.16, 0.0]
    h[8]  = [-0.02,  -0.19, 0.0]
    # Middle
    h[9]  = [0.01,  -0.08, 0.0]
    h[10] = [0.01,  -0.13, 0.0]
    h[11] = [0.01,  -0.17, 0.0]
    h[12] = [0.01,  -0.20, 0.0]
    # Ring
    h[13] = [0.04,  -0.07, 0.0]
    h[14] = [0.04,  -0.12, 0.0]
    h[15] = [0.04,  -0.15, 0.0]
    h[16] = [0.04,  -0.18, 0.0]
    # Pinky
    h[17] = [0.07,  -0.06, 0.0]
    h[18] = [0.07,  -0.10, 0.0]
    h[19] = [0.07,  -0.13, 0.0]
    h[20] = [0.07,  -0.15, 0.0]
    return h

BASE_HAND = make_base_hand()

def curl_finger(h, base_idx, amount=0.9):
    """Curl a finger toward the palm."""
    for i in range(1, 4):
        h[base_idx + i][1] += amount * 0.04 * i
        h[base_idx + i][2] -= amount * 0.02 * i

def make_fist(h):
    for base in [5, 9, 13, 17]:
        curl_finger(h, base, 0.95)
    curl_finger(h, 1, 0.7)

def I_hand(h):
    """Only pinky extended."""
    make_fist(h)
    # Uncurl pinky
    for i in range(1, 4):
        h[17 + i] = BASE_HAND[17 + i].copy()

def index_point(h):
    """Only index finger extended."""
    make_fist(h)
    for i in range(1, 4):
        h[5 + i] = BASE_HAND[5 + i].copy()

def H_hand(h):
    """Index + middle extended, side by side."""
    make_fist(h)
    for i in range(1, 4):
        h[5 + i] = BASE_HAND[5 + i].copy()
        h[9 + i] = BASE_HAND[9 + i].copy()

def flat_hand(h):
    """All fingers extended."""
    h[:] = BASE_HAND.copy()

def flat_hand_curl(h, amount=0.4):
    """Slightly curved/claw hand."""
    h[:] = BASE_HAND.copy()
    for base in [5, 9, 13, 17]:
        curl_finger(h, base, amount)

# ── Trajectory generators ──────────────────────────────────────────────────────

def lerp(a, b, t):
    return a + (b - a) * t

def make_trajectory(keyframe_fn_list, n_frames=SEQ_LENGTH):
    """
    keyframe_fn_list: list of (t_norm, wrist_pos, hand_pose_fn)
    Returns list of (hand_array, wrist_pos) at each frame.
    """
    frames = []
    n_kf = len(keyframe_fn_list)
    for fi in range(n_frames):
        t = fi / (n_frames - 1)
        # find surrounding keyframes
        for ki in range(n_kf - 1):
            t0, w0, h0_fn = keyframe_fn_list[ki]
            t1, w1, h1_fn = keyframe_fn_list[ki + 1]
            if t0 <= t <= t1:
                alpha = (t - t0) / (t1 - t0 + 1e-8)
                w = lerp(np.array(w0), np.array(w1), alpha)
                h0 = make_base_hand(); h0_fn(h0)
                h1 = make_base_hand(); h1_fn(h1)
                h = lerp(h0, h1, alpha)
                frames.append((h, w))
                break
    return frames


def hand_to_features(h):
    """Normalize hand relative to wrist, scale by hand size."""
    pts = h.copy()
    wrist = pts[0].copy()
    pts -= wrist
    scale = np.linalg.norm(pts[9]) + 1e-6
    pts /= scale
    return pts.flatten()  # 63-dim


def build_sequence_features(traj_frames):
    """traj_frames: list of (hand, wrist). Returns (SEQ_LENGTH, 126) array."""
    seq = []
    for h, w in traj_frames:
        h1_feats = hand_to_features(h)
        h2_feats = np.zeros(63)
        seq.append(np.concatenate([h1_feats, h2_feats]))
    return seq  # list of 126-dim arrays


def build_two_hand_sequence(traj_frames):
    """traj_frames: list of (h1, w1, h2, w2). Returns list of 126-dim arrays."""
    seq = []
    for h1, w1, h2, w2 in traj_frames:
        seq.append(np.concatenate([hand_to_features(h1), hand_to_features(h2)]))
    return seq


# ── Sign Trajectories ─────────────────────────────────────────────────────────

def sign_J():
    """J: I-hand traces J shape (down, hook left)."""
    keyframes = [
        (0.0, [0.5, 0.3, 0.0], I_hand),   # start top
        (0.5, [0.5, 0.6, 0.0], I_hand),   # move down
        (0.75,[0.45, 0.65, 0.0], I_hand),  # hook left
        (1.0, [0.42, 0.6, 0.0], I_hand),  # hook curves back up
    ]
    return build_sequence_features(make_trajectory(keyframes))


def sign_Z():
    """Z: Index finger traces Z (right → diagonal → right)."""
    keyframes = [
        (0.0,  [0.3, 0.35, 0.0], index_point),  # top-left
        (0.3,  [0.7, 0.35, 0.0], index_point),  # → top-right
        (0.6,  [0.3, 0.65, 0.0], index_point),  # ↘ bottom-left
        (1.0,  [0.7, 0.65, 0.0], index_point),  # → bottom-right
    ]
    return build_sequence_features(make_trajectory(keyframes))


def sign_PLEASE():
    """PLEASE: Flat hand circles clockwise on chest."""
    frames = []
    for fi in range(SEQ_LENGTH):
        t = fi / (SEQ_LENGTH - 1)
        angle = t * 2 * np.pi
        cx, cy = 0.5, 0.55
        r = 0.08
        w = np.array([cx + r * np.cos(angle), cy + r * np.sin(angle), 0.0])
        h = make_base_hand()
        flat_hand(h)
        frames.append((h, w))
    return build_sequence_features(frames)


def sign_THANK_YOU():
    """THANK YOU: Flat hand from chin outward."""
    keyframes = [
        (0.0, [0.5, 0.25, 0.0], flat_hand),  # at chin
        (1.0, [0.5, 0.45, 0.05], flat_hand), # extend outward/down
    ]
    return build_sequence_features(make_trajectory(keyframes))


def sign_WHERE():
    """WHERE: Index finger wags side to side."""
    frames = []
    for fi in range(SEQ_LENGTH):
        t = fi / (SEQ_LENGTH - 1)
        x = 0.5 + 0.1 * np.sin(t * 4 * np.pi)  # wag ~2 times
        w = np.array([x, 0.4, 0.0])
        h = make_base_hand()
        index_point(h)
        frames.append((h, w))
    return build_sequence_features(frames)


def sign_HOW():
    """HOW: Curved hands roll forward."""
    # Two-hand sign — use single hand approximation for simplicity
    keyframes = [
        (0.0, [0.5, 0.5, 0.0], flat_hand_curl),
        (0.5, [0.5, 0.48, -0.05], flat_hand_curl),
        (1.0, [0.5, 0.45, 0.02], flat_hand_curl),
    ]
    return build_sequence_features(make_trajectory(keyframes))


def sign_COME():
    """COME: Index finger curls toward body (beckoning)."""
    keyframes = [
        (0.0, [0.5, 0.45, 0.0], index_point),
        (0.5, [0.5, 0.5, 0.02], index_point),
        (1.0, [0.5, 0.45, 0.0], index_point),
    ]
    # Curl index during motion
    def curled_index(h):
        make_fist(h)
        # Partially extend index
        h[6] = BASE_HAND[6] + np.array([0, -0.03, 0])
        h[7] = BASE_HAND[7] + np.array([0, -0.01, 0])
        h[8] = BASE_HAND[8] + np.array([0, 0.01, 0.02])

    frames_out = []
    for fi in range(SEQ_LENGTH):
        t = fi / (SEQ_LENGTH - 1)
        x = 0.5
        y = 0.45 + 0.05 * np.sin(t * 2 * np.pi)
        w = np.array([x, y, 0.0])
        h = make_base_hand()
        if t < 0.5:
            index_point(h)
        else:
            curled_index(h)
        frames_out.append((h, w))
    return build_sequence_features(frames_out)


def sign_GO_AWAY():
    """GO AWAY: Wrist flick outward."""
    keyframes = [
        (0.0, [0.5, 0.45, 0.0], flat_hand),
        (0.4, [0.5, 0.45, 0.0], flat_hand_curl),
        (1.0, [0.6, 0.4, -0.03], flat_hand_curl),
    ]
    return build_sequence_features(make_trajectory(keyframes))


def sign_NAME():
    """NAME: H-hand taps twice."""
    frames = []
    for fi in range(SEQ_LENGTH):
        t = fi / (SEQ_LENGTH - 1)
        # Two taps = two dips
        y_offset = 0.03 * abs(np.sin(t * 4 * np.pi))
        w = np.array([0.5, 0.45 + y_offset, 0.0])
        h = make_base_hand()
        H_hand(h)
        frames.append((h, w))
    return build_sequence_features(frames)


# ── Sign registry ──────────────────────────────────────────────────────────────
MOTION_SIGNS = {
    "J":           sign_J,
    "Z":           sign_Z,
    "[PLEASE]":    sign_PLEASE,
    "[THANK_YOU]": sign_THANK_YOU,
    "[WHERE]":     sign_WHERE,
    "[HOW]":       sign_HOW,
    "[COME]":      sign_COME,
    "[GO_AWAY]":   sign_GO_AWAY,
    "[NAME]":      sign_NAME,
}

# ── Augmentation ───────────────────────────────────────────────────────────────
def augment_sequence(seq, rng):
    """Apply noise, scaling, and time-warp to a sequence."""
    seq = np.array(seq)

    # Gaussian noise
    seq += rng.normal(0, NOISE_SCALE, seq.shape)

    # Random scale
    scale = rng.uniform(0.85, 1.15)
    seq *= scale

    # Time warp: randomly stretch/shrink by resampling
    n = len(seq)
    warp_pts = np.linspace(0, n-1, n) + rng.uniform(-0.5, 0.5, n)
    warp_pts = np.clip(warp_pts, 0, n-1)
    idx = np.round(warp_pts).astype(int)
    seq = seq[idx]

    return seq.tolist()


# ── Main ───────────────────────────────────────────────────────────────────────
def make_header():
    return ["label", "frame"] + [f"f{i}" for i in range(FEATURE_DIM)]


def main():
    rng = np.random.default_rng(42)
    header = make_header()
    rows = [header]

    print(f"Generating {SAMPLES_PER_SIGN} sequences × {len(MOTION_SIGNS)} signs...")

    for label, sign_fn in MOTION_SIGNS.items():
        kind = "word" if label.startswith("[") else "letter"
        for _ in range(SAMPLES_PER_SIGN):
            base_seq = sign_fn()
            aug_seq  = augment_sequence(base_seq, rng)
            for fi, frame_feats in enumerate(aug_seq):
                rows.append([label, fi] + list(frame_feats))
        print(f"  {label:16s} ({kind}): {SAMPLES_PER_SIGN} sequences")

    with open(DATA_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    total_seqs = SAMPLES_PER_SIGN * len(MOTION_SIGNS)
    print(f"\nSaved {total_seqs} sequences ({total_seqs * SEQ_LENGTH} rows) to {DATA_FILE}")
    print("Next step: python train_motion_model.py")


if __name__ == "__main__":
    main()
