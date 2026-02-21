"""
Shared feature engineering for ASL hand landmark data.

Fully vectorized with NumPy — no Python loops over landmarks.
~10x faster than the loop-based version at inference time.

Used by both train_model.py and asl_translator.py.
"""

import numpy as np

NUM_LANDMARKS = 21
FEATURES_PER_HAND = NUM_LANDMARKS * 3  # 63

# Landmark indices
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGERTIPS   = np.array([THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP])
FINGER_MCPS  = np.array([THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP])

# Each finger as [mcp/cmc, pip/ip, dip/ip, tip]
FINGER_CHAINS = np.array([
    [THUMB_CMC,  THUMB_MCP,   THUMB_IP,   THUMB_TIP],
    [INDEX_MCP,  INDEX_PIP,   INDEX_DIP,  INDEX_TIP],
    [MIDDLE_MCP, MIDDLE_PIP,  MIDDLE_DIP, MIDDLE_TIP],
    [RING_MCP,   RING_PIP,    RING_DIP,   RING_TIP],
    [PINKY_MCP,  PINKY_PIP,   PINKY_DIP,  PINKY_TIP],
])

PALM_INDICES = np.array([WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP])

# Precompute upper-triangle index pairs for inter-fingertip distances (10 pairs)
_TIP_I, _TIP_J = np.triu_indices(5, k=1)


def engineer_hand_features(pts):
    """
    Vectorized feature engineering for a single (21, 3) hand array.
    Returns a flat list of ~62 floats.
    All operations are batch NumPy — zero Python loops over landmarks.
    """
    feats = []

    tips = pts[FINGERTIPS]       # (5, 3)
    wrist = pts[WRIST]           # (3,)

    # ── 5: fingertip-to-wrist distances ──────────────────────────────────
    feats.append(np.linalg.norm(tips - wrist, axis=1))   # (5,)

    # ── 10: inter-fingertip distances (upper triangle) ───────────────────
    feats.append(np.linalg.norm(tips[_TIP_I] - tips[_TIP_J], axis=1))  # (10,)

    # ── 4: thumb tip to other fingertips ─────────────────────────────────
    feats.append(np.linalg.norm(tips[1:] - tips[0], axis=1))  # (4,)

    # ── 5: finger curl ratios ────────────────────────────────────────────
    # shape (5, 4, 3)
    chains = pts[FINGER_CHAINS]
    tip_to_base = np.linalg.norm(chains[:, -1] - chains[:, 0], axis=1)   # (5,)
    segment_vecs = np.diff(chains, axis=1)                                # (5, 3, 3)
    segment_lens = np.linalg.norm(segment_vecs, axis=2)                   # (5, 3)
    total_len = segment_lens.sum(axis=1)                                  # (5,)
    feats.append(tip_to_base / (total_len + 1e-8))                        # (5,)

    # ── 10: joint angles (2 per finger: at joints [0->1->2] and [1->2->3]) ──
    # angle at joint b = angle between vectors (a-b) and (c-b)
    def batch_angles(a, b, c):
        v1 = a - b
        v2 = c - b
        n1 = np.linalg.norm(v1, axis=1, keepdims=True)
        n2 = np.linalg.norm(v2, axis=1, keepdims=True)
        safe = (n1 > 1e-8) & (n2 > 1e-8)
        cos_a = np.where(safe.squeeze(), (v1 * v2).sum(1) / (n1 * n2).squeeze(), 0.0)
        return np.arccos(np.clip(cos_a, -1.0, 1.0))

    # Interleave: [finger0_angle1, finger0_angle2, finger1_angle1, ...] to match original
    a1 = batch_angles(chains[:, 0], chains[:, 1], chains[:, 2])  # (5,)
    a2 = batch_angles(chains[:, 1], chains[:, 2], chains[:, 3])  # (5,)
    feats.append(np.stack([a1, a2], axis=1).ravel())              # (10,) interleaved

    # ── 5: fingertip y-coordinates ───────────────────────────────────────
    feats.append(tips[:, 1])  # (5,)

    # ── 5: fingertip z-coordinates ───────────────────────────────────────
    feats.append(tips[:, 2])  # (5,)

    # ── 3: palm normal vector ────────────────────────────────────────────
    v1 = pts[INDEX_MCP] - wrist
    v2 = pts[PINKY_MCP] - wrist
    normal = np.cross(v1, v2)
    n_len = np.linalg.norm(normal)
    feats.append(normal / n_len if n_len > 1e-8 else normal)  # (3,)

    # ── 1: palm width ────────────────────────────────────────────────────
    feats.append([np.linalg.norm(pts[INDEX_MCP] - pts[PINKY_MCP])])  # (1,)

    # ── 5: finger spread angles ──────────────────────────────────────────
    tip_dirs = tips - wrist                              # (5, 3)
    norms = np.linalg.norm(tip_dirs, axis=1, keepdims=True)
    tip_dirs = np.where(norms > 1e-8, tip_dirs / norms, tip_dirs)
    # Adjacent pairs (4) + thumb-to-pinky (1)
    adj_cos = np.clip((tip_dirs[:-1] * tip_dirs[1:]).sum(1), -1.0, 1.0)
    tp_cos  = np.clip(np.dot(tip_dirs[0], tip_dirs[4]), -1.0, 1.0)
    feats.append(np.arccos(adj_cos))                     # (4,)
    feats.append([np.arccos(tp_cos)])                    # (1,)

    # ── 5: fingertip-to-palm-center distances ────────────────────────────
    palm_center = pts[PALM_INDICES].mean(axis=0)         # (3,)
    feats.append(np.linalg.norm(tips - palm_center, axis=1))  # (5,)

    # ── 4: adjacent fingertip distances ──────────────────────────────────
    feats.append(np.linalg.norm(np.diff(tips, axis=0), axis=1))  # (4,)

    return np.concatenate([np.ravel(f) for f in feats]).tolist()


def engineer_features_from_raw(raw_130):
    """
    Take a 130-dim raw landmark vector and return it concatenated with
    engineered features.

    Input:  130 floats [h1_landmarks(63), h2_landmarks(63), rel(3), num_hands(1)]
    Output: 130 + engineered features (~254 total)
    """
    raw = np.array(raw_130, dtype=np.float32)

    h1_coords = raw[:FEATURES_PER_HAND].reshape(NUM_LANDMARKS, 3)
    h2_coords = raw[FEATURES_PER_HAND:FEATURES_PER_HAND * 2].reshape(NUM_LANDMARKS, 3)
    meta      = raw[FEATURES_PER_HAND * 2:]

    h1_eng = engineer_hand_features(h1_coords)
    h2_eng = engineer_hand_features(h2_coords) if meta[3] > 0.5 else [0.0] * len(h1_eng)

    return raw.tolist() + h1_eng + h2_eng


def augment_sample(raw_130, rng, noise_scale=0.015):
    """
    Create an augmented copy of a 130-dim landmark sample.
    Applies small noise and random rotation to hand landmarks.
    """
    raw = np.array(raw_130, dtype=np.float64).copy()

    for hand_offset in [0, FEATURES_PER_HAND]:
        hand = raw[hand_offset:hand_offset + FEATURES_PER_HAND]
        if np.all(hand == 0):
            continue
        coords = hand.reshape(NUM_LANDMARKS, 3)
        coords += rng.normal(0, noise_scale, coords.shape)
        angle = rng.normal(0, 0.08)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
        coords = coords @ rot.T
        coords *= rng.normal(1.0, 0.03)
        max_dist = np.max(np.linalg.norm(coords, axis=1))
        if max_dist > 0:
            coords /= max_dist
        raw[hand_offset:hand_offset + FEATURES_PER_HAND] = coords.flatten()

    raw[FEATURES_PER_HAND * 2:FEATURES_PER_HAND * 2 + 3] += rng.normal(0, 0.01, 3)
    return raw.tolist()