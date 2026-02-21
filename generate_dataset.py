"""
Generate a synthetic ASL training dataset with both single-hand alphabet signs
and two-hand word signs. Uses MediaPipe-style hand landmark geometry.

Feature vector (130 dims):
  [0:63]    hand 1 landmarks (left by x-pos), normalized to wrist
  [63:126]  hand 2 landmarks (right by x-pos), zeros if 1 hand
  [126:129] relative wrist offset (hand2 - hand1), zeros if 1 hand
  [129]     num_hands flag (0.0 = one hand, 1.0 = two hands)

Usage:
    python generate_dataset.py
"""

import os
import csv
import numpy as np

DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "landmarks.csv")
NUM_LANDMARKS = 21
FEATURES_PER_HAND = NUM_LANDMARKS * 3  # 63
TOTAL_FEATURES = FEATURES_PER_HAND * 2 + 3 + 1  # 130
SAMPLES_PER_SIGN = 300
NOISE_SCALE = 0.04

WRIST = 0
THUMB = [1, 2, 3, 4]
INDEX = [5, 6, 7, 8]
MIDDLE = [9, 10, 11, 12]
RING = [13, 14, 15, 16]
PINKY = [17, 18, 19, 20]
FINGERS = [THUMB, INDEX, MIDDLE, RING, PINKY]

BASE_HAND = np.array([
    [0.0, 0.0, 0.0],       # 0  WRIST
    [-0.05, -0.04, -0.01],  # 1  THUMB_CMC
    [-0.09, -0.08, -0.02],  # 2  THUMB_MCP
    [-0.12, -0.12, -0.02],  # 3  THUMB_IP
    [-0.14, -0.16, -0.02],  # 4  THUMB_TIP
    [-0.04, -0.18, 0.0],    # 5  INDEX_MCP
    [-0.04, -0.26, 0.0],    # 6  INDEX_PIP
    [-0.04, -0.32, 0.0],    # 7  INDEX_DIP
    [-0.04, -0.36, 0.0],    # 8  INDEX_TIP
    [0.0, -0.18, 0.0],      # 9  MIDDLE_MCP
    [0.0, -0.27, 0.0],      # 10 MIDDLE_PIP
    [0.0, -0.33, 0.0],      # 11 MIDDLE_DIP
    [0.0, -0.37, 0.0],      # 12 MIDDLE_TIP
    [0.04, -0.17, 0.0],     # 13 RING_MCP
    [0.04, -0.25, 0.0],     # 14 RING_PIP
    [0.04, -0.30, 0.0],     # 15 RING_DIP
    [0.04, -0.34, 0.0],     # 16 RING_TIP
    [0.08, -0.15, 0.0],     # 17 PINKY_MCP
    [0.08, -0.21, 0.0],     # 18 PINKY_PIP
    [0.08, -0.25, 0.0],     # 19 PINKY_DIP
    [0.08, -0.28, 0.0],     # 20 PINKY_TIP
], dtype=np.float64)


def mirror_hand(hand):
    """Mirror a right hand to make a left hand (flip x-axis)."""
    mirrored = hand.copy()
    mirrored[:, 0] = -mirrored[:, 0]
    return mirrored


def curl_finger(hand, finger_indices, amount):
    mcp = finger_indices[0]
    for i, idx in enumerate(finger_indices[1:], 1):
        base = hand[mcp].copy()
        vec = hand[idx] - base
        angle = amount * (0.4 + 0.2 * i)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        new_y = vec[1] * cos_a - vec[2] * sin_a
        new_z = vec[1] * sin_a + vec[2] * cos_a
        hand[idx] = base + np.array([vec[0], new_y, new_z])
        hand[idx][1] = base[1] + (hand[idx][1] - base[1]) * max(0.3, 1.0 - amount * 0.4)
        hand[idx][2] = base[2] + abs(amount) * 0.03 * i


def spread_finger(hand, finger_indices, amount):
    for idx in finger_indices[1:]:
        hand[idx][0] += amount


def make_fist(hand):
    for f in FINGERS:
        curl_finger(hand, f, 1.0)


def normalize_hand(hand):
    coords = hand - hand[0]
    max_dist = np.max(np.linalg.norm(coords, axis=1))
    if max_dist > 0:
        coords = coords / max_dist
    return coords.flatten().tolist()


def build_feature_vector(hand1, hand2=None, hand1_wrist_pos=None, hand2_wrist_pos=None):
    """Build the 130-dim feature vector from one or two hands."""
    h1_features = normalize_hand(hand1)

    if hand2 is not None:
        h2_features = normalize_hand(hand2)
        w1 = hand1_wrist_pos if hand1_wrist_pos is not None else np.zeros(3)
        w2 = hand2_wrist_pos if hand2_wrist_pos is not None else np.zeros(3)
        rel = (w2 - w1).tolist()
        num_hands = 1.0
    else:
        h2_features = [0.0] * FEATURES_PER_HAND
        rel = [0.0, 0.0, 0.0]
        num_hands = 0.0

    return h1_features + h2_features + rel + [num_hands]


# ---------------------------------------------------------------------------
# Sign definitions: each returns (hand1, hand2_or_None, wrist1, wrist2)
# hand1 = left hand (lower x in camera view), hand2 = right hand
# For single-hand signs, hand2 is None
# ---------------------------------------------------------------------------
ASL_SIGNS = {}


def define_sign(label, pose_fn):
    ASL_SIGNS[label] = pose_fn


# ==================== ALPHABET (single hand) ====================

def _single(pose_fn):
    """Wrapper: single-hand sign returns (hand, None, origin, None)."""
    def wrapped():
        hand = BASE_HAND.copy()
        pose_fn(hand)
        return hand, None, np.array([0.5, 0.5, 0.0]), None
    return wrapped


def pose_a(h):
    make_fist(h)
    h[4] = h[1] + np.array([-0.03, -0.06, -0.03])

define_sign("A", _single(pose_a))


def pose_b(h):
    curl_finger(h, THUMB, 0.9)

define_sign("B", _single(pose_b))


def pose_c(h):
    for f in FINGERS:
        curl_finger(h, f, 0.4)
    h[4][0] -= 0.02

define_sign("C", _single(pose_c))


def pose_d(h):
    for f in [MIDDLE, RING, PINKY]:
        curl_finger(h, f, 0.95)
    curl_finger(h, THUMB, 0.6)

define_sign("D", _single(pose_d))


def pose_e(h):
    for f in [INDEX, MIDDLE, RING, PINKY]:
        curl_finger(h, f, 0.7)
    curl_finger(h, THUMB, 0.5)

define_sign("E", _single(pose_e))


def pose_f(h):
    curl_finger(h, INDEX, 0.8)
    curl_finger(h, THUMB, 0.7)

define_sign("F", _single(pose_f))


def pose_g(h):
    for f in [MIDDLE, RING, PINKY]:
        curl_finger(h, f, 0.95)
    curl_finger(h, THUMB, 0.3)
    for i in range(len(h)):
        h[i] = np.array([h[i][1], -h[i][0], h[i][2]])

define_sign("G", _single(pose_g))


def pose_h(h):
    for f in [RING, PINKY]:
        curl_finger(h, f, 0.95)
    curl_finger(h, THUMB, 0.7)
    for i in range(len(h)):
        h[i] = np.array([h[i][1], -h[i][0], h[i][2]])

define_sign("H", _single(pose_h))


def pose_i(h):
    for f in [THUMB, INDEX, MIDDLE, RING]:
        curl_finger(h, f, 0.95)

define_sign("I", _single(pose_i))


def pose_j(h):
    pose_i(h)
    h[20][0] += 0.03
    h[20][1] += 0.02

define_sign("J", _single(pose_j))


def pose_k(h):
    for f in [RING, PINKY]:
        curl_finger(h, f, 0.95)
    curl_finger(h, THUMB, 0.4)
    spread_finger(h, INDEX, -0.02)
    spread_finger(h, MIDDLE, 0.02)

define_sign("K", _single(pose_k))


def pose_l(h):
    for f in [MIDDLE, RING, PINKY]:
        curl_finger(h, f, 0.95)

define_sign("L", _single(pose_l))


def pose_m(h):
    make_fist(h)
    h[8][1] = h[5][1] + 0.01
    h[8][2] = -0.04
    h[12][1] = h[9][1] + 0.01
    h[12][2] = -0.04
    h[16][1] = h[13][1] + 0.01
    h[16][2] = -0.04

define_sign("M", _single(pose_m))


def pose_n(h):
    make_fist(h)
    h[8][1] = h[5][1] + 0.01
    h[8][2] = -0.04
    h[12][1] = h[9][1] + 0.01
    h[12][2] = -0.04

define_sign("N", _single(pose_n))


def pose_o(h):
    for f in FINGERS:
        curl_finger(h, f, 0.55)
    h[4][0] = h[8][0]
    h[4][1] = h[8][1]

define_sign("O", _single(pose_o))


def pose_p(h):
    pose_k(h)
    for i in range(len(h)):
        h[i][1] = -h[i][1]

define_sign("P", _single(pose_p))


def pose_q(h):
    pose_g(h)
    for i in range(len(h)):
        h[i][1] = -h[i][1]

define_sign("Q", _single(pose_q))


def pose_r(h):
    for f in [RING, PINKY]:
        curl_finger(h, f, 0.95)
    curl_finger(h, THUMB, 0.8)
    h[8][0] = h[12][0]

define_sign("R", _single(pose_r))


def pose_s(h):
    make_fist(h)
    h[4][1] = h[6][1]
    h[4][2] = -0.03

define_sign("S", _single(pose_s))


def pose_t(h):
    make_fist(h)
    h[4][1] = h[6][1]
    h[4][2] = -0.04
    h[4][0] = h[6][0]

define_sign("T", _single(pose_t))


def pose_u(h):
    for f in [RING, PINKY]:
        curl_finger(h, f, 0.95)
    curl_finger(h, THUMB, 0.8)

define_sign("U", _single(pose_u))


def pose_v(h):
    for f in [RING, PINKY]:
        curl_finger(h, f, 0.95)
    curl_finger(h, THUMB, 0.8)
    spread_finger(h, INDEX, -0.03)
    spread_finger(h, MIDDLE, 0.03)

define_sign("V", _single(pose_v))


def pose_w(h):
    curl_finger(h, PINKY, 0.95)
    curl_finger(h, THUMB, 0.8)
    spread_finger(h, INDEX, -0.03)
    spread_finger(h, MIDDLE, 0.0)
    spread_finger(h, RING, 0.03)

define_sign("W", _single(pose_w))


def pose_x(h):
    for f in [MIDDLE, RING, PINKY]:
        curl_finger(h, f, 0.95)
    curl_finger(h, INDEX, 0.5)
    curl_finger(h, THUMB, 0.7)

define_sign("X", _single(pose_x))


def pose_y(h):
    for f in [INDEX, MIDDLE, RING]:
        curl_finger(h, f, 0.95)

define_sign("Y", _single(pose_y))


def pose_z(h):
    for f in [MIDDLE, RING, PINKY]:
        curl_finger(h, f, 0.95)
    curl_finger(h, THUMB, 0.7)
    h[8][0] += 0.02

define_sign("Z", _single(pose_z))


# ==================== WORDS (single and two-hand) ====================

def sign_ily():
    """I-LOVE-YOU: thumb + index + pinky extended, middle + ring curled."""
    h = BASE_HAND.copy()
    curl_finger(h, MIDDLE, 0.95)
    curl_finger(h, RING, 0.95)
    spread_finger(h, PINKY, 0.02)
    spread_finger(h, INDEX, -0.02)
    return h, None, np.array([0.5, 0.5, 0.0]), None

define_sign("[I-LOVE-YOU]", sign_ily)


def sign_thumbs_up():
    """THUMBS-UP / GOOD / LIKE: thumb up, all fingers curled."""
    h = BASE_HAND.copy()
    for f in [INDEX, MIDDLE, RING, PINKY]:
        curl_finger(h, f, 1.0)
    # rotate thumb to point straight up
    h[2] = h[1] + np.array([0.0, -0.05, -0.02])
    h[3] = h[1] + np.array([0.0, -0.10, -0.02])
    h[4] = h[1] + np.array([0.0, -0.14, -0.02])
    return h, None, np.array([0.5, 0.5, 0.0]), None

define_sign("[GOOD]", sign_thumbs_up)


def sign_more():
    """MORE: Both hands in flat-O/bunched shape, fingertips touching."""
    left = BASE_HAND.copy()
    for f in FINGERS:
        curl_finger(left, f, 0.55)
    left[4][0] = left[8][0]
    left[4][1] = left[8][1]

    right = mirror_hand(BASE_HAND.copy())
    for f in FINGERS:
        curl_finger(right, f, 0.55)
    right[4][0] = right[8][0]
    right[4][1] = right[8][1]

    w1 = np.array([0.4, 0.5, 0.0])
    w2 = np.array([0.6, 0.5, 0.0])
    return left, right, w1, w2

define_sign("[MORE]", sign_more)


def sign_help():
    """HELP: Dominant fist resting on non-dominant open palm."""
    left = BASE_HAND.copy()  # open palm (base is already open)
    curl_finger(left, THUMB, 0.3)

    right = mirror_hand(BASE_HAND.copy())
    make_fist(right)
    right[4] = right[1] + np.array([0.03, -0.06, -0.03])

    w1 = np.array([0.45, 0.55, 0.0])
    w2 = np.array([0.50, 0.45, 0.0])
    return left, right, w1, w2

define_sign("[HELP]", sign_help)


def sign_book():
    """BOOK: Two open palms together, like opening a book."""
    left = BASE_HAND.copy()
    curl_finger(left, THUMB, 0.3)

    right = mirror_hand(BASE_HAND.copy())
    curl_finger(right, THUMB, 0.3)

    w1 = np.array([0.42, 0.5, 0.0])
    w2 = np.array([0.58, 0.5, 0.0])
    return left, right, w1, w2

define_sign("[BOOK]", sign_book)


def sign_stop():
    """STOP: One flat hand edge hits the other open palm."""
    left = BASE_HAND.copy()  # open palm horizontal
    curl_finger(left, THUMB, 0.3)

    right = mirror_hand(BASE_HAND.copy())  # perpendicular blade
    curl_finger(right, THUMB, 0.3)
    # rotate right hand 90 degrees to be perpendicular
    for i in range(len(right)):
        right[i] = np.array([right[i][2], right[i][1], -right[i][0]])

    w1 = np.array([0.45, 0.55, 0.0])
    w2 = np.array([0.50, 0.45, -0.02])
    return left, right, w1, w2

define_sign("[STOP]", sign_stop)


def sign_play():
    """PLAY: Both hands in Y-shape (thumb + pinky), shaking."""
    left = BASE_HAND.copy()
    for f in [INDEX, MIDDLE, RING]:
        curl_finger(left, f, 0.95)

    right = mirror_hand(BASE_HAND.copy())
    for f in [INDEX, MIDDLE, RING]:
        curl_finger(right, f, 0.95)

    w1 = np.array([0.38, 0.5, 0.0])
    w2 = np.array([0.62, 0.5, 0.0])
    return left, right, w1, w2

define_sign("[PLAY]", sign_play)


def sign_want():
    """WANT: Both hands in claw/5-shape, fingers spread and slightly curled."""
    left = BASE_HAND.copy()
    for f in FINGERS:
        curl_finger(left, f, 0.3)
    spread_finger(left, INDEX, -0.02)
    spread_finger(left, RING, 0.02)
    spread_finger(left, PINKY, 0.03)

    right = mirror_hand(BASE_HAND.copy())
    for f in FINGERS:
        curl_finger(right, f, 0.3)
    spread_finger(right, INDEX, 0.02)
    spread_finger(right, RING, -0.02)
    spread_finger(right, PINKY, -0.03)

    w1 = np.array([0.38, 0.5, 0.0])
    w2 = np.array([0.62, 0.5, 0.0])
    return left, right, w1, w2

define_sign("[WANT]", sign_want)


def sign_with():
    """WITH: Two A-fists brought together."""
    left = BASE_HAND.copy()
    make_fist(left)
    left[4] = left[1] + np.array([-0.03, -0.06, -0.03])

    right = mirror_hand(BASE_HAND.copy())
    make_fist(right)
    right[4] = right[1] + np.array([0.03, -0.06, -0.03])

    w1 = np.array([0.45, 0.5, 0.0])
    w2 = np.array([0.55, 0.5, 0.0])
    return left, right, w1, w2

define_sign("[WITH]", sign_with)


def sign_same():
    """SAME/ALSO: Both hands index finger pointing, brought together."""
    left = BASE_HAND.copy()
    for f in [MIDDLE, RING, PINKY]:
        curl_finger(left, f, 0.95)
    curl_finger(left, THUMB, 0.8)
    # rotate to point forward
    for i in range(len(left)):
        left[i] = np.array([left[i][1], -left[i][0], left[i][2]])

    right = mirror_hand(BASE_HAND.copy())
    for f in [MIDDLE, RING, PINKY]:
        curl_finger(right, f, 0.95)
    curl_finger(right, THUMB, 0.8)
    for i in range(len(right)):
        right[i] = np.array([-right[i][1], right[i][0], right[i][2]])

    w1 = np.array([0.42, 0.5, 0.0])
    w2 = np.array([0.58, 0.5, 0.0])
    return left, right, w1, w2

define_sign("[SAME]", sign_same)


def sign_no():
    """NO: Index + middle + thumb snapping together (open position)."""
    h = BASE_HAND.copy()
    for f in [RING, PINKY]:
        curl_finger(h, f, 0.95)
    spread_finger(h, INDEX, -0.02)
    spread_finger(h, MIDDLE, 0.02)
    # thumb partially extended toward index/middle
    h[3] = h[2] + np.array([-0.01, -0.03, -0.02])
    h[4] = h[3] + np.array([-0.01, -0.03, -0.02])
    return h, None, np.array([0.5, 0.5, 0.0]), None

define_sign("[NO]", sign_no)


def sign_yes():
    """YES: Fist nodding (S-hand shape)."""
    h = BASE_HAND.copy()
    make_fist(h)
    h[4][1] = h[6][1]
    h[4][2] = -0.03
    for i in range(len(h)):
        h[i][1] += 0.02
    return h, None, np.array([0.5, 0.5, 0.0]), None

define_sign("[YES]", sign_yes)


def sign_friend():
    """FRIEND: Two hooked index fingers (X-shape) linked together."""
    left = BASE_HAND.copy()
    for f in [MIDDLE, RING, PINKY]:
        curl_finger(left, f, 0.95)
    curl_finger(left, INDEX, 0.5)
    curl_finger(left, THUMB, 0.7)

    right = mirror_hand(BASE_HAND.copy())
    for f in [MIDDLE, RING, PINKY]:
        curl_finger(right, f, 0.95)
    curl_finger(right, INDEX, 0.5)
    curl_finger(right, THUMB, 0.7)

    w1 = np.array([0.44, 0.48, 0.0])
    w2 = np.array([0.56, 0.52, 0.0])
    return left, right, w1, w2

define_sign("[FRIEND]", sign_friend)


def sign_work():
    """WORK: Dominant S-fist tapping non-dominant S-fist (stacked)."""
    left = BASE_HAND.copy()
    make_fist(left)
    left[4][1] = left[6][1]
    left[4][2] = -0.03

    right = mirror_hand(BASE_HAND.copy())
    make_fist(right)
    right[4][1] = right[6][1]
    right[4][2] = -0.03

    w1 = np.array([0.47, 0.55, 0.0])
    w2 = np.array([0.50, 0.42, -0.01])
    return left, right, w1, w2

define_sign("[WORK]", sign_work)


def sign_finish():
    """FINISH/DONE: Both open 5-hands, fingers spread, palms out."""
    left = BASE_HAND.copy()
    spread_finger(left, INDEX, -0.03)
    spread_finger(left, MIDDLE, -0.01)
    spread_finger(left, RING, 0.01)
    spread_finger(left, PINKY, 0.03)

    right = mirror_hand(BASE_HAND.copy())
    spread_finger(right, INDEX, 0.03)
    spread_finger(right, MIDDLE, 0.01)
    spread_finger(right, RING, -0.01)
    spread_finger(right, PINKY, -0.03)

    w1 = np.array([0.35, 0.45, 0.0])
    w2 = np.array([0.65, 0.45, 0.0])
    return left, right, w1, w2

define_sign("[FINISH]", sign_finish)


def sign_go():
    """GO: Both index fingers pointing forward."""
    left = BASE_HAND.copy()
    for f in [MIDDLE, RING, PINKY]:
        curl_finger(left, f, 0.95)
    curl_finger(left, THUMB, 0.8)
    for i in range(len(left)):
        left[i] = np.array([left[i][2], left[i][1], -left[i][0]])

    right = mirror_hand(BASE_HAND.copy())
    for f in [MIDDLE, RING, PINKY]:
        curl_finger(right, f, 0.95)
    curl_finger(right, THUMB, 0.8)
    for i in range(len(right)):
        right[i] = np.array([-right[i][2], right[i][1], right[i][0]])

    w1 = np.array([0.43, 0.5, 0.0])
    w2 = np.array([0.57, 0.5, 0.0])
    return left, right, w1, w2

define_sign("[GO]", sign_go)


def sign_sit():
    """SIT: H/U hand (2 fingers) sitting on other H/U hand."""
    left = BASE_HAND.copy()
    for f in [RING, PINKY]:
        curl_finger(left, f, 0.95)
    curl_finger(left, THUMB, 0.8)
    for i in range(len(left)):
        left[i] = np.array([left[i][1], -left[i][0], left[i][2]])

    right = mirror_hand(BASE_HAND.copy())
    for f in [RING, PINKY]:
        curl_finger(right, f, 0.95)
    curl_finger(right, THUMB, 0.8)

    w1 = np.array([0.47, 0.55, 0.0])
    w2 = np.array([0.50, 0.45, -0.01])
    return left, right, w1, w2

define_sign("[SIT]", sign_sit)


def sign_big():
    """BIG: Both B/open hands spread far apart."""
    left = BASE_HAND.copy()
    curl_finger(left, THUMB, 0.3)

    right = mirror_hand(BASE_HAND.copy())
    curl_finger(right, THUMB, 0.3)

    w1 = np.array([0.25, 0.5, 0.0])
    w2 = np.array([0.75, 0.5, 0.0])
    return left, right, w1, w2

define_sign("[BIG]", sign_big)


def sign_small():
    """SMALL: Both B/open hands close together."""
    left = BASE_HAND.copy()
    curl_finger(left, THUMB, 0.3)

    right = mirror_hand(BASE_HAND.copy())
    curl_finger(right, THUMB, 0.3)

    w1 = np.array([0.46, 0.5, 0.0])
    w2 = np.array([0.54, 0.5, 0.0])
    return left, right, w1, w2

define_sign("[SMALL]", sign_small)


def sign_love():
    """LOVE: Two A-fists crossed over chest."""
    left = BASE_HAND.copy()
    make_fist(left)
    left[4] = left[1] + np.array([-0.03, -0.06, -0.03])

    right = mirror_hand(BASE_HAND.copy())
    make_fist(right)
    right[4] = right[1] + np.array([0.03, -0.06, -0.03])

    w1 = np.array([0.55, 0.55, 0.0])
    w2 = np.array([0.45, 0.50, -0.03])
    return left, right, w1, w2

define_sign("[LOVE]", sign_love)


def sign_eat():
    """EAT: Flat-O / bunched fingers toward mouth (single hand)."""
    h = BASE_HAND.copy()
    for f in FINGERS:
        curl_finger(h, f, 0.55)
    h[4][0] = h[8][0]
    h[4][1] = h[8][1]
    # tilt hand upward (toward face)
    for i in range(len(h)):
        h[i] = np.array([h[i][0], h[i][1] - 0.03, h[i][2] - 0.02])
    return h, None, np.array([0.5, 0.4, 0.0]), None

define_sign("[EAT]", sign_eat)


def sign_drink():
    """DRINK: C-hand shape tilted (like tipping a cup)."""
    h = BASE_HAND.copy()
    for f in FINGERS:
        curl_finger(h, f, 0.4)
    h[4][0] -= 0.02
    # tilt the hand
    angle = 0.4
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    for i in range(len(h)):
        y, z = h[i][1], h[i][2]
        h[i][1] = y * cos_a - z * sin_a
        h[i][2] = y * sin_a + z * cos_a
    return h, None, np.array([0.5, 0.4, 0.0]), None

define_sign("[DRINK]", sign_drink)


# ---------------------------------------------------------------------------
# Sample generation with noise and augmentation
# ---------------------------------------------------------------------------

def augment_hand(hand, rng):
    noise = rng.normal(0, NOISE_SCALE, hand.shape)
    hand = hand + noise

    angle = rng.normal(0, 0.15)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
    hand = hand @ rot.T

    scale = rng.normal(1.0, 0.05)
    hand *= scale
    return hand


def generate_sample(label, rng):
    h1, h2, w1, w2 = ASL_SIGNS[label]()

    h1 = augment_hand(h1, rng)

    # jitter wrist positions
    w1 = w1 + rng.normal(0, 0.03, 3)

    if h2 is not None:
        h2 = augment_hand(h2, rng)
        w2 = w2 + rng.normal(0, 0.03, 3)
        return build_feature_vector(h1, h2, w1, w2)
    else:
        return build_feature_vector(h1)


def make_header():
    cols = ["label"]
    for prefix in ("h1", "h2"):
        for i in range(NUM_LANDMARKS):
            for axis in ("x", "y", "z"):
                cols.append(f"{prefix}_{axis}{i}")
    cols += ["rel_x", "rel_y", "rel_z", "num_hands"]
    return cols


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    rng = np.random.default_rng(42)

    header = make_header()
    rows = [header]
    labels = sorted(ASL_SIGNS.keys())

    print(f"Generating {SAMPLES_PER_SIGN} samples for each of {len(labels)} signs...")

    for label in labels:
        for _ in range(SAMPLES_PER_SIGN):
            features = generate_sample(label, rng)
            rows.append([label] + features)
        kind = "word" if label.startswith("[") else "letter"
        print(f"  {label:18s} ({kind}): {SAMPLES_PER_SIGN} samples")

    with open(DATA_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    total = len(rows) - 1
    n_letters = sum(1 for l in labels if not l.startswith("["))
    n_words = sum(1 for l in labels if l.startswith("["))
    print(f"\nSaved {total} samples to {DATA_FILE}")
    print(f"  {n_letters} letter signs + {n_words} word signs = {len(labels)} classes")
    print("Now run: python train_model.py")


if __name__ == "__main__":
    main()
