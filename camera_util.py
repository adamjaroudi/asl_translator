"""
Open webcam with fallbacks so it works on more Windows setups.
On Windows: tries DirectShow first, then MSMF (Media Foundation) if DSHOW fails or gives black frames.

To force a specific camera: set env CAMERA_INDEX=1 or pass preferred_index.
"""

import os
import sys
import cv2

if sys.platform == "win32":
    # Try DSHOW first; MSMF (1400) often works when DSHOW doesn't or gives black feed
    BACKENDS = [cv2.CAP_DSHOW, 1400]  # 1400 = CAP_MSMF
else:
    BACKENDS = [cv2.CAP_ANY]


def _try_open(index, backend, num_reads=5):
    """Open one index with one backend; return (cap, True) if we get a non-black frame."""
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        return None, False
    for _ in range(num_reads):
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0 and frame.mean() >= 5:
            return cap, True
    cap.release()
    return None, False


def open_camera(preferred_index=None):
    """
    Try to open the default or preferred camera.
    Uses env CAMERA_INDEX if set. On Windows tries DSHOW then MSMF per index.
    Returns (cv2.VideoCapture, success). If failed, cap is None.
    """
    env_idx = os.environ.get("CAMERA_INDEX")
    if env_idx is not None:
        try:
            preferred_index = int(env_idx)
        except ValueError:
            preferred_index = None
    if preferred_index is None:
        preferred_index = 0

    indices = [preferred_index, 0, 1, 2]
    for index in indices:
        if index < 0:
            continue
        for backend in BACKENDS:
            cap, ok = _try_open(index, backend)
            if ok:
                return cap, True
    return None, False
