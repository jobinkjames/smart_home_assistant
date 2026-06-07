"""
vision/face_detector.py
Loads known faces from data/known_faces/ and matches a given frame.
Designed for on-demand use — load, detect, unload.
Works on both PC (cv2) and Pi Zero 2W (picamera2).
"""

import os
import re
import numpy as np
import face_recognition


# ── Config ────────────────────────────────────────────────
_BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWN_FACES_DIR = os.path.join(_BASE_DIR, "data", "known_faces")
MATCH_THRESHOLD = 0.55
DETECTION_SCALE = 0.5     # run detection on half-res, encoding on full-res
# ─────────────────────────────────────────────────────────


class FaceDetector:
    """
    Loads known face embeddings from data/known_faces/ on init.
    Call detect(frame_rgb) with a single RGB frame.
    Returns name + confidence.
    """

    def __init__(self):
        self._known_names     = []
        self._known_encodings = []

    def load(self):
        """Load all known face embeddings from known_faces folder."""
        self._known_names     = []
        self._known_encodings = []

        if not os.path.exists(KNOWN_FACES_DIR):
            print(f"[face_detector] Folder not found: {KNOWN_FACES_DIR}")
            return

        print("[face_detector] Loading known faces...")

        for filename in sorted(os.listdir(KNOWN_FACES_DIR)):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            path  = os.path.join(KNOWN_FACES_DIR, filename)
            image = face_recognition.load_image_file(path)
            encs  = face_recognition.face_encodings(image)

            if not encs:
                print(f"[face_detector] ⚠ No face in {filename} — skipping")
                continue

            base = os.path.splitext(filename)[0]
            name = re.sub(r"_\d+$", "", base).replace("_", " ").title()

            self._known_names.append(name)
            self._known_encodings.append(encs[0])
            print(f"[face_detector] ✅ {name} ({filename})")

        print(f"[face_detector] {len(self._known_names)} face(s) loaded: "
              f"{sorted(set(self._known_names))}\n")

    def detect(self, frame_rgb: np.ndarray) -> dict:
        """
        Run face detection + recognition on a single RGB frame.
        Returns dict: {person, confidence, face_box}
        face_box is (x1, y1, x2, y2) or None if no face found.
        """
        result = {
            "person":     "Unknown",
            "confidence": 0.0,
            "face_box":   None,
        }

        if not self._known_encodings:
            print("[face_detector] No known faces loaded — call load() first")
            return result

        # detect on smaller frame for speed
        h, w  = frame_rgb.shape[:2]
        small = face_recognition.resize_image(
            frame_rgb,
            int(w * DETECTION_SCALE),
            int(h * DETECTION_SCALE)
        ) if hasattr(face_recognition, "resize_image") else \
            __import__("cv2").resize(frame_rgb, (0, 0),
                                     fx=DETECTION_SCALE, fy=DETECTION_SCALE)

        locations = face_recognition.face_locations(small, model="hog")

        if not locations:
            return result

        # scale locations back to full resolution
        scale     = 1.0 / DETECTION_SCALE
        full_locs = [
            (int(t * scale), int(r * scale),
             int(b * scale), int(l * scale))
            for t, r, b, l in locations
        ]

        encodings = face_recognition.face_encodings(frame_rgb, full_locs)
        if not encodings:
            return result

        # match against known faces
        distances  = face_recognition.face_distance(self._known_encodings, encodings[0])
        best_idx   = int(np.argmin(distances))
        best_dist  = distances[best_idx]
        confidence = max(0.0, min(1.0, 1.0 - best_dist))

        top, right, bottom, left = full_locs[0]
        result["face_box"] = (left, top, right, bottom)

        if best_dist < MATCH_THRESHOLD:
            result["person"]     = self._known_names[best_idx]
            result["confidence"] = confidence

        return result

    def unload(self):
        """Release embeddings from memory."""
        self._known_names     = []
        self._known_encodings = []
        print("[face_detector] Unloaded.")
