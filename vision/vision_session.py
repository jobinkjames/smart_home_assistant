"""
vision/vision_session.py
Orchestrates face detection + pose detection in a single session.
This is the only vision file that main.py ever calls.

Usage:
    from vision.vision_session import VisionSession

    session = VisionSession()
    result  = session.run(frame_rgb)
    # result = {
    #     "person":     "Jobin",
    #     "confidence": 0.87,
    #     "activity":   "sitting",
    #     "timestamp":  "08:02:31"
    # }

On Pi Zero 2W:
    - Models are loaded on demand, run once, then unloaded
    - Call session.run(frame_rgb) with a single frame from camera.py
    - Total vision time: ~2-4 seconds on Pi Zero 2W

On PC:
    - Same code, faster execution (~0.5s)
"""

import time
import numpy as np


from vision.face_detector import FaceDetector
from vision.pose_detector import PoseDetector


class VisionSession:
    """
    Single entry point for all vision processing.
    Loads models → runs detection → unloads models.
    """

    def __init__(self):
        self._face = FaceDetector()
        self._pose = PoseDetector()

    def run(self, frame_rgb: np.ndarray) -> dict:
        """
        Run face + pose detection on a single RGB frame.

        Args:
            frame_rgb: numpy array (H, W, 3) in RGB format

        Returns:
            dict with person, confidence, activity, timestamp
        """
        result = {
            "person": "Unknown",
            "confidence": 0.0,
            "activity": "unknown",
            "timestamp": time.strftime("%H:%M:%S"),
        }

        try:
            # ── Load both models ──────────────────────────────────────────
            self._face.load()
            self._pose.load()

            # ── Face detection ────────────────────────────────────────────
            face_result = self._face.detect(frame_rgb)
            result["person"] = face_result["person"]
            result["confidence"] = face_result["confidence"]

            # ── Pose detection ────────────────────────────────────────────
            result["activity"] = self._pose.detect(frame_rgb)

            # ── Log result ────────────────────────────────────────────────
            print(
                f"[vision] {result['timestamp']} | "
                f"{result['person']} ({result['confidence']:.0%}) | "
                f"{result['activity']}"
            )

        except Exception as e:
            print(f"[vision] Error during session: {e}")

        finally:
            # ── Always unload to free RAM ─────────────────────────────────
            self._face.unload()
            self._pose.unload()

        return result
