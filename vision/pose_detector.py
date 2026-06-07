"""
vision/pose_detector.py
Single-frame pose activity classification using MediaPipe Pose.
Designed for on-demand use — load, detect, unload.
Optimised for upper-body visibility (desk/room webcam + Pi camera).
"""

import numpy as np
import mediapipe as mp


class PoseDetector:
    """
    Runs MediaPipe Pose on a single RGB frame.
    Call detect(frame_rgb) to get activity string.
    """

    def __init__(self):
        self._pose = None

    def load(self):
        """Load MediaPipe Pose model."""
        print("[pose_detector] Loading MediaPipe Pose...")
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=True,       # single frame mode — more accurate
            model_complexity=0,           # lightest model — critical for Pi Zero
            min_detection_confidence=0.5,
        )
        print("[pose_detector] Loaded.")

    def detect(self, frame_rgb: np.ndarray) -> str:
        """
        Classify activity from a single RGB frame.
        Returns activity string: standing | sitting | waving |
                                 arms raised | leaning | unknown
        """
        if self._pose is None:
            print("[pose_detector] Not loaded — call load() first")
            return "unknown"

        # run on smaller frame for speed
        import cv2
        small  = cv2.resize(frame_rgb, (320, 240))
        result = self._pose.process(small)

        if not result.pose_landmarks:
            return "unknown"

        return self._classify(result.pose_landmarks)

    def unload(self):
        """Release MediaPipe model from memory."""
        if self._pose:
            self._pose.close()
            self._pose = None
        print("[pose_detector] Unloaded.")

    # ── Activity classifier ───────────────────────────────────────────────────

    def _classify(self, landmarks) -> str:
        """
        Classify posture from landmark positions.
        Works with upper-body only (no ankle/knee required).
        """
        lm = landmarks.landmark

        LEFT_SHOULDER  = lm[11]
        RIGHT_SHOULDER = lm[12]
        LEFT_HIP       = lm[23]
        RIGHT_HIP      = lm[24]
        LEFT_WRIST     = lm[15]
        RIGHT_WRIST    = lm[16]

        shoulder_y    = (LEFT_SHOULDER.y  + RIGHT_SHOULDER.y) / 2
        hip_y         = (LEFT_HIP.y       + RIGHT_HIP.y)      / 2
        wrist_l_y     = LEFT_WRIST.y
        wrist_r_y     = RIGHT_WRIST.y
        shoulder_tilt = abs(LEFT_SHOULDER.y - RIGHT_SHOULDER.y)
        body_height   = abs(shoulder_y - hip_y)

        # waving — one wrist clearly above shoulder
        if wrist_l_y < shoulder_y - 0.1 or wrist_r_y < shoulder_y - 0.1:
            return "waving"

        # arms raised — both wrists above shoulder
        if wrist_l_y < shoulder_y - 0.05 and wrist_r_y < shoulder_y - 0.05:
            return "arms raised"

        # leaning — shoulders tilted
        if shoulder_tilt > 0.04:
            return "leaning"

        # sitting — upper body compressed vertically
        if body_height < 0.22:
            return "sitting"

        # standing — shoulders and hips well separated
        if body_height >= 0.22:
            return "standing"

        return "unknown"
