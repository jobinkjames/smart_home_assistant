"""
voice/wake_word.py
Runs OpenWakeWord in a background thread.
Fires a callback AND sets a threading.Event when wake word is detected.

Usage in main.py:
    from voice.wake_word import WakeWordDetector

    def on_wake():
        print("Wake word heard — start listening")

    detector = WakeWordDetector(on_detect=on_wake)
    detector.start()
    ...
    detector.stop()
"""

import os
import threading
import numpy as np
import pyaudio
from openwakeword.model import Model
import scipy

# ── Config (mirrors your working test script) ─────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WAKE_WORD_MODEL = os.path.join(
    _BASE_DIR, "data", "wake_models", "Hey_Nova_20260328_194345.tflite"
)
THRESHOLD = 0.5
SAMPLE_RATE = 44100
CHUNK_SIZE = 3528  # 80ms at 44100Hz (1280/16000 * 44100)
MIC_DEVICE_INDEX = 6  # pulse — allows sharing
# ─────────────────────────────────────────────────────────────────────────────


class WakeWordDetector:
    """
    Listens for the wake word in a background daemon thread.
    On detection:
      - Sets self.detected (threading.Event)
      - Calls on_detect callback (if provided)
    """

    def __init__(self, on_detect=None):
        """
        Args:
            on_detect: optional callable — called with no args on detection.
        """
        self._on_detect = on_detect
        self.detected = threading.Event()  # callers can wait() on this
        self._stop_flag = threading.Event()
        self._thread = None
        self._model = None
        self._model_name = None

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        """Load model and start background listening thread."""
        print("[wake_word] Loading model...")
        self._model = Model(
            wakeword_models=[WAKE_WORD_MODEL], inference_framework="tflite"
        )
        self._model_name = list(self._model.models.keys())[0]
        print(f"[wake_word] Model loaded: {self._model_name}")

        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        print("[wake_word] Listening in background...")

    def stop(self):
        """Signal the background thread to stop and wait for it."""
        self._stop_flag.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        print("[wake_word] Stopped.")

    def reset(self):
        """
        Clear the detected event and reset model buffer.
        Call this after handling a detection so the detector is ready again.
        """
        self.detected.clear()
        if self._model:
            self._model.reset()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _listen_loop(self):
        """Background thread — reads mic chunks and runs inference."""
        pa = pyaudio.PyAudio()
        stream = pa.open(
            rate=SAMPLE_RATE,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
            input_device_index=MIC_DEVICE_INDEX,
        )

        try:
            while not self._stop_flag.is_set():
                raw = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                audio = np.frombuffer(raw, dtype=np.int16)
                audio_16k = scipy.signal.resample(audio, 1280).astype(np.int16)
                predictions = self._model.predict(audio_16k)

                if self._model_name not in predictions:
                    continue

                score = float(predictions[self._model_name])

                if score >= THRESHOLD:
                    print(f"\n[wake_word] ✅ Detected! score={score:.3f}")
                    self.detected.set()  # signal the Event
                    if self._on_detect:
                        self._on_detect()  # fire callback
                    # pause inference until reset() is called externally
                    self._stop_flag.wait(timeout=2)
                    self._model.reset()

        except Exception as e:
            print(f"[wake_word] Error in listen loop: {e}")

        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()
