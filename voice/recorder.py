"""
voice/recorder.py
Records voice after wake word detection.
Uses energy-based silence detection — no torch, no heavy dependencies.
Stops on silence or max timeout — whichever comes first.
Returns WAV bytes ready to send to Gemini.

Install:
    pip install pyaudio numpy
"""

import io
import time
import wave
import numpy as np
import pyaudio

# ── Config ────────────────────────────────────────────────
SAMPLE_RATE = 44100
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK_SIZE = 1024
MAX_DURATION = 15  # seconds — hard cutoff
SILENCE_TIMEOUT = 1.5  # seconds of silence = end of speech
SILENCE_THRESHOLD = 1000  # RMS energy below this = silence
# raise if background is noisy
# lower if mic is too sensitive
MIC_INDEX = 4  # None = system default
# ─────────────────────────────────────────────────────────


class Recorder:
    """
    Records voice after wake word fires.
    Stops when silence detected or max duration reached.
    No external ML model needed — pure energy-based VAD.

    Usage:
        recorder = Recorder()
        wav_bytes = recorder.record()
    """

    def record(self) -> bytes | None:
        """
        Record until silence or max timeout.
        Returns WAV bytes or None if no speech detected.
        """
        pa = pyaudio.PyAudio()
        stream = pa.open(
            rate=SAMPLE_RATE,
            channels=CHANNELS,
            format=FORMAT,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
            input_device_index=MIC_INDEX,
        )

        print("[recorder] 🎙 Speak now...")

        frames = []
        silent_duration = 0.0
        speech_started = False
        chunk_duration = CHUNK_SIZE / SAMPLE_RATE  # seconds per chunk
        start_time = time.time()

        try:
            while True:
                # hard timeout check
                if time.time() - start_time > MAX_DURATION:
                    print(f"\n[recorder] Max duration ({MAX_DURATION}s) reached.")
                    break

                raw = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                frames.append(raw)

                # energy-based voice detection
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                rms = np.sqrt(np.mean(audio**2))
                is_speech = rms > SILENCE_THRESHOLD

                if is_speech:
                    speech_started = True
                    silent_duration = 0.0
                    print("▌", end="", flush=True)
                else:
                    if speech_started:
                        silent_duration += chunk_duration
                        if silent_duration >= SILENCE_TIMEOUT:
                            print(f"\n[recorder] Silence detected — done.")
                            break

        except Exception as e:
            print(f"\n[recorder] Error: {e}")
            return None

        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()

        if not speech_started:
            print("[recorder] No speech detected.")
            return None

        wav_bytes = self._to_wav(frames)
        print(f"[recorder] {len(wav_bytes) / 1024:.1f} KB WAV captured.")
        return wav_bytes

    # ── Internal ──────────────────────────────────────────

    def _to_wav(self, frames: list[bytes]) -> bytes:
        """Convert raw PCM frames to WAV bytes."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b"".join(frames))
        buf.seek(0)
        return buf.read()
