"""
main.py — updated with vision session integrated
"""

import os
import cv2
from dotenv import load_dotenv

load_dotenv()

from music.queue_manager import QueueManager
from music import player
from voice.wake_word import WakeWordDetector
from vision.vision_session import VisionSession


# ── Shared state ──────────────────────────────────────────
detector = None
vision   = VisionSession()


# ── Wake word callback ────────────────────────────────────

def on_wake_word():
    print("\n[main] Wake word detected!")

    was_playing = player.is_playing()
    if was_playing:
        player.pause()

    # ── Vision: capture frame → detect person + activity ──
    # On Pi: replace cv2.VideoCapture with picamera2 in hardware/camera.py
    cap   = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    vision_result = {"person": "Unknown", "activity": "unknown"}
    if ret:
        frame_rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        vision_result = vision.run(frame_rgb)

    print(f"[main] Person: {vision_result['person']} | "
          f"Activity: {vision_result['activity']}")

    # ── TODO: pass vision_result to prompt_builder ────────
    # ── TODO: recorder.record() → audio bytes ────────────
    # ── TODO: gemini_client.send(audio, vision_result) ───
    # ── TODO: speaker.speak(response) ────────────────────

    if was_playing:
        player.resume()

    detector.reset()


# ── Main ──────────────────────────────────────────────────

def main():
    global detector

    print("[main] Smart Home Assistant starting...")

    detector = WakeWordDetector(on_detect=on_wake_word)
    detector.start()

    queue = QueueManager()

    print("\nCommands: play <song> | pause | resume | stop | next | prev | queue | quit")

    while True:
        try:
            cmd = input("\n> ").strip().lower()
        except KeyboardInterrupt:
            player.stop()
            detector.stop()
            print("\n[main] Shutting down.")
            break

        if cmd.startswith("play "):
            queue.play_song(cmd[5:].strip())
        elif cmd == "pause":
            player.pause()
        elif cmd == "resume":
            player.resume()
        elif cmd == "stop":
            player.stop()
        elif cmd == "next":
            queue.next()
        elif cmd == "prev":
            queue.previous()
        elif cmd == "queue":
            queue.list_queue()
        elif cmd in ("quit", "q", "exit"):
            player.stop()
            detector.stop()
            print("[main] Bye.")
            break
        else:
            print("Unknown command.")


if __name__ == "__main__":
    main()