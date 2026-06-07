"""
main.py — updated with recorder + gemini wired in
"""

import os
import time
import cv2
import threading
from dotenv import load_dotenv

load_dotenv()

from music.queue_manager import QueueManager
from music import player
from voice.wake_word import WakeWordDetector
from voice.recorder import Recorder
from vision.vision_session import VisionSession
from assistant.gemini_client import GeminiClient
from assistant.prompt_builder import build_system_prompt, save_summary

# ── Shared state ──────────────────────────────────────────
detector = None
vision = VisionSession()
recorder = Recorder()
client = GeminiClient()


# ── Wake word callback ────────────────────────────────────


def on_wake_word():
    # run everything in a new thread so wake_word callback returns immediately
    threading.Thread(target=_handle_session, daemon=True).start()


def _handle_session():
    print("\n[main] Wake word detected!")

    was_playing = player.is_playing()
    if was_playing:
        player.pause()

    time.sleep(0.5)  # give mic time to release after wake word

    # face + pose
    time.sleep(1.0)
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    vision_result = {"person": "Unknown", "activity": "unknown"}
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        vision_result = vision.run(frame_rgb)

    system_prompt = build_system_prompt(vision_result)
    client.start_session(system_prompt)

    print("[main] Conversation started.")
    while True:
        wav_bytes = recorder.record()
        if not wav_bytes:
            break

        response = client.send(wav_bytes)
        if not response:
            break

        print(f"\n🤖 Nova: {response}\n")

        lower = response.lower()
        if any(w in lower for w in ["bye", "goodbye", "see you", "നന്ദി", "ശരി"]):
            break

    summary = client.end_session()
    if summary and vision_result["person"] != "Unknown":
        save_summary(vision_result["person"], summary)

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

    print(
        "\nCommands: play <song> | pause | resume | stop | next | prev | queue | quit"
    )

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
