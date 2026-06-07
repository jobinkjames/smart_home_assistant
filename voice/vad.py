"""
test_recorder.py — Standalone recorder test
Run this before integrating into the project.

Usage:
    python test_recorder.py

Install:
    pip install pyaudio torch numpy
"""

from voice.recorder import Recorder


def main():
    recorder = Recorder()
    recorder.load()

    print("\nSpeak after the prompt. Stop speaking to end recording.\n")
    input("Press Enter when ready...")
    print()

    wav_bytes = recorder.record()

    if wav_bytes:
        # save to file so you can listen back
        with open("test_recording.wav", "wb") as f:
            f.write(wav_bytes)
        print(f"\n✅ Saved to test_recording.wav — play it back to verify quality.")
    else:
        print("\n❌ Nothing recorded.")

    recorder.unload()


if __name__ == "__main__":
    main()
