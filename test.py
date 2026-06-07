from voice.recorder import Recorder


def main():
    recorder = Recorder()

    print("\nSpeak after the prompt. Stop speaking to end recording.\n")
    input("Press Enter when ready...")
    print()

    wav_bytes = recorder.record()

    if wav_bytes:
        with open("test_recording.wav", "wb") as f:
            f.write(wav_bytes)
        print("✅ Saved to test_recording.wav")
    else:
        print("❌ Nothing recorded.")


if __name__ == "__main__":
    main()
