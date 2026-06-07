"""
music/player.py
Controls mpv subprocess for audio-only playback.
Supports play, pause, resume, stop, and playback state check.
"""

import signal
import subprocess

_mpv_process = None


def play(stream_url: str, title: str = "", volume: int = 75):
    """
    Start audio playback via mpv.
    Stops any currently playing audio first.
    """
    global _mpv_process
    stop()  # always stop previous before starting new

    print(f"[player] Playing: {title}")
    _mpv_process = subprocess.Popen(
        [
            "mpv",
            "--no-video",       # audio only — critical for Pi Zero 2W
            "--really-quiet",   # suppress mpv terminal output
            f"--volume={volume}",
            stream_url,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def pause():
    """Pause playback without killing the process (SIGSTOP)."""
    global _mpv_process
    if _mpv_process and _mpv_process.poll() is None:
        _mpv_process.send_signal(signal.SIGSTOP)
        print("[player] Paused")


def resume():
    """Resume a paused playback (SIGCONT)."""
    global _mpv_process
    if _mpv_process and _mpv_process.poll() is None:
        _mpv_process.send_signal(signal.SIGCONT)
        print("[player] Resumed")


def stop():
    """Terminate mpv process completely."""
    global _mpv_process
    if _mpv_process and _mpv_process.poll() is None:
        _mpv_process.terminate()
        _mpv_process.wait()
        print("[player] Stopped")
    _mpv_process = None


def is_playing() -> bool:
    """Returns True if mpv is currently running."""
    return _mpv_process is not None and _mpv_process.poll() is None
