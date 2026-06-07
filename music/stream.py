"""
music/stream.py
Uses yt-dlp to extract a direct audio stream URL from a YouTube video.
No file download — streams directly into mpv.
"""

import os
import yt_dlp

# Path to cookies file — only needed if YouTube starts blocking the Pi
COOKIES_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "cookies.txt"
)


def get_stream_url(youtube_url: str) -> str | None:
    """
    Extract direct audio stream URL from a YouTube video URL.
    Returns the stream URL string, or None on failure.
    """
    ydl_opts = {
        "format":      "bestaudio/best",
        "quiet":       True,
        "no_warnings": True,
        "extractor_args": {
            "youtube": {"skip": ["dash", "hls"]}  # faster extraction on Pi
        },
    }

    # attach cookies only if file exists
    if os.path.exists(COOKIES_PATH):
        ydl_opts["cookiefile"] = COOKIES_PATH

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            return info["url"]
    except Exception as e:
        print(f"[stream] yt-dlp error: {e}")
        return None
