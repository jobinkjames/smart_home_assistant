"""
music/youtube_search.py
Search YouTube Data API v3 and return the best music match.
"""

import os
import requests

YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"


def search_song(song_name: str) -> dict | None:
    """
    Search YouTube for a song.

    Returns:
    {
        "title": str,
        "channel": str,
        "video_id": str,
        "url": str,
        "thumbnail": str
    }

    Returns None if nothing found.
    """

    api_key = os.getenv("YOUTUBE_API_KEY")

    if not api_key:
        raise RuntimeError(
            "YOUTUBE_API_KEY environment variable not found"
        )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/137.0.0.0 Safari/537.36"
        )
    }

    params = {
        "part": "snippet",
        "q": song_name,
        "maxResults": 1,
        "type": "video",
        "videoCategoryId": "10",
        "key": api_key,
    }

    try:
        response = requests.get(
            YOUTUBE_SEARCH_URL,
            headers=headers,
            params=params,
            timeout=10,
        )

        response.raise_for_status()
        data = response.json()

    except requests.Timeout:
        return None

    except requests.RequestException:
        return None

    items = data.get("items", [])

    if not items:
        return None

    item = items[0]

    video_id = item["id"]["videoId"]
    snippet = item["snippet"]

    return {
        "title": snippet["title"],
        "channel": snippet["channelTitle"],
        "video_id": video_id,
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "thumbnail": snippet["thumbnails"]["high"]["url"],
    }