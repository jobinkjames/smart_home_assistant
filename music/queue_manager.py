"""
music/queue_manager.py
Manages a song queue — add, next, previous, clear.
Works together with youtube_search, stream, and player.
"""

from music.youtube_search import search_song
from music.stream import get_stream_url
from music import player


class QueueManager:
    def __init__(self):
        self._queue: list[dict] = []   # list of search result dicts
        self._index: int = -1          # current position in queue

    def add(self, song_name: str) -> dict | None:
        """Search for a song and add it to the queue."""
        result = search_song(song_name)
        if result:
            self._queue.append(result)
            print(f"[queue] Added: {result['title']}")
        else:
            print(f"[queue] Song not found: {song_name}")
        return result

    def play_current(self):
        """Play the song at the current queue index."""
        if not self._queue:
            print("[queue] Queue is empty")
            return
        if self._index < 0 or self._index >= len(self._queue):
            print("[queue] Invalid queue position")
            return

        song = self._queue[self._index]
        stream_url = get_stream_url(song["url"])
        if stream_url:
            player.play(stream_url, title=song["title"])
        else:
            print(f"[queue] Could not stream: {song['title']}")

    def play_song(self, song_name: str):
        """Search, add to queue, and immediately play a song."""
        result = self.add(song_name)
        if result:
            self._index = len(self._queue) - 1
            self.play_current()

    def next(self):
        """Skip to next song in queue."""
        if self._index < len(self._queue) - 1:
            self._index += 1
            self.play_current()
        else:
            print("[queue] No next song in queue")
            player.stop()

    def previous(self):
        """Go back to previous song in queue."""
        if self._index > 0:
            self._index -= 1
            self.play_current()
        else:
            print("[queue] Already at first song")

    def clear(self):
        """Clear queue and stop playback."""
        self._queue.clear()
        self._index = -1
        player.stop()
        print("[queue] Queue cleared")

    def current(self) -> dict | None:
        """Return currently playing song info."""
        if 0 <= self._index < len(self._queue):
            return self._queue[self._index]
        return None

    def list_queue(self):
        """Print all songs in queue."""
        if not self._queue:
            print("[queue] Empty")
            return
        for i, song in enumerate(self._queue):
            marker = "▶" if i == self._index else " "
            print(f"  {marker} {i+1}. {song['title']} — {song['channel']}")
