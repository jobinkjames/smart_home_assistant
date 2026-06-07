"""
assistant/gemini_client.py
Sends recorded audio + context to Gemini 2.0 Flash.
Returns text response.
Maintains conversation history within a session.

Install:
    pip install google-generativeai
"""

import os
import base64
from pyexpat.errors import messages
from google import genai

# ── Config ────────────────────────────────────────────────
MODEL = "gemini-2.0-flash"
MAX_HISTORY_TURNS = 6  # keep last 6 turns (3 user + 3 assistant)
AUDIO_MIME = "audio/wav"
# ─────────────────────────────────────────────────────────


class GeminiClient:
    """
    Handles all Gemini API communication.

    Usage:
        client = GeminiClient()
        client.start_session(system_prompt)
        response = client.send(wav_bytes)
        client.end_session()
    """

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not set in .env")
        self._client = genai.Client(api_key=api_key)
        self._history = []  # conversation turns
        self._system = ""  # current session system prompt

    # ── Session control ───────────────────────────────────

    def start_session(self, system_prompt: str):
        """
        Start a new conversation session.
        Clears history and sets system prompt.
        Call this each time a new person is detected.
        """
        self._system = system_prompt
        self._history = []
        print(f"[gemini] Session started.")

    def end_session(self) -> str | None:
        """
        End session — ask Gemini to summarize the conversation.
        Returns 2-sentence summary string for saving to profile.
        """
        if not self._history:
            return None

        summary_prompt = (
            "Summarize this conversation in max 2 short sentences. "
            "Focus on what the person said, their mood, and any topics mentioned. "
            "No fluff."
        )

        try:
            # build full conversation for summary
            summary_messages = self._history + [
                {"role": "user", "parts": [summary_prompt]}
            ]
            response = self._client.models.generate_content(
                model=MODEL, contents=messages
            )
            summary = response.text.strip()
            print(f"[gemini] Session summary: {summary}")
            self._history = []
            return summary

        except Exception as e:
            print(f"[gemini] Summary error: {e}")
            return None

    # ── Send audio ────────────────────────────────────────

    def send(self, wav_bytes: bytes) -> str | None:
        """
        Send WAV audio to Gemini with conversation history.
        Returns text response or None on error.

        Args:
            wav_bytes: raw WAV bytes from recorder.record()

        Returns:
            str: Gemini's text response
        """
        if not wav_bytes:
            print("[gemini] No audio to send.")
            return None

        # encode audio as base64
        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")

        # build user message with audio
        user_message = {
            "role": "user",
            "parts": [
                {
                    "inline_data": {
                        "mime_type": AUDIO_MIME,
                        "data": audio_b64,
                    }
                }
            ],
        }

        # build full message list:
        # system prompt + trimmed history + new user message
        messages = self._build_messages(user_message)

        try:
            print("[gemini] Sending audio...")
            response = self._model.generate_content(messages)
            reply = response.text.strip()
            print(f"[gemini] Response: {reply}")

            # save this turn to history
            self._history.append(user_message)
            self._history.append({"role": "model", "parts": [reply]})

            # trim history to max turns
            self._trim_history()

            return reply

        except Exception as e:
            print(f"[gemini] Error: {e}")
            return None

    # ── Internal ──────────────────────────────────────────

    def _build_messages(self, new_user_message: dict) -> list:
        """
        Build full message list for Gemini.
        System prompt injected as first user turn (Gemini 2.0 style).
        """
        messages = []

        # inject system prompt as first turn if set
        if self._system:
            messages.append({"role": "user", "parts": [self._system]})
            messages.append({"role": "model", "parts": ["Understood. I'm ready."]})

        # add conversation history
        messages.extend(self._history)

        # add new user message
        messages.append(new_user_message)

        return messages

    def _trim_history(self):
        """Keep only last MAX_HISTORY_TURNS turns to control token usage."""
        max_messages = MAX_HISTORY_TURNS * 2  # each turn = user + model
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]
