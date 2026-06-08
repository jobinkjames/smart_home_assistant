"""
assistant/gemini_client.py
Sends recorded audio + context to Gemini 2.0 Flash.
Uses the new google-genai SDK.
Returns text response and maintains conversation history.

Install:
    pip install google-genai
"""

import os
import base64
from google import genai
from google.genai import types

MODEL = "gemini-2.0-flash"
MAX_HISTORY_TURNS = 6
AUDIO_MIME = "audio/wav"


class GeminiClient:

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not set in .env")
        self._client = genai.Client(api_key=api_key)
        self._history = []
        self._system = ""

    # ── Session control ───────────────────────────────────

    def start_session(self, system_prompt: str):
        self._system = system_prompt
        self._history = []
        print("[gemini] Session started.")

    def end_session(self) -> str | None:
        if not self._history:
            return None
        summary_prompt = (
            "Summarize this conversation in max 2 short sentences. "
            "Focus on what the person said and any topics mentioned. No fluff."
        )
        try:
            messages = self._history + [
                types.Content(role="user", parts=[types.Part(text=summary_prompt)])
            ]
            response = self._client.models.generate_content(
                model=MODEL, contents=messages
            )
            summary = response.text.strip()
            print(f"[gemini] Summary: {summary}")
            self._history = []
            return summary
        except Exception as e:
            print(f"[gemini] Summary error: {e}")
            return None

    # ── Send audio ────────────────────────────────────────

    def send(self, wav_bytes: bytes) -> str | None:
        if not wav_bytes:
            print("[gemini] No audio to send.")
            return None

        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")

        # new user message with audio
        user_message = types.Content(
            role="user",
            parts=[
                types.Part(inline_data=types.Blob(mime_type=AUDIO_MIME, data=audio_b64))
            ],
        )

        messages = self._build_messages(user_message)

        try:
            print("[gemini] Sending audio...")
            response = self._client.models.generate_content(
                model=MODEL,
                contents=messages,
                config=types.GenerateContentConfig(
                    system_instruction=self._system,
                    max_output_tokens=300,
                ),
            )
            reply = response.text.strip()
            print(f"[gemini] Response: {reply}")

            # save turn to history
            self._history.append(user_message)
            self._history.append(
                types.Content(role="model", parts=[types.Part(text=reply)])
            )
            self._trim_history()
            return reply

        except Exception as e:
            print(f"[gemini] Error: {e}")
            return None

    # ── Internal ──────────────────────────────────────────

    def _build_messages(self, new_user_message) -> list:
        messages = list(self._history)
        messages.append(new_user_message)
        return messages

    def _trim_history(self):
        max_messages = MAX_HISTORY_TURNS * 2
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]
