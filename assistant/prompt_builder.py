"""
assistant/prompt_builder.py
Builds a compressed, token-efficient Gemini system prompt
from vision result + person profile + time context.

Target: ~40-60 tokens of context per call (not 200+).
"""

import json
import os
from datetime import datetime

# ── Config ────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILES_DIR = os.path.join(_BASE_DIR, "data", "profiles")
MAX_SUMMARY_LEN = 120
# ─────────────────────────────────────────────────────────


# ── Profile loader ────────────────────────────────────────


def _profile_filename(name: str) -> str:
    return name.lower().replace(" ", "_") + ".json"


def load_profile(name: str) -> dict | None:
    path = os.path.join(PROFILES_DIR, _profile_filename(name))
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_summary(name: str, summary: str):
    path = os.path.join(PROFILES_DIR, _profile_filename(name))
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as f:
        profile = json.load(f)

    profile["last_summary"] = summary[:MAX_SUMMARY_LEN].strip()

    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=4, ensure_ascii=False)

    print(f"[prompt_builder] Summary saved for {name}")


# ── Time context ───────────────────────────────────────────


def _time_context():
    now = datetime.now()
    hour = now.hour
    day = now.strftime("%A").lower()

    if 5 <= hour < 12:
        g = "morning"
    elif 12 <= hour < 17:
        g = "afternoon"
    elif 17 <= hour < 21:
        g = "evening"
    else:
        g = "night"

    return g, day


def _day_type(schedule: dict, day: str) -> str:
    return schedule.get(day, "regular")


# ── System prompt (ultra compressed) ───────────────────────

_NOVA_SYSTEM = (
    "Nova AI: reply in user language. "
    "Ask 1 short warm question. Avoid generic phrases."
)

_UNKNOWN_SYSTEM = "Nova AI: unknown user. Greet warmly and ask name."


# ── Prompt builder ─────────────────────────────────────────


def build_system_prompt(vision_result: dict) -> str:
    name = vision_result.get("person", "Unknown")
    activity = vision_result.get("activity", "unknown")

    gtime, gday = _time_context()

    # ── Unknown user ─────────────────────────────────────
    if name == "Unknown":
        return _UNKNOWN_SYSTEM

    profile = load_profile(name)

    if not profile:
        return f"{_NOVA_SYSTEM}\n" f"P:{name}|A:{activity}|T:{gtime}:{gday}|lang:en"

    # ── Extract compressed fields ────────────────────────
    role = profile.get("role", "")
    lang = profile.get("language", "en")
    style = profile.get("talk_style", "")
    interests = ",".join(profile.get("interests", []))[:50]

    schedule = profile.get("schedule", {})
    day_type = _day_type(schedule, gday)

    summary = profile.get("last_summary", "")[:80]
    greet = profile.get("greet_style", "")

    # ── COMPRESSED CONTEXT LINE ─────────────────────────
    context = (
        f"P:{name}|R:{role}|L:{lang}|S:{style}|"
        f"I:{interests}|"
        f"T:{gtime}:{gday}:{day_type}|"
        f"A:{activity}"
    )

    if summary:
        context += f"|LS:{summary}"

    # ── FINAL PROMPT ────────────────────────────────────
    prompt = f"{_NOVA_SYSTEM}\n{context}|G:{greet}"

    return prompt


# ── Debug helper ──────────────────────────────────────────


def preview(vision_result: dict):
    prompt = build_system_prompt(vision_result)
    est_tokens = len(prompt) // 4

    print("\n" + "-" * 50)
    print(f"[prompt_builder] ~{est_tokens} tokens")
    print("-" * 50)
    print(prompt)
    print("-" * 50 + "\n")

    return prompt


# ── Test ──────────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        {"person": "Jobin", "activity": "standing"},
        {"person": "Bindhu Amma", "activity": "sitting"},
        {"person": "Pappa", "activity": "standing"},
        {"person": "Jesnamma", "activity": "waving"},
        {"person": "Pappamma", "activity": "sitting"},
        {"person": "Unknown", "activity": "standing"},
    ]

    for t in test_cases:
        preview(t)
