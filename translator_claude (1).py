# translator.py
# Translates farmer advisory text into regional Indian languages.
# Uses the same free Google Translate endpoint as the Leaf Translate HTML app:
#   https://translate.googleapis.com/translate_a/single?client=gtx&...
# No API key required.

import json
import urllib.request
import urllib.parse
import urllib.error
from typing import Optional


# ============================================================
# === Supported Languages ===
# ============================================================

# Maps human-friendly language names to their BCP-47 / ISO-639-1 codes.
SUPPORTED_LANGUAGES: dict[str, str] = {
    "Hindi":      "hi",
    "Kannada":    "kn",
    "Tamil":      "ta",
    "Telugu":     "te",
    "Malayalam":  "ml",
    "Marathi":    "mr",
    "Gujarati":   "gu",
    "Bengali":    "bn",
    "Punjabi":    "pa",
    "Odia":       "or",
    "Assamese":   "as",
    "Urdu":       "ur",
    "English":    "en",
}


# ============================================================
# === Main public function ===
# ============================================================

def translate_text(
    text: str,
    target_language: str,
    source_language: str = "en",
) -> dict:
    """
    Translate advisory text into the specified regional language using
    the free Google Translate endpoint (no API key required).

    Args:
        text            (str): Text to translate.
        target_language (str): Target language name (e.g. "Kannada") or
                               BCP-47 code (e.g. "kn"). Use "auto" to
                               let Google detect the source language.
        source_language (str): Source language code (default "en").
                               Pass "auto" for automatic detection.

    Returns:
        dict with keys:
            - translated_text      (str)        — translated output
            - source_language      (str)         — source language code used
            - target_language      (str)         — target language code
            - target_language_name (str)
            - detected_language    (str | None)  — detected language if auto was used
            - backend_used         (str)         — always "gtx"
            - success              (bool)
            - error                (str | None)

    Example:
        >>> result = translate_text("Neem is good for skin.", "Kannada")
        >>> print(result["translated_text"])
    """
    # Resolve language name → code
    target_code = _resolve_language_code(target_language)
    if target_code is None:
        return _error_response(
            text, source_language, target_language,
            f"Unsupported language: '{target_language}'. "
            f"Supported: {', '.join(SUPPORTED_LANGUAGES.keys())}"
        )

    # If source == target, return as-is
    if target_code == source_language:
        return {
            "translated_text":      text,
            "source_language":      source_language,
            "target_language":      target_code,
            "target_language_name": _code_to_name(target_code),
            "detected_language":    None,
            "backend_used":         "passthrough",
            "success":              True,
            "error":                None,
        }

    try:
        translated, detected = _translate_gtx(text, source_language, target_code)
        return {
            "translated_text":      translated,
            "source_language":      source_language,
            "target_language":      target_code,
            "target_language_name": _code_to_name(target_code),
            "detected_language":    detected,
            "backend_used":         "gtx",
            "success":              True,
            "error":                None,
        }
    except TranslationError as e:
        return _error_response(text, source_language, target_code, str(e))


def get_supported_languages() -> list[str]:
    """Return a sorted list of supported language names."""
    return sorted(SUPPORTED_LANGUAGES.keys())


# ============================================================
# === Google Translate free endpoint (gtx) ===
# ============================================================

def _translate_gtx(text: str, source: str, target: str) -> tuple[str, Optional[str]]:
    """
    Translate using the same free endpoint the Leaf Translate HTML app uses:
        https://translate.googleapis.com/translate_a/single?client=gtx&sl=...&tl=...&dt=t&q=...

    Returns a (translated_text, detected_language_code) tuple.
    detected_language_code is None when source is not "auto".
    """
    params = urllib.parse.urlencode({
        "client": "gtx",
        "sl":     source,
        "tl":     target,
        "dt":     "t",
        "q":      text,
    })
    url = f"https://translate.googleapis.com/translate_a/single?{params}"

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"},  # mimic browser, same as gtx
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise TranslationError(f"GTX HTTP {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise TranslationError(f"GTX network error: {e}") from e

    # Response shape mirrors what the HTML app parses:
    # data[0] → list of [translated_chunk, original_chunk, ...]
    # data[2] → detected language code (only present when sl="auto")
    try:
        result = "".join(chunk[0] for chunk in data[0] if chunk[0])
    except (TypeError, IndexError, KeyError) as e:
        raise TranslationError(f"GTX unexpected response format: {e}") from e

    if not result:
        raise TranslationError("GTX returned an empty translation.")

    detected = data[2] if source == "auto" and len(data) > 2 else None
    return result, detected


# ============================================================
# === Helpers ===
# ============================================================

class TranslationError(Exception):
    """Raised when the translation request fails."""


def _resolve_language_code(language: str) -> Optional[str]:
    """
    Accept either a language name ("Kannada") or a BCP-47 code ("kn").
    Returns the BCP-47 code, or None if not supported.
    """
    if language in SUPPORTED_LANGUAGES:
        return SUPPORTED_LANGUAGES[language]
    lower = language.lower()
    for name, code in SUPPORTED_LANGUAGES.items():
        if name.lower() == lower:
            return code
    if language in SUPPORTED_LANGUAGES.values():
        return language
    return None


def _code_to_name(code: str) -> str:
    for name, c in SUPPORTED_LANGUAGES.items():
        if c == code:
            return name
    return code


def _error_response(text, source, target, error_msg):
    return {
        "translated_text":      text,   # Return original on failure
        "source_language":      source,
        "target_language":      target if isinstance(target, str) else "unknown",
        "target_language_name": _code_to_name(target) if isinstance(target, str) else target,
        "detected_language":    None,
        "backend_used":         None,
        "success":              False,
        "error":                error_msg,
    }


# ============================================================
# === Quick self-test ===
# ============================================================
if __name__ == "__main__":
    sample = (
        "Neem is one of the most versatile medicinal plants. "
        "It is used for skin diseases, dental health, and as a natural insecticide."
    )

    print("Supported languages:", get_supported_languages())
    print()

    for lang in ["Hindi", "Kannada", "Tamil"]:
        print(f"Translating to {lang}...")
        result = translate_text(sample, lang)
        if result["success"]:
            print(f"  [{result['backend_used']}] {result['translated_text']}")
        else:
            print(f"  ❌ Failed: {result['error']}")
        print()
