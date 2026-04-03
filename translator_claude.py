# translator_claude.py
# Translates farmer advisory text into regional Indian languages.
# Uses the same free Google Translate endpoint as the Leaf Translate HTML app:
#   https://translate.googleapis.com/translate_a/single?client=gtx&...
# No API key required.

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional


SUPPORTED_LANGUAGES: dict[str, str] = {
    "Hindi": "hi",
    "Kannada": "kn",
    "Tamil": "ta",
    "Telugu": "te",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Bengali": "bn",
    "Punjabi": "pa",
    "Odia": "or",
    "Assamese": "as",
    "Urdu": "ur",
    "English": "en",
}


def translate_text(
    text: str,
    target_language: str,
    source_language: str = "en",
) -> dict:
    target_code = _resolve_language_code(target_language)
    if target_code is None:
        return _error_response(
            text,
            source_language,
            target_language,
            f"Unsupported language: '{target_language}'. "
            f"Supported: {', '.join(SUPPORTED_LANGUAGES.keys())}",
        )

    if target_code == source_language:
        return {
            "translated_text": text,
            "source_language": source_language,
            "target_language": target_code,
            "target_language_name": _code_to_name(target_code),
            "detected_language": None,
            "backend_used": "passthrough",
            "success": True,
            "error": None,
        }

    try:
        translated, detected = _translate_gtx(text, source_language, target_code)
        return {
            "translated_text": translated,
            "source_language": source_language,
            "target_language": target_code,
            "target_language_name": _code_to_name(target_code),
            "detected_language": detected,
            "backend_used": "gtx",
            "success": True,
            "error": None,
        }
    except TranslationError as exc:
        return _error_response(text, source_language, target_code, str(exc))


def get_supported_languages() -> list[str]:
    return sorted(SUPPORTED_LANGUAGES.keys())


def _translate_gtx(text: str, source: str, target: str) -> tuple[str, Optional[str]]:
    params = urllib.parse.urlencode(
        {
            "client": "gtx",
            "sl": source,
            "tl": target,
            "dt": "t",
            "q": text,
        }
    )
    url = f"https://translate.googleapis.com/translate_a/single?{params}"

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise TranslationError(f"GTX HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise TranslationError(f"GTX network error: {exc}") from exc

    try:
        result = "".join(chunk[0] for chunk in data[0] if chunk[0])
    except (TypeError, IndexError, KeyError) as exc:
        raise TranslationError(f"GTX unexpected response format: {exc}") from exc

    if not result:
        raise TranslationError("GTX returned an empty translation.")

    detected = data[2] if source == "auto" and len(data) > 2 else None
    return result, detected


class TranslationError(Exception):
    """Raised when the translation request fails."""


def _resolve_language_code(language: str) -> Optional[str]:
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
    for name, lang_code in SUPPORTED_LANGUAGES.items():
        if lang_code == code:
            return name
    return code


def _error_response(text, source, target, error_msg):
    return {
        "translated_text": text,
        "source_language": source,
        "target_language": target if isinstance(target, str) else "unknown",
        "target_language_name": _code_to_name(target) if isinstance(target, str) else target,
        "detected_language": None,
        "backend_used": None,
        "success": False,
        "error": error_msg,
    }
