# translator.py
# Translates farmer advisory text into regional Indian languages.
# Supports multiple translation backends with automatic fallback:
#   1. Google Cloud Translation API  (preferred — set GOOGLE_TRANSLATE_API_KEY)
#   2. MyMemory free API             (no key needed, rate-limited)
#   3. LibreTranslate self-hosted    (set LIBRETRANSLATE_URL if available)

import os
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
    backend: Optional[str] = None,
) -> dict:
    """
    Translate advisory text into the specified regional language.

    Args:
        text            (str): English text to translate.
        target_language (str): Target language name (e.g. "Kannada") or
                               BCP-47 code (e.g. "kn").
        source_language (str): Source language code (default "en").
        backend         (str): Force a specific backend — "google", "mymemory",
                               or "libretranslate". If None, auto-selects.

    Returns:
        dict with keys:
            - translated_text   (str)   — translated output
            - source_language   (str)   — source language code
            - target_language   (str)   — target language code
            - target_language_name (str)
            - backend_used      (str)   — which API was used
            - success           (bool)
            - error             (str | None)

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
            "backend_used":         "passthrough",
            "success":              True,
            "error":                None,
        }

    # Choose backend order
    if backend:
        backends = [backend]
    else:
        backends = _auto_backend_order()

    last_error = "No backend available."
    for b in backends:
        try:
            translated = _call_backend(b, text, source_language, target_code)
            return {
                "translated_text":      translated,
                "source_language":      source_language,
                "target_language":      target_code,
                "target_language_name": _code_to_name(target_code),
                "backend_used":         b,
                "success":              True,
                "error":                None,
            }
        except TranslationError as e:
            last_error = str(e)
            continue  # try next backend

    return _error_response(text, source_language, target_code, last_error)


def get_supported_languages() -> list[str]:
    """Return a sorted list of supported language names."""
    return sorted(SUPPORTED_LANGUAGES.keys())


# ============================================================
# === Backend dispatcher ===
# ============================================================

def _call_backend(backend: str, text: str, source: str, target: str) -> str:
    """Dispatch to the appropriate translation backend."""
    if backend == "google":
        return _translate_google(text, source, target)
    elif backend == "mymemory":
        return _translate_mymemory(text, source, target)
    elif backend == "libretranslate":
        return _translate_libretranslate(text, source, target)
    else:
        raise TranslationError(f"Unknown backend: '{backend}'")


def _auto_backend_order() -> list[str]:
    """Select backend priority based on available API keys/config."""
    order = []
    if os.getenv("GOOGLE_TRANSLATE_API_KEY"):
        order.append("google")
    libre_url = os.getenv("LIBRETRANSLATE_URL")
    if libre_url:
        order.append("libretranslate")
    order.append("mymemory")  # Always available as last resort
    return order


# ============================================================
# === Backend 1: Google Cloud Translation API ===
# ============================================================

def _translate_google(text: str, source: str, target: str) -> str:
    """
    Translate using the Google Cloud Translation REST API (v2).
    Requires environment variable GOOGLE_TRANSLATE_API_KEY.
    """
    api_key = os.getenv("GOOGLE_TRANSLATE_API_KEY")
    if not api_key:
        raise TranslationError("GOOGLE_TRANSLATE_API_KEY not set.")

    url = "https://translation.googleapis.com/language/translate/v2"
    payload = json.dumps({
        "q":      text,
        "source": source,
        "target": target,
        "format": "text",
        "key":    api_key,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["data"]["translations"][0]["translatedText"]
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise TranslationError(f"Google API HTTP {e.code}: {body}") from e
    except (KeyError, IndexError) as e:
        raise TranslationError(f"Google API unexpected response: {e}") from e


# ============================================================
# === Backend 2: MyMemory Free API ===
# ============================================================

def _translate_mymemory(text: str, source: str, target: str) -> str:
    """
    Translate using the MyMemory free API (no key required).
    Rate limit: ~5000 characters/day for anonymous requests.
    Set MYMEMORY_EMAIL env var for a higher limit (10000 chars/day).
    """
    langpair = f"{source}|{target}"
    email    = os.getenv("MYMEMORY_EMAIL", "")
    params   = {"q": text, "langpair": langpair}
    if email:
        params["de"] = email

    url = "https://api.mymemory.translated.net/get?" + urllib.parse.urlencode(params)

    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise TranslationError(f"MyMemory network error: {e}") from e

    response_status = data.get("responseStatus", 0)
    if response_status != 200:
        details = data.get("responseDetails", "Unknown error")
        raise TranslationError(f"MyMemory API error {response_status}: {details}")

    translated = data.get("responseData", {}).get("translatedText", "")
    if not translated:
        raise TranslationError("MyMemory returned empty translation.")

    return translated


# ============================================================
# === Backend 3: LibreTranslate (self-hosted) ===
# ============================================================

def _translate_libretranslate(text: str, source: str, target: str) -> str:
    """
    Translate using a LibreTranslate instance.
    Set LIBRETRANSLATE_URL env var (e.g. http://localhost:5000).
    Optionally set LIBRETRANSLATE_API_KEY if the instance requires it.
    """
    base_url = os.getenv("LIBRETRANSLATE_URL", "").rstrip("/")
    if not base_url:
        raise TranslationError("LIBRETRANSLATE_URL not set.")

    payload = {
        "q":      text,
        "source": source,
        "target": target,
        "format": "text",
    }
    api_key = os.getenv("LIBRETRANSLATE_API_KEY", "")
    if api_key:
        payload["api_key"] = api_key

    encoded = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/translate",
        data=encoded,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise TranslationError(f"LibreTranslate network error: {e}") from e

    translated = data.get("translatedText", "")
    if not translated:
        raise TranslationError("LibreTranslate returned empty translation.")
    return translated


# ============================================================
# === Helpers ===
# ============================================================

class TranslationError(Exception):
    """Raised when a translation backend fails."""


def _resolve_language_code(language: str) -> Optional[str]:
    """
    Accept either a language name ("Kannada") or a BCP-47 code ("kn").
    Returns the BCP-47 code, or None if not supported.
    """
    # Direct name lookup
    if language in SUPPORTED_LANGUAGES:
        return SUPPORTED_LANGUAGES[language]
    # Case-insensitive name lookup
    lower = language.lower()
    for name, code in SUPPORTED_LANGUAGES.items():
        if name.lower() == lower:
            return code
    # Already a code?
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
