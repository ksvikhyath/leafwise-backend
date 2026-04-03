# advisory.py
# Generates farmer-friendly advisory content for identified medicinal plants.
# Loads context from plant_knowledge.json and formats structured output.

import json
import os
import re

# === Path to knowledge base ===
KNOWLEDGE_BASE_PATH = os.path.join(os.path.dirname(__file__), "plant_knowledge.json")


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _stringify_field(value) -> str:
    if value is None:
        return "N/A"

    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        if not items:
            return "N/A"
        return "\n".join(f"- {item}" for item in items)

    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, indent=2)

    text = str(value).strip()
    return text or "N/A"


def _aliases_for_entry(key: str, data: dict) -> list:
    aliases = [
        key,
        data.get("canonical_name"),
        data.get("plant_name"),
        data.get("common_name"),
        data.get("scientific_name"),
    ]

    raw_aliases = data.get("aliases", [])
    if isinstance(raw_aliases, list):
        aliases.extend(raw_aliases)
    elif raw_aliases:
        aliases.append(raw_aliases)

    cleaned = []
    seen = set()
    for alias in aliases:
        if not alias:
            continue
        alias_text = str(alias).strip()
        alias_key = _normalize_text(alias_text)
        if not alias_key or alias_key in seen:
            continue
        seen.add(alias_key)
        cleaned.append(alias_text)
    return cleaned

# === Load knowledge base once at module level ===
def _load_knowledge_base(path: str = KNOWLEDGE_BASE_PATH) -> dict:
    """
    Load plant knowledge base from JSON file.

    Returns:
        dict: Mapping of plant name → knowledge dict.
    Raises:
        FileNotFoundError: If the knowledge base JSON is missing.
        json.JSONDecodeError: If the file is malformed.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"plant_knowledge.json not found at: {path}\n"
            "Please ensure the file is present in the project directory."
        )
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        return raw

    if isinstance(raw, list):
        normalized = {}
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            key = (
                entry.get("canonical_name")
                or entry.get("plant_name")
                or entry.get("common_name")
            )
            if key:
                normalized[str(key)] = entry
        return normalized

    raise ValueError("Knowledge base JSON must contain a dict or list of plant entries.")


# Module-level cache
try:
    _KNOWLEDGE_BASE: dict = _load_knowledge_base()
except Exception as _e:
    _KNOWLEDGE_BASE = {}
    print(f"⚠️  Advisory module: Could not load knowledge base — {_e}")


# ============================================================
# === Public API ===
# ============================================================

def get_plant_advisory(plant_name: str) -> dict:
    """
    Return structured advisory information for a given plant name.

    Tries an exact match first, then falls back to a case-insensitive
    substring search so minor naming differences are handled gracefully.

    Args:
        plant_name (str): The identified plant name (as returned by the model).

    Returns:
        dict with keys:
            - plant_name       (str)
            - common_name      (str)
            - scientific_name  (str)
            - medicinal_uses   (str)
            - cultivation      (str)
            - income_potential (str)
            - care_instructions(str)
            - found            (bool) — False if plant not in knowledge base

    Example:
        >>> advisory = get_plant_advisory("Tulsi")
        >>> print(advisory["medicinal_uses"])
    """
    if not _KNOWLEDGE_BASE:
        return _not_found_response(plant_name)

    # 1. Exact match
    if plant_name in _KNOWLEDGE_BASE:
        return _build_response(plant_name, _KNOWLEDGE_BASE[plant_name])

    normalized_name = _normalize_text(plant_name)

    # 2. Alias / case-insensitive match
    for key, data in _KNOWLEDGE_BASE.items():
        if _normalize_text(key) == normalized_name:
            return _build_response(key, data)

        aliases = _aliases_for_entry(key, data)
        if normalized_name in {_normalize_text(alias) for alias in aliases}:
            return _build_response(key, data)

    # 3. Partial match (common name, scientific name, or aliases)
    for key, data in _KNOWLEDGE_BASE.items():
        haystacks = [
            key,
            data.get("canonical_name"),
            data.get("plant_name"),
            data.get("common_name"),
            data.get("scientific_name"),
        ]
        raw_aliases = data.get("aliases", [])
        if isinstance(raw_aliases, list):
            haystacks.extend(raw_aliases)

        if any(normalized_name and normalized_name in _normalize_text(value) for value in haystacks if value):
            return _build_response(key, data)

    return _not_found_response(plant_name)


def format_advisory_text(advisory: dict) -> str:
    """
    Format an advisory dict into a human-readable text block
    suitable for display in Streamlit or printing to console.

    Args:
        advisory (dict): Output of get_plant_advisory().

    Returns:
        str: Formatted advisory string.
    """
    if not advisory.get("found", True):
        return (
            f"⚠️  No advisory information is available for '{advisory['plant_name']}' yet.\n"
            "Please consult a local agricultural extension officer or Ayurvedic practitioner."
        )

    lines = [
        f"🌿  {advisory['common_name']}  ({advisory['scientific_name']})",
        "=" * 60,
        "",
        "💊  MEDICINAL USES",
        f"    {advisory['medicinal_uses']}",
        "",
        "🌱  CULTIVATION GUIDE",
        f"    {advisory['cultivation']}",
        "",
        "💰  INCOME POTENTIAL",
        f"    {advisory['income_potential']}",
        "",
        "🛠️  CARE INSTRUCTIONS",
        f"    {advisory['care_instructions']}",
        "",
        "=" * 60,
    ]
    return "\n".join(lines)


def get_advisory_as_html(plant_name: str) -> str:
    """
    Convenience wrapper that returns a ready-to-embed HTML block
    for use in Streamlit's st.markdown(..., unsafe_allow_html=True).

    Args:
        plant_name (str): Identified plant name.

    Returns:
        str: HTML string.
    """
    advisory = get_plant_advisory(plant_name)

    if not advisory.get("found", True):
        return (
            f"<p>⚠️ No advisory information found for <strong>{plant_name}</strong>.</p>"
        )

    html = f"""
    <div style="background:#f1f8e9;border-left:5px solid #388e3c;border-radius:8px;padding:16px;margin:10px 0;">
        <h3 style="color:#1b5e20;margin-top:0;">🌿 {advisory['common_name']}</h3>
        <p style="color:#555;font-style:italic;margin-top:-10px;">{advisory['scientific_name']}</p>
        <hr style="border-color:#a5d6a7;"/>

        <h4 style="color:#2e7d32;">💊 Medicinal Uses</h4>
        <p>{advisory['medicinal_uses']}</p>

        <h4 style="color:#2e7d32;">🌱 Cultivation Guide</h4>
        <p>{advisory['cultivation']}</p>

        <h4 style="color:#2e7d32;">💰 Income Potential</h4>
        <p>{advisory['income_potential']}</p>

        <h4 style="color:#2e7d32;">🛠️ Care Instructions</h4>
        <p>{advisory['care_instructions']}</p>
    </div>
    """
    return html


def list_supported_plants() -> list:
    """
    Return a sorted list of all plant names available in the knowledge base.

    Returns:
        list[str]: Sorted plant name keys.
    """
    return sorted(_KNOWLEDGE_BASE.keys())


# ============================================================
# === Internal helpers ===
# ============================================================

def _build_response(key: str, data: dict) -> dict:
    aliases = _aliases_for_entry(key, data)
    return {
        "plant_name":        key,
        "canonical_name":    data.get("canonical_name", key),
        "aliases":           aliases,
        "common_name":       data.get("common_name",       key),
        "scientific_name":   data.get("scientific_name",   "N/A"),
        "medicinal_uses":    _stringify_field(data.get("medicinal_uses",    "N/A")),
        "cultivation":       _stringify_field(data.get("cultivation",       "N/A")),
        "income_potential":  _stringify_field(data.get("income_potential",  "N/A")),
        "care_instructions": _stringify_field(data.get("care_instructions", "N/A")),
        "source_files":      data.get("source_files", []),
        "found": True,
    }


def _not_found_response(plant_name: str) -> dict:
    return {
        "plant_name":        plant_name,
        "canonical_name":    plant_name,
        "aliases":           [plant_name],
        "common_name":       "Unknown",
        "scientific_name":   "Unknown",
        "medicinal_uses":    "N/A",
        "cultivation":       "N/A",
        "income_potential":  "N/A",
        "care_instructions": "N/A",
        "source_files":      [],
        "found": False,
    }


# ============================================================
# === Quick self-test ===
# ============================================================
if __name__ == "__main__":
    print(f"📚 Loaded {len(_KNOWLEDGE_BASE)} plants from knowledge base.\n")

    test_plants = ["Tulsi", "Neem", "tulsi", "UnknownPlant", "Mint"]
    for name in test_plants:
        adv = get_plant_advisory(name)
        print(f"--- Query: '{name}' ---")
        print(format_advisory_text(adv))
        print()
