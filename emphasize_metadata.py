# emphasize_metadata.py

def emphasize_text(metadata: dict) -> str:
    """
    Emphasize important metadata features by repeating their values.
    
    Parameters:
    - metadata (dict): Dictionary containing metadata attributes for a plant.

    Returns:
    - str: A combined string with repeated emphasis on selected features.
    """
    aroma = str(metadata.get("Aroma", "")).strip()
    tissue = str(metadata.get("Internal tissue", "")).strip()

    # Repeat important features
    emphasized = (aroma + " ") * 2 + (tissue + " ") * 3

    # Append the rest of the metadata, excluding already emphasized keys
    rest = " ".join(str(v).strip() for k, v in metadata.items() if k not in ["Aroma", "Internal tissue"])
    
    return (emphasized + rest).strip()
