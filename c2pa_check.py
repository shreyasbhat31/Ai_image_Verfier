import json
from pathlib import Path
import c2pa


def inspect_c2pa(file_path: str | Path) -> dict:
    """
    Inspects a saved image file for a C2PA manifest.
    Returns a simple status dictionary that app.py can display.
    """
    path = Path(file_path)

    try:
        # Reader reads and validates C2PA manifest data from a file
        with c2pa.Reader(str(path)) as reader:
            manifest_json = reader.json()
            active = reader.get_active_manifest()

            if not manifest_json or active is None:
                return {
                    "status": "missing",
                    "message": "No C2PA manifest found. Proceed to Tier 2 forensic analysis.",
                    "details": None,
                }

            store = json.loads(manifest_json)
            validation_status = store.get("validation_status", [])

            # If C2PA reports validation errors, surface them clearly
            if validation_status:
                return {
                    "status": "invalid_or_tampered",
                    "message": "C2PA manifest found, but validation reported issues.",
                    "details": {
                        "claim_generator": active.get("claim_generator"),
                        "title": active.get("title"),
                        "validation_status": validation_status,
                    },
                }

            return {
                "status": "verified",
                "message": "Valid C2PA manifest found.",
                "details": {
                    "claim_generator": active.get("claim_generator"),
                    "title": active.get("title"),
                    "assertions": [a.get("label") for a in active.get("assertions", [])],
                },
            }

    except Exception as exc:
        # Most normal internet images/screenshots have no JUMBF/C2PA manifest
        err_text = str(exc)
        if "ManifestNotFound" in err_text or "no JUMBF" in err_text.lower():
            return {
                "status": "missing",
                "message": "No C2PA manifest found. Proceed to Tier 2 forensic analysis.",
                "details": None,
            }

        return {
            "status": "error",
            "message": f"Could not parse C2PA data: {err_text}",
            "details": None,
        }