"""Exploratory trace analysis package."""
import json
from pathlib import Path

def pilot_data_present() -> bool:
    manifest = Path("repeated_draws/manifest.json")
    if not manifest.exists():
        return False
    seeds = json.loads(manifest.read_text())["completed_seeds"]
    return all(Path(f"repeated_draws/seed_{s}.json").exists() for s in seeds)
