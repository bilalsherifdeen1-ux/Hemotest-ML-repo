"""Utility helpers."""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
from loguru import logger

def save_json(data: dict, path: str | Path) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    logger.info(f"Saved -> {p}")

def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
