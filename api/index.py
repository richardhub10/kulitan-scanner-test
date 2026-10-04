from __future__ import annotations

import os
from pathlib import Path

from src.kulitan_scanner.web_app import create_app

checkpoint = Path(os.getenv("CHECKPOINT_PATH", "artifacts/best_model.pt"))
min_confidence = float(os.getenv("MIN_CONFIDENCE", "0.70"))
top_k = int(os.getenv("TOP_K", "3"))
unknown_label = os.getenv("UNKNOWN_LABEL", "unknown")
allow_no_symbol = os.getenv("ALLOW_NO_SYMBOL", "false").lower() in {"1", "true", "yes", "on"}

app = create_app(
    checkpoint=checkpoint,
    min_confidence=min_confidence,
    top_k=top_k,
    unknown_label=unknown_label,
    allow_no_symbol=allow_no_symbol,
)
