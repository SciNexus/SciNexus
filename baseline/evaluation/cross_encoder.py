from __future__ import annotations
import threading
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from SciNexus import paths as P

class CrossEncoderScorer:
    def __init__(self, model_dir: str | Path | None = None, device: str | None = None):
        model_path = str(model_dir or P.env_path(P.ENV_CROSSENCODER_MODEL)).strip()
        if not model_path:
            raise ValueError(
                f"Missing required path: set {P.ENV_CROSSENCODER_MODEL} "
                "or pass --crossencoder"
            )
        model_dir = Path(model_path)
        if not (model_dir / "config.json").exists():
            raise FileNotFoundError(f"Cross-encoder config not found under {model_dir}")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.model.to(self.device)
        self.model.eval()
        self._lock = threading.Lock()

    @torch.no_grad()
    def score(self, text_a: str, text_b: str, *, max_length: int = 512) -> float:
        pairs = [[text_a.strip() or " ", text_b.strip() or " "]]
        with self._lock:
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=max_length,
            ).to(self.device)
            logits = self.model(**inputs, return_dict=True).logits.view(-1).float()
        prob = F.sigmoid(logits).item()
        return float(prob * 100.0)

    def effectiveness_objective(self, keywords: str, idea: str) -> float:
        return self.score(keywords, idea)

    def novelty_objective(self, solutions: str, idea: str) -> float:
        rel = self.score(solutions, idea)
        return (1.0 - rel / 100.0) * 100.0
