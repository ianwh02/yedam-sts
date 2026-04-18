"""KoELECTRA-based phrase translatability classifier.

Uses an ONNX-exported KoELECTRA-small model to predict whether a Korean text
fragment contains enough meaning to produce a natural English translation.

Used as a confidence gate on KoreanEndingDetector phrase flushes:
rule-based detects candidate → model confirms → only flush if both agree.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Default model location (relative to stt-server root)
_DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "phrase-classifier"


class PhraseClassifier:
    """ONNX-based Korean phrase translatability classifier.

    Loads once at startup, runs inference on CPU via onnxruntime.
    Thread-safe (ONNX Runtime sessions are thread-safe for inference).
    """

    def __init__(
        self,
        model_dir: str | Path | None = None,
        threshold: float = 0.50,
        max_length: int = 128,
    ):
        model_dir = Path(model_dir) if model_dir else _DEFAULT_MODEL_DIR
        onnx_path = model_dir / "model.onnx"

        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        # Load tokenizer — use the fast tokenizers library directly
        # to avoid pulling in torch via transformers
        try:
            from tokenizers import Tokenizer

            tokenizer_path = model_dir / "tokenizer.json"
            if not tokenizer_path.exists():
                raise FileNotFoundError(f"tokenizer.json not found in {model_dir}")
            self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
            self._tokenizer.enable_truncation(max_length=max_length)
            self._tokenizer.enable_padding(length=max_length)
            self._use_fast_tokenizer = True
        except ImportError:
            # Fall back to transformers AutoTokenizer
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            self._max_length = max_length
            self._use_fast_tokenizer = False

        # Load ONNX model
        import onnxruntime as ort

        sess_opts = ort.SessionOptions()
        sess_opts.inter_op_num_threads = 1
        sess_opts.intra_op_num_threads = 2
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )

        self._threshold = threshold
        logger.info(
            f"PhraseClassifier loaded from {model_dir} "
            f"(threshold={threshold}, max_len={max_length})"
        )

    def predict(self, text: str) -> tuple[bool, float]:
        """Predict whether Korean text is translatable.

        Returns:
            (is_translatable, confidence) where confidence is the
            softmax probability for the predicted class.
        """
        if self._use_fast_tokenizer:
            encoded = self._tokenizer.encode(text)
            input_ids = np.array([encoded.ids], dtype=np.int64)
            attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
        else:
            encoded = self._tokenizer(
                text,
                truncation=True,
                max_length=self._max_length,
                padding="max_length",
                return_tensors="np",
            )
            input_ids = encoded["input_ids"].astype(np.int64)
            attention_mask = encoded["attention_mask"].astype(np.int64)

        logits = self._session.run(
            ["logits"],
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )[0]  # shape: (1, 2)

        # Softmax
        exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp / exp.sum(axis=-1, keepdims=True)

        translatable_prob = float(probs[0, 1])
        is_translatable = translatable_prob >= self._threshold

        return is_translatable, translatable_prob


def load_phrase_classifier() -> PhraseClassifier | None:
    """Load the phrase classifier if model files exist and deps are available.

    Returns None (with a warning) if model is missing or deps unavailable,
    so the system gracefully degrades to rule-based only.
    """
    model_dir = os.environ.get("PHRASE_CLASSIFIER_MODEL_DIR", "")
    threshold = float(os.environ.get("PHRASE_CLASSIFIER_THRESHOLD", "0.50"))

    model_path = Path(model_dir) if model_dir else _DEFAULT_MODEL_DIR

    if not (model_path / "model.onnx").exists():
        logger.warning(
            f"Phrase classifier model not found at {model_path}, "
            f"falling back to rule-based only"
        )
        return None

    try:
        return PhraseClassifier(model_dir=model_path, threshold=threshold)
    except ImportError as e:
        logger.warning(f"Phrase classifier deps missing ({e}), rule-based only")
        return None
    except Exception as e:
        logger.error(f"Failed to load phrase classifier: {e}")
        return None
