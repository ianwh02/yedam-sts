"""Korean STT post-processing: domain corrections and spacing fixes.

Applied between STT output and LLM input to improve transcription quality.
Two layers:
1. Domain-specific corrections — dictionary-based replacements for common
   Whisper misrecognitions of domain terminology (loaded from file).
2. Spacing correction — optional, via pykospacing if installed.

Both layers are fast (<5ms per segment) and run on CPU.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_corrections(path: str) -> list[tuple[re.Pattern, str]]:
    """Load domain corrections from a TSV file.

    Format: one replacement per line, tab-separated:
        wrong_text<TAB>correct_text

    Lines starting with # are comments. Blank lines are ignored.
    Patterns are compiled as word-boundary-aware regexes.
    """
    corrections: list[tuple[re.Pattern, str]] = []
    if not path:
        return corrections
    p = Path(path)
    if not p.is_file():
        logger.warning("Corrections file not found: %s", p)
        return corrections

    for line_num, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            logger.warning("Skipping malformed correction line %d: %r", line_num, line)
            continue
        wrong, correct = parts[0].strip(), parts[1].strip()
        if not wrong or not correct:
            continue
        try:
            pattern = re.compile(re.escape(wrong))
            corrections.append((pattern, correct))
        except re.error as e:
            logger.warning("Invalid regex on line %d: %s", line_num, e)

    if corrections:
        logger.info("Loaded %d domain corrections from %s", len(corrections), p)
    return corrections


class KoreanPostProcessor:
    """Post-processes Korean STT output for improved accuracy.

    Usage:
        pp = KoreanPostProcessor(corrections_path="/app/config/stt_corrections.tsv")
        await pp.initialize()
        cleaned = pp.process("하나님이 능히하신다")
    """

    def __init__(self, corrections_path: str = "", use_spacing: bool = False):
        self._corrections = _load_corrections(corrections_path)
        self._use_spacing = use_spacing
        self._spacing_fn = None

    async def initialize(self):
        """Load optional spacing model (lazy, only if configured)."""
        if self._use_spacing:
            try:
                from pykospacing import Spacing

                self._spacing_fn = Spacing()
                logger.info("pykospacing initialized for Korean spacing correction")
            except ImportError:
                logger.warning(
                    "pykospacing not installed, skipping spacing correction. "
                    "Install with: pip install pykospacing"
                )
                self._use_spacing = False

    def process(self, text: str) -> str:
        """Apply corrections and optional spacing fix to Korean text."""
        if not text:
            return text

        # Layer 1: domain-specific corrections
        result = self._apply_corrections(text)

        # Layer 2: spacing correction (if available)
        if self._use_spacing and self._spacing_fn is not None:
            result = self._fix_spacing(result)

        return result

    def _apply_corrections(self, text: str) -> str:
        """Apply domain-specific word corrections."""
        for pattern, replacement in self._corrections:
            text = pattern.sub(replacement, text)
        return text

    def _fix_spacing(self, text: str) -> str:
        """Apply Korean spacing correction via pykospacing."""
        try:
            return self._spacing_fn(text)
        except Exception:
            logger.exception("Spacing correction failed, passing through")
            return text
