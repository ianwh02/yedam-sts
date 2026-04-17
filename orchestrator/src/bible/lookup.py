"""Bible verse lookup via Supabase.

Detects verse references in Korean STT output (e.g. "에베소서 4장 2절")
and fetches the exact English text from the bible_verses table.

This is a simple key lookup, not RAG — we know the exact reference.
"""

from __future__ import annotations

import logging
import re

import httpx

logger = logging.getLogger(__name__)

# ── Korean book name → English mapping ──────────────────────────────
# Covers the 66 canonical books. Korean names that STT commonly produces.
_BOOK_MAP: dict[str, str] = {
    # Old Testament
    "창세기": "Genesis",
    "출애굽기": "Exodus",
    "레위기": "Leviticus",
    "민수기": "Numbers",
    "신명기": "Deuteronomy",
    "여호수아": "Joshua",
    "사사기": "Judges",
    "룻기": "Ruth",
    "사무엘상": "1 Samuel",
    "사무엘하": "2 Samuel",
    "열왕기상": "1 Kings",
    "열왕기하": "2 Kings",
    "역대상": "1 Chronicles",
    "역대하": "2 Chronicles",
    "에스라": "Ezra",
    "느헤미야": "Nehemiah",
    "에스더": "Esther",
    "욥기": "Job",
    "시편": "Psalms",
    "잠언": "Proverbs",
    "전도서": "Ecclesiastes",
    "아가": "Song of Solomon",
    "이사야": "Isaiah",
    "예레미야": "Jeremiah",
    "예레미야애가": "Lamentations",
    "에스겔": "Ezekiel",
    "다니엘": "Daniel",
    "호세아": "Hosea",
    "요엘": "Joel",
    "아모스": "Amos",
    "오바댜": "Obadiah",
    "요나": "Jonah",
    "미가": "Micah",
    "나훔": "Nahum",
    "하박국": "Habakkuk",
    "스바냐": "Zephaniah",
    "학개": "Haggai",
    "스가랴": "Zechariah",
    "말라기": "Malachi",
    # New Testament
    "마태복음": "Matthew",
    "마가복음": "Mark",
    "누가복음": "Luke",
    "요한복음": "John",
    "사도행전": "Acts",
    "로마서": "Romans",
    "고린도전서": "1 Corinthians",
    "고린도후서": "2 Corinthians",
    "갈라디아서": "Galatians",
    "에베소서": "Ephesians",
    "빌립보서": "Philippians",
    "골로새서": "Colossians",
    "데살로니가전서": "1 Thessalonians",
    "데살로니가후서": "2 Thessalonians",
    "디모데전서": "1 Timothy",
    "디모데후서": "2 Timothy",
    "디도서": "Titus",
    "빌레몬서": "Philemon",
    "히브리서": "Hebrews",
    "야고보서": "James",
    "베드로전서": "1 Peter",
    "베드로후서": "2 Peter",
    "요한일서": "1 John",
    "요한이서": "2 John",
    "요한삼서": "3 John",
    "유다서": "Jude",
    "요한계시록": "Revelation",
    # Common STT variants / abbreviations
    "갈라비아서": "Galatians",
    "에배소서": "Ephesians",
    "에페소서": "Ephesians",
    "빌립보 서": "Philippians",
}

# Build regex: match any book name + chapter + optional verse(s)
# Pattern: [book] N장 (N절)? or [book] N장 N절에서 N절
_book_names = sorted(_BOOK_MAP.keys(), key=len, reverse=True)
_book_pattern = "|".join(re.escape(b) for b in _book_names)

# Matches: "에베소서 4장 2절", "에베소서 4장 1절에서 3절", "에베소서 4장"
_VERSE_REF_RE = re.compile(
    rf"({_book_pattern})\s*(\d+)\s*장\s*(?:(\d+)\s*절)?(?:\s*(?:에서|부터|-)\s*(\d+)\s*절)?"
)


def detect_verse_references(text: str) -> list[dict]:
    """Extract verse references from Korean text.

    Returns a list of dicts: {book, chapter, verse_start, verse_end}
    """
    refs = []
    for m in _VERSE_REF_RE.finditer(text):
        korean_book = m.group(1)
        english_book = _BOOK_MAP.get(korean_book)
        if not english_book:
            continue

        chapter = int(m.group(2))
        verse_start = int(m.group(3)) if m.group(3) else None
        verse_end = int(m.group(4)) if m.group(4) else verse_start

        refs.append({
            "book": english_book,
            "chapter": chapter,
            "verse_start": verse_start,
            "verse_end": verse_end,
            "korean_ref": m.group(0),
        })

    return refs


class BibleLookup:
    """Fetches verse text from Supabase bible_verses table."""

    def __init__(self, supabase_url: str, supabase_key: str, translation: str = "kjv"):
        self._url = supabase_url.rstrip("/")
        self._key = supabase_key
        self._translation = translation
        self._client: httpx.AsyncClient | None = None
        # Session-level cache: detected verses persist for the whole session
        self._cache: dict[str, str] = {}

    async def initialize(self):
        self._client = httpx.AsyncClient(timeout=10.0)

    async def shutdown(self):
        if self._client:
            await self._client.aclose()

    async def fetch_verses(
        self, book: str, chapter: int, verse_start: int | None, verse_end: int | None
    ) -> str | None:
        """Fetch verse text from Supabase. Returns formatted string or None."""
        # Skip chapter-only references — fetching an entire chapter
        # would blow the LLM context window (max_model_len=2048).
        if verse_start is None:
            logger.debug("Skipping chapter-only reference: %s %d", book, chapter)
            return None

        cache_key = f"{book}:{chapter}:{verse_start}-{verse_end}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if not self._client:
            return None

        # Build Supabase REST query
        params = {
            "select": "verse,text",
            "translation": f"eq.{self._translation}",
            "book": f"eq.{book}",
            "chapter": f"eq.{chapter}",
            "order": "verse",
        }
        if verse_start is not None and verse_end is not None:
            params["verse"] = f"gte.{verse_start}"
            params["verse"] += f"&verse=lte.{verse_end}"
            # Use proper Supabase AND filtering
            del params["verse"]
            params["and"] = f"(verse.gte.{verse_start},verse.lte.{verse_end})"
        elif verse_start is not None:
            params["verse"] = f"eq.{verse_start}"

        try:
            resp = await self._client.get(
                f"{self._url}/rest/v1/bible_verses",
                params=params,
                headers={
                    "apikey": self._key,
                    "Authorization": f"Bearer {self._key}",
                },
            )
            resp.raise_for_status()
            rows = resp.json()
        except Exception:
            logger.warning("Bible lookup failed for %s", cache_key, exc_info=True)
            return None

        if not rows:
            return None

        # Format: "Book Chapter:Verse-Verse — text"
        verses_text = " ".join(row["text"] for row in rows)
        v_start = rows[0]["verse"]
        v_end = rows[-1]["verse"]
        if v_start == v_end:
            ref = f"{book} {chapter}:{v_start}"
        else:
            ref = f"{book} {chapter}:{v_start}-{v_end}"

        result = f"[{ref}] {verses_text}"
        self._cache[cache_key] = result
        logger.info("Bible lookup: %s", ref)
        return result

    async def lookup_from_text(self, korean_text: str) -> list[str]:
        """Detect and fetch all verse references in Korean text."""
        refs = detect_verse_references(korean_text)
        results = []
        for ref in refs:
            text = await self.fetch_verses(
                ref["book"], ref["chapter"], ref["verse_start"], ref["verse_end"]
            )
            if text:
                results.append(text)
        return results
