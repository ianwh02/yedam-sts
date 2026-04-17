"""Glossary registry for denomination-specific translation terms.

Two loading modes:
1. **Supabase** (preferred): Fetches glossary templates and org-specific
   glossaries from the ``glossaries`` table. Enabled when
   ``BIBLE_SUPABASE_URL`` is set (reuses the same Supabase connection).
2. **Local JSON** (fallback): Loads all ``.json`` files from a directory
   at startup. Used when Supabase is not configured.

Sessions reference a glossary by name (e.g. ``"nazarene"``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_registry: dict[str, dict] = {}


# ── Local JSON loading (fallback) ──────────────────────────────

def load_glossaries(directory: str) -> int:
    """Load all .json files from *directory* into the registry.

    Returns the number of glossaries loaded.
    """
    d = Path(directory)
    if not d.is_dir():
        logger.warning("Glossary directory not found: %s", d)
        return 0

    count = 0
    for f in sorted(d.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            glossary_id = f.stem
            _registry[glossary_id] = data
            logger.info(
                "Loaded glossary '%s' (%s)",
                glossary_id,
                data.get("denomination", "unknown"),
            )
            count += 1
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Failed to load glossary %s: %s", f, e)

    return count


# ── Supabase loading ───────────────────────────────────────────

def load_glossaries_from_supabase(supabase_url: str, supabase_key: str) -> int:
    """Fetch all template glossaries from Supabase into the registry.

    Returns the number of glossaries loaded.
    """
    try:
        from supabase import create_client
        client = create_client(supabase_url, supabase_key)

        # Fetch all templates (org_id is null)
        resp = client.table("glossaries").select("name, denomination, terms").eq("is_template", True).execute()

        count = 0
        for row in resp.data:
            name = row["name"]
            terms = row["terms"] if isinstance(row["terms"], dict) else json.loads(row["terms"])
            terms["denomination"] = row.get("denomination", "")
            _registry[name] = terms
            logger.info("Loaded glossary '%s' from Supabase (%s)", name, row.get("denomination", "unknown"))
            count += 1

        return count
    except Exception:
        logger.exception("Failed to load glossaries from Supabase")
        return 0


def load_org_glossary(supabase_url: str, supabase_key: str, org_id: str) -> dict | None:
    """Fetch an org-specific glossary from Supabase (not cached).

    Returns the glossary dict, or None if the org has no custom glossary.
    """
    try:
        from supabase import create_client
        client = create_client(supabase_url, supabase_key)

        resp = (
            client.table("glossaries")
            .select("name, denomination, terms")
            .eq("org_id", org_id)
            .limit(1)
            .execute()
        )

        if not resp.data:
            return None

        row = resp.data[0]
        terms = row["terms"] if isinstance(row["terms"], dict) else json.loads(row["terms"])
        terms["denomination"] = row.get("denomination", "")
        return terms
    except Exception:
        logger.exception("Failed to load org glossary for %s", org_id)
        return None


# ── Public API ─────────────────────────────────────────────────

def get_glossary(glossary_id: str) -> dict | None:
    """Return a glossary by ID, or None if not found."""
    return _registry.get(glossary_id)


def list_glossaries() -> list[str]:
    """Return all registered glossary IDs."""
    return list(_registry.keys())
