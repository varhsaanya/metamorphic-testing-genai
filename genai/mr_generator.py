from __future__ import annotations

import json
import logging
import textwrap
from pathlib import Path

from config import MR_CACHE_FILE, OPENAI_API_KEY, OPENAI_MODEL

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = textwrap.dedent("""
You are an expert software-testing researcher specialising in metamorphic
testing of machine-learning models.

Your task: propose Metamorphic Relations (MRs) for a BINARY DEFECT-PREDICTION
model trained on the JM1 NASA software-metric dataset.

### Dataset columns (z-score scaled, all numeric)
loc, v(g), ev(g), iv(g), n, v, l, d, i, e, b, t,
lOCode, lOComment, lOBlank, locCodeAndComment,
uniq_Op, uniq_Opnd, total_Op, total_Opnd, branchCount
Target: defects  (0 = clean, 1 = defective)

### Output format
Return a JSON ARRAY of objects. Each object MUST have EXACTLY these keys:

{
  "name":              string   — unique identifier, e.g. "MR9_HalsteadBalance"
  "description":       string   — one sentence: what the transform does
  "expected_relation": string   — what invariant must hold after transform
  "transform_hint":    string   — Python pseudo-code showing the transform
  "rationale":         string   — why this MR makes software-engineering sense
  "violation_type":    string   — MUST be one of:
                                    "label_invariance"
                                    "monotone_increase"
                                    "monotone_decrease"
                                    "must_be_defective"
                                    "must_be_clean"
  "affected_columns":  array    — list of column name strings (lowercase)
  "scale_factor":      number or null
                               — if transform is multiplicative, the factor
                                 (e.g. 1.05 or 0.5); null if it is a swap
}

### Rules
1. affected_columns entries must be lowercase exact matches from the column list.
2. scale_factor must be a number if the transform multiplies a column; null for swaps.
3. For a swap, affected_columns must have exactly 2 entries and scale_factor must be null.
4. Do NOT add extra keys. Do NOT wrap in markdown. Return raw JSON only.
5. Each MR must be novel — do not repeat MR1-MR8 which already exist:
   MR1=noise, MR2=proportional scaling, MR3=permutation,
   MR4=complexity increase, MR5=dead-code removal,
   MR6=duplication, MR7=extreme outlier, MR8=near-zero.
""").strip()

_USER_PROMPT = textwrap.dedent("""
Generate {n} novel, diverse metamorphic relations for the JM1 defect-prediction model.

Cover a mix of:
- Halstead metric transformations (v, l, d, i, e, b, t)
- McCabe metric transformations (v(g), ev(g), iv(g))
- LOC / comment ratio transformations
- Operator/operand symmetry swaps
- Boundary / stress tests

Return valid JSON array only — no prose, no markdown.
""").strip()

_VALID_COLS = {
    "loc", "v(g)", "ev(g)", "iv(g)", "n", "v", "l", "d", "i", "e", "b", "t",
    "locode", "locomment", "loblank", "locodeandcomment",
    "uniq_op", "uniq_opnd", "total_op", "total_opnd", "branchcount",
}

_VALID_VIOLATION_TYPES = {
    "label_invariance", "monotone_increase", "monotone_decrease",
    "must_be_defective", "must_be_clean",
}

_COL_ALIASES: dict[str, str] = {
    "locode":            "locode",
    "lOCode":            "locode",
    "locomment":         "locomment",
    "lOComment":         "locomment",
    "loblank":           "loblank",
    "lOBlank":           "loblank",
    "locodeandcomment":  "locodeandcomment",
    "lOCodeAndComment":  "locodeandcomment",
    "locCodeAndComment": "locodeandcomment",
    "uniq_op":           "uniq_op",
    "uniq_Op":           "uniq_op",
    "uniq_opnd":         "uniq_opnd",
    "uniq_Opnd":         "uniq_opnd",
    "total_op":          "total_op",
    "total_Op":          "total_op",
    "total_opnd":        "total_opnd",
    "total_Opnd":        "total_opnd",
    "branchcount":       "branchcount",
    "branchCount":       "branchcount",
}


def _normalise_col(col: str) -> str:
    return _COL_ALIASES.get(col, col.lower())


def _repair(mr: dict) -> dict:
    """Best-effort repair of common GPT schema deviations."""
    # Normalise affected_columns
    cols = mr.get("affected_columns", mr.get("affectedColumns", []))
    if isinstance(cols, str):
        cols = [cols]
    mr["affected_columns"] = [_normalise_col(c) for c in cols]

    # Ensure scale_factor is float or None
    sf = mr.get("scale_factor", mr.get("scaleFactor"))
    if sf is not None:
        try:
            mr["scale_factor"] = float(sf)
        except (TypeError, ValueError):
            mr["scale_factor"] = None
    else:
        mr["scale_factor"] = None

    # Lowercase violation_type
    vt = mr.get("violation_type", mr.get("violationType", "label_invariance"))
    mr["violation_type"] = str(vt).strip().lower()

    # Fill missing optional text fields
    mr.setdefault("transform_hint", "")
    mr.setdefault("rationale", "")
    mr.setdefault("description", "")
    mr.setdefault("expected_relation", "")

    return mr

def _load_cache() -> list[dict]:
    p = Path(MR_CACHE_FILE)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return []


def _save_cache(mrs: list[dict], raw_response: str = "") -> None:
    Path(MR_CACHE_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(MR_CACHE_FILE, "w") as f:
        json.dump(mrs, f, indent=2)
    if raw_response:
        raw_path = Path(MR_CACHE_FILE).with_suffix(".raw.txt")
        raw_path.write_text(raw_response, encoding="utf-8")
        log.info("Raw GPT response saved → %s", raw_path)


def generate_mrs_from_genai(n: int = 5, use_cache: bool = True) -> list[dict]:
    """
    Call OpenAI to generate n MR proposals, repair + cache them, return list.
    Falls back gracefully to [] if API key is missing.
    """
    cached = _load_cache()
    if use_cache and len(cached) >= n:
        log.info("Cache hit — returning %d MRs.", n)
        return cached[:n]

    if not OPENAI_API_KEY:
        log.warning(
            "OPENAI_API_KEY is not set.\n"
            "  Export it with:  export OPENAI_API_KEY=sk-..."
        )
        return []

    raw = ""
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(token=OPENAI_API_KEY)

        log.info("→ Calling %s  (requesting %d MRs) …", OPENAI_MODEL, n)
        response = client.chat_completion(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": _USER_PROMPT.format(n=n)},
            ],
            temperature=0.7,
            max_tokens=2048,
        )

        raw = response.choices[0].message.content.strip()
        log.info("← Received %d chars from OpenAI.", len(raw))

        clean = raw
        for fence in ("```json", "```"):
            clean = clean.removeprefix(fence).removesuffix("```").strip()

        mrs = json.loads(clean)
        if isinstance(mrs, dict):
            mrs = [mrs]

        mrs = [_repair(m) for m in mrs]
        _save_cache(mrs, raw_response=raw)
        log.info("✓ Generated and cached %d MRs.", len(mrs))
        return mrs

    except json.JSONDecodeError as exc:
        log.error("JSON parse error: %s", exc)
        if raw:
            log.error("First 500 chars of raw response:\n%s", raw[:500])
        return []
    except Exception as exc:  # noqa: BLE001
        log.error("OpenAI call failed: %s", exc)
        return []


def print_genai_mrs(mrs: list[dict]) -> None:
    if not mrs:
        print("No GenAI MRs to display.")
        return
    print("\n" + "=" * 65)
    print("  GenAI-Proposed Metamorphic Relations")
    print("=" * 65)
    for i, mr in enumerate(mrs, 1):
        print(f"\n  [{i}] {mr.get('name', 'unnamed')}")
        print(f"       Description  : {mr.get('description', '')}")
        print(f"       Invariant    : {mr.get('expected_relation', '')}")
        print(f"       Violation    : {mr.get('violation_type', '')}")
        print(f"       Columns      : {mr.get('affected_columns', [])}")
        print(f"       Scale factor : {mr.get('scale_factor')}")
        print(f"       Transform    : {mr.get('transform_hint', '')}")
        print(f"       Rationale    : {mr.get('rationale', '')}")
    print("=" * 65 + "\n")