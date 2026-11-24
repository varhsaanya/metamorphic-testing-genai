from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from metamorphic.relations import MetamorphicRelation

log = logging.getLogger(__name__)

_REQUIRED_KEYS = {"name", "description", "expected_relation",
                  "transform_hint", "violation_type", "affected_columns"}

def validate_mr_dict(mr_dict: dict) -> tuple[bool, str]:
    missing = _REQUIRED_KEYS - set(mr_dict.keys())
    if missing:
        return False, f"Missing keys: {missing}"

    vt = mr_dict.get("violation_type", "")
    valid_vts = {
        "label_invariance", "monotone_increase",
        "monotone_decrease", "must_be_defective", "must_be_clean",
    }
    if vt not in valid_vts:
        return False, f"Unknown violation_type '{vt}'. Must be one of {valid_vts}"

    cols = mr_dict.get("affected_columns", [])
    if not isinstance(cols, list) or len(cols) == 0:
        return False, "affected_columns must be a non-empty list"

    return True, ""


def _make_transform(mr_dict: dict):
    cols         = mr_dict["affected_columns"]
    scale_factor = mr_dict.get("scale_factor")  # may be None
    name         = mr_dict["name"]

    if scale_factor is not None:
        # Multiplicative transform
        factor = float(scale_factor)

        def transform(X: pd.DataFrame, stats: dict) -> pd.DataFrame:
            X_t = X.copy()
            for col in cols:
                c = col.lower()
                if c in X_t.columns:
                    X_t[c] = X_t[c] * factor
                else:
                    log.debug("[%s] column '%s' not found — skipped", name, c)
            return X_t

    elif len(cols) == 2:
        # Swap two columns
        c0, c1 = cols[0].lower(), cols[1].lower()

        def transform(X: pd.DataFrame, stats: dict) -> pd.DataFrame:
            X_t = X.copy()
            if c0 in X_t.columns and c1 in X_t.columns:
                X_t[c0], X_t[c1] = X_t[c1].copy(), X_t[c0].copy()
            else:
                log.warning(
                    "[%s] swap columns %s/%s not both present", name, c0, c1
                )
            return X_t

    else:
        log.warning("[%s] no transform strategy matched — using identity", name)

        def transform(X: pd.DataFrame, stats: dict) -> pd.DataFrame:
            return X.copy()

    return transform

def _make_check(violation_type: str):
    """Return a check(y_orig, y_trans, prob_orig, prob_trans) → ndarray[bool]."""

    if violation_type == "label_invariance":
        def check(y_orig, y_trans, prob_orig=None, prob_trans=None):
            return y_orig != y_trans

    elif violation_type == "monotone_increase":
        def check(y_orig, y_trans, prob_orig=None, prob_trans=None):
            if prob_orig is not None and prob_trans is not None:
                return prob_trans < prob_orig - 1e-6
            return (y_orig == 1) & (y_trans == 0)

    elif violation_type == "monotone_decrease":
        def check(y_orig, y_trans, prob_orig=None, prob_trans=None):
            if prob_orig is not None and prob_trans is not None:
                return prob_trans > prob_orig + 1e-6
            return (y_orig == 0) & (y_trans == 1)

    elif violation_type == "must_be_defective":
        def check(y_orig, y_trans, prob_orig=None, prob_trans=None):
            return np.asarray(y_trans) != 1

    elif violation_type == "must_be_clean":
        def check(y_orig, y_trans, prob_orig=None, prob_trans=None):
            return np.asarray(y_trans) != 0

    else:
        raise ValueError(f"Unknown violation_type: {violation_type}")

    return check


def build_dynamic_mr(mr_dict: dict) -> MetamorphicRelation:
    valid, reason = validate_mr_dict(mr_dict)
    if not valid:
        raise ValueError(f"Invalid MR dict for '{mr_dict.get('name')}': {reason}")

    transform_fn = _make_transform(mr_dict)
    check_fn     = _make_check(mr_dict["violation_type"])

    # Dynamically create a named subclass
    DynamicMR = type(
        mr_dict["name"],          # class name
        (MetamorphicRelation,),   # base class
        {
            "name":              mr_dict["name"],
            "description":       mr_dict["description"],
            "expected_relation": mr_dict["expected_relation"],
            "transform":         lambda self, X, stats, _tf=transform_fn: _tf(X, stats),
            "check":             lambda self, yo, yt, po=None, pt=None, _ch=check_fn: _ch(yo, yt, po, pt),
            # Store full metadata for reporting
            "_mr_dict":          mr_dict,
        },
    )
    return DynamicMR()


def build_all_dynamic_mrs(
    mr_dicts: list[dict],
) -> tuple[list[MetamorphicRelation], list[dict]]:
    valid_mrs    = []
    failed_dicts = []

    for d in mr_dicts:
        ok, reason = validate_mr_dict(d)
        if not ok:
            log.warning("Skipping MR '%s': %s", d.get("name", "?"), reason)
            failed_dicts.append({**d, "_error": reason})
            continue
        try:
            mr = build_dynamic_mr(d)
            valid_mrs.append(mr)
            log.info("Built dynamic MR: %s", mr.name)
        except Exception as exc:  # noqa: BLE001
            log.error("Error building MR '%s': %s", d.get("name"), exc)
            failed_dicts.append({**d, "_error": str(exc)})

    log.info(
        "Dynamic MR build: %d succeeded, %d failed",
        len(valid_mrs), len(failed_dicts),
    )
    return valid_mrs, failed_dicts