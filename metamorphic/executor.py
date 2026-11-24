from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from metamorphic.relations import MetamorphicRelation

log = logging.getLogger(__name__)


@dataclass
class MRResult:
    mr_name:          str
    n_samples:        int
    n_violations:     int
    violation_rate:   float
    exec_time_s:      float
    violated_indices: list[int] = field(default_factory=list)
    prob_orig:        np.ndarray | None = None
    prob_trans:       np.ndarray | None = None

    def summary(self) -> str:
        return (
            f"[{self.mr_name}] "
            f"violations={self.n_violations}/{self.n_samples} "
            f"({self.violation_rate:.2%}) | "
            f"time={self.exec_time_s:.3f}s"
        )


def run_mr(
    mr: MetamorphicRelation,
    model,
    X: pd.DataFrame,
    stats: dict,
) -> MRResult:
    t0 = time.perf_counter()

    # Original predictions
    y_orig  = model.predict(X)
    p_orig  = (
        model.predict_proba(X)[:, 1]
        if hasattr(model, "predict_proba") else None
    )

    # Transform
    X_trans = mr.transform(X, stats)

    # Transformed predictions
    y_trans = model.predict(X_trans)
    p_trans = (
        model.predict_proba(X_trans)[:, 1]
        if hasattr(model, "predict_proba") else None
    )

    # Detect violations
    violations = mr.check(y_orig, y_trans, p_orig, p_trans)
    viol_idx   = np.where(violations)[0].tolist()

    elapsed = time.perf_counter() - t0

    result = MRResult(
        mr_name          = mr.name,
        n_samples        = len(X),
        n_violations     = int(violations.sum()),
        violation_rate   = float(violations.mean()),
        exec_time_s      = elapsed,
        violated_indices = viol_idx,
        prob_orig        = p_orig,
        prob_trans       = p_trans,
    )
    log.info(result.summary())
    return result


def run_all_mrs(
    mrs: list[MetamorphicRelation],
    model,
    X: pd.DataFrame,
    stats: dict,
) -> list[MRResult]:
    results = []
    for mr in mrs:
        log.info("Running %s …", mr.name)
        r = run_mr(mr, model, X, stats)
        results.append(r)
    return results