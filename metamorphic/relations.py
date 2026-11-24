from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    MR_NOISE_SCALE,
    MR_PERMUTATION_SEED,
    MR_SCALE_FACTOR,
)

class MetamorphicRelation:
    name: str = "base"
    description: str = ""
    expected_relation: str = ""

    def transform(
        self, X: pd.DataFrame, stats: dict
    ) -> pd.DataFrame:                          
        raise NotImplementedError

    def check(
        self,
        y_orig: np.ndarray,
        y_trans: np.ndarray,
        prob_orig: np.ndarray | None = None,
        prob_trans: np.ndarray | None = None,
    ) -> np.ndarray:
        return y_orig != y_trans


class MR1_AdditiveNoise(MetamorphicRelation):
    name = "MR1_AdditiveNoise"
    description = (
        "Add small Gaussian noise (scale = MR_NOISE_SCALE × feature std) "
        "to all continuous features."
    )
    expected_relation = "Output label must remain unchanged."

    def transform(self, X: pd.DataFrame, stats: dict) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        X_t = X.copy()
        for col in X.columns:
            sigma = stats[col]["std"] * MR_NOISE_SCALE
            X_t[col] += rng.normal(0, max(sigma, 1e-9), size=len(X))
        return X_t


class MR2_ProportionalScaling(MetamorphicRelation):
    name = "MR2_ProportionalScaling"
    description = (
        f"Multiply all features by {MR_SCALE_FACTOR} "
        "(uniform proportional scale-up)."
    )
    expected_relation = "Output label must remain unchanged."

    def transform(self, X: pd.DataFrame, stats: dict) -> pd.DataFrame:
        return X * MR_SCALE_FACTOR


class MR3_FeaturePermutation(MetamorphicRelation):
    name = "MR3_FeaturePermutation"
    description = (
        "Shuffle values within predefined interchangeable feature groups "
        "(operand counts, operator counts)."
    )
    expected_relation = "Output label must remain unchanged."

    _GROUPS = [
        ["uniq_op", "uniq_opnd"],
        ["total_op", "total_opnd"],
        ["v(g)", "ev(g)", "iv(g)"],
    ]

    def transform(self, X: pd.DataFrame, stats: dict) -> pd.DataFrame:
        rng = np.random.default_rng(MR_PERMUTATION_SEED)
        X_t = X.copy()
        for group in self._GROUPS:
            present = [c for c in group if c in X_t.columns]
            if len(present) > 1:
                vals = X_t[present].values.copy()
                # Shuffle columns (not rows) → swap feature roles
                rng.shuffle(vals.T)
                X_t[present] = vals
        return X_t


class MR4_MonotoneComplexityIncrease(MetamorphicRelation):
    name = "MR4_MonotoneComplexityIncrease"
    description = (
        "Double LOC and cyclomatic complexity (v(g), ev(g), iv(g), n, e)."
    )
    expected_relation = (
        "Defect probability must be ≥ original after complexity increase."
    )

    _COMPLEXITY_COLS = ["loc", "v(g)", "ev(g)", "iv(g)", "n", "e", "branchcount"]

    def transform(self, X: pd.DataFrame, stats: dict) -> pd.DataFrame:
        X_t = X.copy()
        for col in self._COMPLEXITY_COLS:
            if col in X_t.columns:
                X_t[col] *= 2.0
        return X_t

    def check(self, y_orig, y_trans, prob_orig=None, prob_trans=None):
        if prob_orig is not None and prob_trans is not None:
            return prob_trans < prob_orig - 1e-6   
        return y_orig > y_trans   


class MR5_DeadCodeRemoval(MetamorphicRelation):
    name = "MR5_DeadCodeRemoval"
    description = (
        "Halve LOC, lOCode, lOComment, lOBlank "
        "(simulating dead-code / comment removal)."
    )
    expected_relation = (
        "Defect probability must be ≤ original after code shrinkage."
    )

    _SHRINK_COLS = ["loc", "locode", "locomment", "loblank", "locodeandcomment"]

    def transform(self, X: pd.DataFrame, stats: dict) -> pd.DataFrame:
        X_t = X.copy()
        for col in self._SHRINK_COLS:
            if col in X_t.columns:
                X_t[col] *= 0.5
        return X_t

    def check(self, y_orig, y_trans, prob_orig=None, prob_trans=None):
        if prob_orig is not None and prob_trans is not None:
            return prob_trans > prob_orig + 1e-6
        return y_orig < y_trans   # label should not go 0 → 1


class MR6_ConsistentDuplication(MetamorphicRelation):
    name = "MR6_ConsistentDuplication"
    description = "Duplicate each row exactly (identity transform)."
    expected_relation = "Prediction for duplicated row == prediction for original."

    def transform(self, X: pd.DataFrame, stats: dict) -> pd.DataFrame:
        return X.copy()   # pure duplication

    def check(self, y_orig, y_trans, prob_orig=None, prob_trans=None):
        return y_orig != y_trans   # must be identical


class MR7_ExtremeOutlier(MetamorphicRelation):
    name = "MR7_ExtremeOutlier"
    description = (
        "Set complexity features to mean + 10 × std "
        "(extreme outlier injection)."
    )
    expected_relation = "Prediction must be defective (1) for extreme complexity."

    _COMPLEXITY_COLS = ["loc", "v(g)", "ev(g)", "iv(g)", "n", "e", "branchcount"]

    def transform(self, X: pd.DataFrame, stats: dict) -> pd.DataFrame:
        X_t = X.copy()
        for col in self._COMPLEXITY_COLS:
            if col in X_t.columns:
                m, s = stats[col]["mean"], stats[col]["std"]
                X_t[col] = m + 10 * max(s, 1e-9)
        return X_t

    def check(self, y_orig, y_trans, prob_orig=None, prob_trans=None):
        return y_trans != 1


class MR8_NearZeroComplexity(MetamorphicRelation):
    name = "MR8_NearZeroComplexity"
    description = "Set all complexity features to 1e-3 (near-zero)."
    expected_relation = "Prediction must be non-defective (0) for trivial module."

    def transform(self, X: pd.DataFrame, stats: dict) -> pd.DataFrame:
        return pd.DataFrame(
            np.full_like(X.values, 1e-3),
            columns=X.columns,
            index=X.index,
        )

    def check(self, y_orig, y_trans, prob_orig=None, prob_trans=None):
        return y_trans != 0

ALL_MRS: list[MetamorphicRelation] = [
    MR1_AdditiveNoise(),
    MR2_ProportionalScaling(),
    MR3_FeaturePermutation(),
    MR4_MonotoneComplexityIncrease(),
    MR5_DeadCodeRemoval(),
    MR6_ConsistentDuplication(),
    MR7_ExtremeOutlier(),
    MR8_NearZeroComplexity(),
]

MR_BY_NAME: dict[str, MetamorphicRelation] = {mr.name: mr for mr in ALL_MRS}