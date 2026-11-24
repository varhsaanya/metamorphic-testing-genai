"""
Fallback mock GenAI MRs — used when --mock flag is passed or the HF API fails.
These mirror the schema expected by dynamic_mr_builder.
"""

MOCK_GENAI_MRS = [
    {
        "name": "MR9_HalsteadVolumeIncrease",
        "description": "Increase Halstead volume (v) by 20% to simulate a more complex module.",
        "expected_relation": "Defect probability must be >= original after volume increase.",
        "transform_hint": "X['v'] *= 1.20",
        "rationale": "Higher Halstead volume indicates more complex logic, which should increase defect likelihood.",
        "violation_type": "monotone_increase",
        "affected_columns": ["v"],
        "scale_factor": 1.20,
    },
    {
        "name": "MR10_HalsteadDifficultyIncrease",
        "description": "Increase Halstead difficulty (d) by 25%, simulating harder-to-understand code.",
        "expected_relation": "Defect probability must be >= original after difficulty increase.",
        "transform_hint": "X['d'] *= 1.25",
        "rationale": "Higher difficulty score correlates with more error-prone code.",
        "violation_type": "monotone_increase",
        "affected_columns": ["d"],
        "scale_factor": 1.25,
    },
    {
        "name": "MR11_OperatorOperandSwap",
        "description": "Swap unique operators (uniq_op) and unique operands (uniq_opnd) columns.",
        "expected_relation": "Output label must remain unchanged after operand/operator swap.",
        "transform_hint": "X[['uniq_op','uniq_opnd']] = X[['uniq_opnd','uniq_op']].values",
        "rationale": "Operator and operand counts play symmetrical roles in Halstead complexity.",
        "violation_type": "label_invariance",
        "affected_columns": ["uniq_op", "uniq_opnd"],
        "scale_factor": None,
    },
    {
        "name": "MR12_BranchCountDoubling",
        "description": "Double branchCount to simulate additional conditional paths.",
        "expected_relation": "Defect probability must be >= original after branch count doubling.",
        "transform_hint": "X['branchcount'] *= 2.0",
        "rationale": "More branches increase cyclomatic complexity and defect risk.",
        "violation_type": "monotone_increase",
        "affected_columns": ["branchcount"],
        "scale_factor": 2.0,
    },
    {
        "name": "MR13_LOCCommentReduction",
        "description": "Halve lOComment to simulate removal of documentation comments.",
        "expected_relation": "Defect probability must be >= original after comment reduction.",
        "transform_hint": "X['locomment'] *= 0.5",
        "rationale": "Fewer comments reduce maintainability and may indicate higher defect density.",
        "violation_type": "monotone_increase",
        "affected_columns": ["locomment"],
        "scale_factor": 0.5,
    },
]
