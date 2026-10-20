"""Experiment-local adapters; historical model/workflow contracts stay unchanged."""

METHODS = ("semantic_editor", "ot_editor", "state_flow", "ot_flow", "ot_gaussian")
EDITORS = METHODS[:2]
ACTIONS = ({"X": 0}, {"X": 1}, {"X": 2}, {"X": 3})
AUDIT_COLUMNS = ("X", "T", "D", "U", "M", "P", "L", "H", "A")
FORMAT = "hz_misspecification_v1"
