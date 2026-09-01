# Synthetic hardware profiles

These JSON files are synthetic descriptions of hardware the project does not own. They are shaped like `nmesh doctor --json` output so they can exercise planner behavior and regression tests without pretending to measure hardware.

Plans computed from these files are planning decisions, not runtime measurements. The profiles are for exercising planner behavior and locking in regressions; they do not verify backend performance or GPU runtime correctness.
