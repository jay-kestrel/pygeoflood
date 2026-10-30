"""pygeoflood benchmark harness (spec deliverable D1).

Times the canonical FIM workload — ``run_fim_workflow()`` on the OC1mTest
DEM with ``custom_Q=500`` — end to end and per stage, cold (fresh process,
capturing the numba JIT tax) and warm (repeated in-process), on the full
16.0M-pixel tile and a fixed 2000x2000 crop. Results are JSON plus a
markdown summary, with a full environment manifest on every run.

Deliberately lightweight at import time: ``benchmarks.schema`` must stay
importable without pygeoflood or any heavy geo dependency so the schema
unit tests run anywhere.
"""
