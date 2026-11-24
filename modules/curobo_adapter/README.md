# Independent cuRobo adapter

This module retains cuRobo version comparisons, IK and collision diagnostics, and the planner endpoint. Shared native candidate provenance and ARX pose conversion live in `modules/maniskill/planning_common/`.

The approved action frame is `simulation/robots/arx/runtime/action_tcp.json`: EEF +Z offset 43 mm, action X=EEF +Y, action Y=EEF -X, action Z=EEF +Z. `live/action_tcp.py` loads that robot-owned profile and uses the shared rigid-frame operations in `modules/arx_core/scripts/curobo_arx/action_frame.py`.

The EASG cloud worker, cloud rollout launcher, prompts and VLM configuration have been removed. `live/service.py` accepts only the planner kind `curobo`. This module does not start an experiment on import. Historical records remain immutable and do not authorize new runs.
