# zerograsp server

First-stage ZeroGrasp service.

Responsibilities:

- Call `arx-ops` for RGB-D capture.
- Call `langsam server` for target segmentation.
- Run ZeroGrasp inference.
- Centralize grasp planning and coordinate transforms.
- Optionally call `arx-ops` to execute planned poses.

Phase one keeps ZeroGrasp inference behind subprocess wrappers and preserves the original main-chain script as a compatibility/debug entry. Stage two should make `coolas_zerograsp.orchestrator` the only formal business orchestration path for `POST /task/run-grasp`.

