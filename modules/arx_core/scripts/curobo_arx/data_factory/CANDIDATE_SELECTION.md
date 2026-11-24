# Candidate selection correction (2026-10-30)

Production registration `runs/data_factory_20261030_09` stopped with zero dispatched/physical episodes and zero admissions. Both native ARX environments passed their historical runtime-physics fingerprint checks. Two freshly observed objects each rejected all 64 reserved G candidates; these raw outcomes remain intact.

The first selection implementation iterated string-sorted surface/direction/roll group keys, truncating the first pass at 64. Real Rubik's-cube output demonstrates the resulting bias: selected surface-region counts were X−21, X+13, Y−17, Y+13, Z−0, Z+0. Thus a purportedly diverse pool silently omitted both vertical regions.

`balanced_group_order` now orders groups by deficient surface-region, approach-direction and roll counts, with network score and a stable key as tie breakers. It retains actual network G, the existing coordinate transform, the 64-G reserve, symmetric deduplication and every frozen robot/collision constraint. No downward-only filter or fabricated grasp is added.

Reprocessing the same 6,010 real network rows produced X−12, X+11, Y−11, Y+10, Z−10, Z+10 candidates. All six approach-direction classes also remain represented (10–11 each). This establishes the coverage correction, not physical grasp success. The bounded archived-scene IK comparison is recorded in `runs/data_factory_development_20261030/ik_diagnosis_03/`.

A model-only measured-current-EEF/lift probe accepted the actual starting pose under the production planning-scene convention. Diagnosis 01 temporarily included robot-body meshes as world obstacles; inspection of the historical `prepare.py` showed that this differs from established scene preparation. That diagnostic is retained as development evidence only and is not a production-model comparison. Diagnosis 02 restores the historical external-geometry convention; diagnosis 03 changes candidate selection only.

Diagnosis 03 completed: the balanced pool still rejected all 64 fixed G. Diagnosis 04 accepted both the actual historical cube G and measured current EEF/lift under the same production-model convention. This supports a fresh-proposal feasibility problem for this cube input; the coverage fix alone has not produced a usable G or any new demonstration. Historical G remains diagnostic only.
