# Implementation / execution handoff

Updated 2026-10-30 ~10:33 Asia/Shanghai. This is implementation status, not completed collection acceptance.

User requested implementing the four specifications in `research_plan/docs/automated_teacher_pipeline_20261029/` in a new independent directory. Implementation is `scripts/curobo_arx/data_factory/`. Preserve preexisting dirty worktree, all old S02 artifacts, robot/scene physics and original evaluator. No training, commit or push performed.

## Current continuation (2026-10-30)

Authoritative active continuation is `runs/data_factory_20261030_15/`, launched ~10:24 Asia/Shanghai from the repository root, compute GPU2 / simulator GPU3, with current indices resolved to UUIDs in `launch.json`. It retains one native ARX environment and four-attempt object blocks. At ~10:31, fresh bagel inference/qualification completed (64 reserved G rejected before motion), and the live group rotated to apple. Both scenes passed runtime physics fingerprints with zero differences. Main-thread diagnostic samples have been written without another crash at this observation; long-run stability is still pending. No new physical grasp/admission at this snapshot. Read `state.json` and `REPORT.md` for later changes; never relaunch a running registration.

13 ended with a native segmentation fault at ~08:29, after adding six successes (raisin 3, BBQ 3). Cumulative accepted episodes inherited by 15: 11 (mustard 5, raisin 3, BBQ 3). There are 26 consumed physical attempts, including two unknown physical executions across the lineage; a separate old claim was never physically dispatched.

12/13 faults align exactly with asynchronous 120-s stack dumps. New real ARX differential pressure: asynchronous sampling reproduced a segmentation fault; main-thread sampling completed 250 steps and the diagnostic hold with unchanged physics fingerprints. Interpreter-only comparison did not reproduce. `diagnostics.py` removes the asynchronous timed walker and adds main-thread safe-point snapshots. 46 offline checks pass. See `DIAGNOSIS_WATCHDOG.md` for evidence and limits; long-run collection stability is not yet established. Diagnostic holds contribute zero grasp demonstrations.

14 copied/recovered the ledger without changing old files, then the coordinator was mistakenly started from the source snapshot root: external Python paths resolved under the snapshot and startup failed before any simulator dispatch. Its STOP, launch, traceback and unchanged inherited ledger are retained. Its owned processes have exited. 15 inherits 14, retaining the complete lineage, 11 successes, 26 consumed physical attempts and original 2026-10-31 03:13:30 deadline. It must start the coordinator from the repository root as documented; workers/simulators still use the registered snapshot.

The resume gate now also handles a proved pre-simulator exit: owned coordinator/processes must have exited, a launch record must exist, no group registration may exist, and Podman inventory must exclude orphaned factory simulators. Three offline fault tests cover these constraints; the full suite passes 49 checks (`offline_checks_diagnostics_02.log`). The idle old LiLo policy service was gracefully stopped after no clients/experiment consumers were found; this released our GPU2 memory without terminating another user's work. Resource placement was freshly checked before launch.

Historical 13 launch status below is retained for context:

13 launched ~07:11 Asia/Shanghai, compute GPU1 / simulator GPU3, **one** native ARX environment, four-dispatch object blocks. Initial real physics fingerprint PASS with zero differences; fresh cube observation generated 6,060 raw/64 reserved network candidates. At ~07:15 it was qualifying G (no new physical command yet).

Inherited: five accepted demonstrations and seven ordinary failures, plus two records recovered only in the copied ledger (one unknown physical execution; one claimed before dispatch). 12 ledger/STOP/record bytes were preserved. Exit proof, recovery receipt, source/dependency equivalence and original deadline 2026-10-31 03:13:30 are registered in 13. Counter/budget inheritance is verified; 43 offline checks pass. No native SDK root cause is claimed; single-environment continuation reduces concurrency pending stability evidence. Read 13 `state.json`, `REPORT.md`, `groups/*/FAULT.json`, and `workers/*/FAULT.json` before further operations.

The following earlier execution snapshots are historical.

## Execution status

Snapshot 2026-10-30 03:38:28 CST. Latest production registration **`runs/data_factory_20261030_12/`**, **running** on compute GPU1 and simulator GPU3, native group **`group_55dc094f185d`**, two ARX environments. Read SQLite/REPORT and FAULT/state files for later changes; never relaunch a running registration.

Both real environments passed historical runtime physics comparison with **zero mismatches**. Fresh mustard RGB-D inference produced **7,384 raw proposals**, 44 official post-collision proposals; the diversified raw reserve admitted **1 of 64** through frozen planning gates.

At this snapshot: **7 physical command attempts, 3 newly admitted demonstrations, 3 actual failures**, other attempts in progress. First 10 mm entry touched the target during reaching and is retained as a normal failure. **40 and 60 mm entries completed actual approach, bilateral 3 s closure, lift and 3 s hold and were committed after independent physical/data checks.** These are newly executed trajectories, not historical replays.

First success `be6d11adee574037bb8ad0db60b063d1`: hold minimum rise **0.1185712218 m**, hold duration **3.0000001565 s**, zero forbidden contacts, zero dense-invalid samples/limit exceedance. Quality PASS: **111 raw wrist RGB-D samples, 31 action chunks, one actual onset for each local phase**. Real export `export_0001/` contains a stable snapshot of 1 accepted episode, 31 chunks and 80 training transitions, with streaming statistics/quantile bounds. This is export validation, not VLA training.

Video `videos/be6d11adee574037bb8ad0db60b063d1.mp4`: **640×480, 5 FPS, 12.8 s, all 64 frames decoded successfully** using the existing bundled imageio-FFmpeg. `first_decode_receipt.json` binds the source hash. The earlier `first_decode.log` records a missing system executable, not a corrupt-video result; the bundled-decoder receipt supersedes that command failure. Six-object/video coverage is still incomplete.

Full original catalog, asset order, inactive reset indices, placements and quotas remain registered; mustard was chosen as the first queue tie only. Target 34,304, max physical 137,216, wall 24h are bounds/targets, not achieved counts. Previous registrations and STOP/fault receipts remain intact; do not reuse their launch directories.

## Implemented / checked

Native nonblocking per-environment state machines; independent local reset and contact/camera routing; original within-environment collision groups plus cross-clone exclusions; shared World stepping and original RTX sensors on the same render tick. Native frozen-physics validation passes on two real clones. Original camera calibration/rendering/AA/resolution and .2s samples retained. No tiled-camera claim.

Resident ZeroGrasp and serial shared open/closed cuRobo contexts; exact original tool conversion, model relocation, FK guard, measured gripper update, nominal closed-G/lift gate, IK/PRM/trajectory optimization and dense collision gates. Global G dedup/reserve, actual fresh G per attempt, object leases with no heartbeat reclaim, coverage quotas and finite budgets. Same-renderer object rotation remains in the same World (real cube→can→raisin observed); renderer rotation requires a new group.

Bounded asynchronous raw writer, durable request/execution chain checks including orphan executions, original physical evaluator and independent new-protocol quality checks, atomic durable SQLite admission and content/actual-entry dedup. Streaming index export and train-only mean/std/min/max; deterministic six-clip selection and background CPU MP4 support. The first two new native episodes exercised the complete execution→physical acceptance→data validation→durable admission path.

Latest offline suite: **39/39 passed**, `offline_checks_v17.log` (factory + numerical export tests). Tests include real historical ARX five-stage success and wrist/onset fixtures, not new production demonstrations. They also cover actual ARX USD collision membership, cross-environment identity, missing commands, incomplete writes, interrupted claims, leases and duplicate counting.

Derived Kit geometry/shader cache persists at `runs/data_factory_runtime_cache/<GPU UUID>/kit`. Never reuse trajectories as new data. One native start plus mesh export, model warmup, inference, planning and IO all count in wall-clock throughput.

## Preserved prior records

01: stopped before dispatch after role mismatch. 02: stopped before dispatch after historical placement mismatch. 03 and 05: prepared only, never launched. 04: four actual clone/camera contexts, overlapping-collision-group defect, stopped idle before dispatch. 06: 31 fingerprint mismatches (30 API schema-order and missing /World scope), fault before dispatch; fixed without relaxing values. 07: both clone fingerprints pass, real NN inference; SceneCfg API integration fault, three H dispatch records but **no physical commands**. 08: launch fails on initially absent image Kit-cache directory; no native simulation. 09 as above. Never merge those preparation/integration faults into new successful collection counts.

Own old idle S02 simulators were stopped with fresh idle/recorders-off evidence in the development directory. Other users' jobs and own old LiLo server on GPU2 were untouched.

## Remaining limits

1. Full 34,304-episode/67-object/G/H coverage is not complete. Native mustard success does not establish generality or full production throughput.
2. Scene export/JSON serialization and terminal quality/fsync still block the shared main loop; improve immutable-mesh/pose cache and background finalization without omitting actual scene changes.
3. Selected-video background encoder and corrected GPU interval accounting are in current code and future immutable registrations; old 09 results remain unchanged. Never claim six clips exist when no motion has been recorded.
4. Explicit `resume.py` now creates an append-only continuation with verified process/container exit, copied-ledger interruption recovery, original budget/deadline and inherited success receipts. It never relaunches an old registration or automatically retries an unknown command.
5. Local **1 GiB** reserve is enforced; automatic dedicated `/coolas/data` continuation is not implemented. Local space currently ample.
6. Local asset importer exists, but additional-asset historical-fingerprint handling remains unverified. Runtime support beyond the frozen 67-object scene must not be claimed yet.
7. Export now includes two-pass histogram quantile intervals with explicit per-dimension value-error bounds and numerical checks. One real new accepted episode has now passed export validation; full-scale export/normalization performance remains unmeasured. No VLA training or service integration is part of this task.

10 initialized the two real contexts and exited idle with 0 dispatches/physical/admissions after a mistaken cadence concern. Installed World.step implementation and identical measured times (15 render calls -> 3.0 s plus initial physics ticks) confirm original .2 s render stepping already matched the contract. The STOP is preserved; no simulation timing was changed. Current code adds actual render-advance and early frame-interval guards. New 11 registration was launched at about 03:03; inspect its state/logs for later progress.

12 will register mustard as the first queue tie choice while retaining original catalog/asset order (therefore inactive-object reset indices), all 67 objects, placements and quotas. This is the same formal production queue; no historical G/q is replayed and no separate pilot dataset is created. A test explicitly protects asset/reset order from scheduling priority. Boundary-noise checking now precedes the existing planner interior margin, matching original 1e-4 raw-limit allowance. All old STOP/fault/registration files remain.


## 2026-10-30 user-authorized continuation

12 ended 03:51 with a native SDK segmentation fault, 5 durable successes, 7 ordinary failures, one physical execution without final receipt and one claimed/not-dispatched task. No original process/container remains. The crash's exact cause is unconfirmed; last periodic stack involved a PhysX joint-target write but does not establish causality.

Current resume.py verifies process/container exit, all accepted hashes and frozen dependency equality, then creates a new registration/SQLite copy. Only the copy receives recovery transitions. Original deadline, quotas and consumed budget persist; cumulative/new successes and native infrastructure stops are separate. Four new offline recovery cases pass with existing checks (43 total). No repeated unknown command or old STOP removal.

13 will use one ARX environment on the available simulator GPU, four-dispatch object blocks, fresh observations for deficient objects. All original assets/order/placement/physics/models/gates remain frozen. This is an explicit bounded continuation, not an automatic restart loop or an SDK-root-cause claim. Inspect 13 state.json/REPORT.md for actual launch progress.
