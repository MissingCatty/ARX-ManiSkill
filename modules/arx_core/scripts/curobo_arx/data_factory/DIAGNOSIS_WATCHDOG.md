# 2026-10-30 timed stack sampler diagnosis

Evidence directory: `runs/data_factory_development_20261030/watchdog_diagnosis_01/`.
This diagnosis concerns an instrumentation change, not a grasp algorithm or frozen robot/physics change.

## Evidence

Production registrations 12 and 13 crashed after approximately 2285.399 and 1085.059 seconds from their group process registrations. Their logs contain respectively 19 and 9 periodic `Timeout (0:02:00)!` stack dumps and end inside a partially printed Python frame (`File `). Subtracting 19 or 9 times 120 seconds leaves approximately 5.4 or 5.1 seconds of Python startup. This implicates the periodic diagnostic walker; a sampled PhysX/warp frame alone does not establish a fault in that operation.

An interpreter-only metadata-churn comparison using the same image's Python 3.12.13 completed in all three modes (off, asynchronous, main-thread). It did **not** reproduce the crash. These software tests are not physical validation.

The real ARX comparison loads the original ARX robot, full frozen assets, historical white-packer-bottle initial state, RaytracedLighting cameras and original physics. Each case registers one 20-s hold command, then runs rendering/getter pressure. This is a diagnostic hold, not a grasp demonstration. Both runtime fingerprints passed. No data was admitted to the teacher ledger.

| Case | Pressure instrumentation | Outcome |
| --- | --- | --- |
| `arx_async_02` | `dump_traceback_later(.001, repeat=True)` | Native segmentation fault, Python wrapper exit 1, before ten pressure steps; no terminal hold receipt |
| `arx_main_thread_02` | `traceback.print_stack` on the simulator main thread | Exit 0, 250 pressure steps / 60.902 s, 20-s hold receipt present |

The pressure interval is deliberately much shorter than production's 120 s. This controlled comparison supports removing asynchronous sampling; it does not prove the precise native failing instruction or guarantee indefinite production stability. The first launch without the container's required `--user 0` failed on mount permissions; its separate logs are setup errors, not comparison results.

CPython reports this general hazard in [issue 158200](https://github.com/python/cpython/issues/158200) and [issue 140815](https://github.com/python/cpython/issues/140815). The former's reported version/reproducer differs from our Python 3.12.13 environment; do not claim it directly proves the identical CPython bug here. The [Python 3.12 documentation](https://docs.python.org/3.12/library/faulthandler.html) also requires keeping the handler's file descriptor alive until the handler is disabled.

## Change and validation

`diagnostics.py` cancels timed watchdogs and samples its own stack only from `Group.step` on the simulator's main thread. Fatal-signal handling remains enabled for the current thread. Closing disables the handler before closing its file descriptor. There is no timed foreign-frame walk. External coordinator process-exit checks, 900-s attempt limits and all physical/data gates remain unchanged.

The complete offline suite passed 46 checks, including no asynchronous timer registration, rejection of cross-thread diagnostic ticks and descriptor lifetime. See `offline_checks_diagnostics_01.log` in the development directory. Real ARX diagnostic and interpreter results remain separately recorded.

Continue through a new registration and copied-ledger recovery only. Preserve registrations 12 and 13, their STOP files, all 11 admitted episodes, ordinary failures, planning rejects and unknown physical executions. Do not replay an interrupted command or reset the original wall/attempt budgets.
