"""User-requested 3 cm lift; retain the original execution and trace checks."""
import inspect
import json
from pathlib import Path
import textwrap
import types

LIFT_DISTANCE_M = .03
MIN_HOLD_LIFT_M = .01
HOLD_SECONDS = 3
PROTOCOL = 'arx_tcp43_native_post_truncation_lift30_hold10_v1'


def replace_function(function, old, new):
    source = textwrap.dedent(inspect.getsource(function))
    assert source.count(old) == 1, 'Frozen function no longer matches the registered change'
    source = source.replace(old, new)
    scope = dict(function.__globals__)
    exec(compile(source, __file__ + ':' + function.__name__, 'exec'), scope)
    generated = scope[function.__name__]
    # Execution/worker hooks are installed later by the original timing wrapper.
    # Keep the original live module globals rather than a copied globals dict.
    result = types.FunctionType(generated.__code__, function.__globals__,
                                function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    result.registered_short_lift_source = source
    return result


def install_verifier(baseline):
    if getattr(baseline.verify_trial, 'short_lift_protocol', None) == PROTOCOL:
        return
    verify = replace_function(baseline.verify_trial, 'minimum>.06', 'minimum>=.01')

    def verify_trial(directory):
        verify(directory)
        path = Path(directory) / 'independent_evaluation.json'
        data = json.loads(path.read_text())
        data.update(evaluation_protocol=PROTOCOL, planned_lift_m=LIFT_DISTANCE_M,
                    minimum_continuous_object_lift_m=MIN_HOLD_LIFT_M,
                    minimum_height_comparison='>=', required_hold_seconds=HOLD_SECONDS)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')

    verify_trial.short_lift_protocol = PROTOCOL
    verify_trial.registered_short_lift_source = verify.registered_short_lift_source
    baseline.verify_trial = verify_trial


def install_execution_policy(run, baseline):
    if getattr(run.trial, 'short_lift_protocol', None) != PROTOCOL:
        trial = replace_function(run.trial,
            'result["minimum_hold_lift_m"] > .06', 'result["minimum_hold_lift_m"] >= .01')
        trial.short_lift_protocol = PROTOCOL
        run.trial = trial
    install_verifier(baseline)


def validate_selected(selected):
    import numpy as np
    grasp = np.asarray(selected['grasp_world'])
    lift = np.asarray(selected['lift_world'])
    expected = grasp.copy()
    expected[2, 3] += LIFT_DISTANCE_M
    assert np.allclose(lift, expected, atol=1e-10, rtol=0), 'Lift target must be exactly +3 cm'

