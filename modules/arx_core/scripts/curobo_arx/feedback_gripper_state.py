"""Bound numerical PhysX prismatic overshoot when updating an ARX planning model.

Only planning-model locks are projected. Raw state, USD limits and actuator
commands are unchanged. The 20 micrometer maximum projection is one percent of
the existing 2 mm conservative finger collision buffer. Larger errors stop.
"""
import json
import math

import torch


def project_for_planning(update, measured, out):
    raw = {name: float(measured[name]) for name in update.names}
    if not all(math.isfinite(value) for value in raw.values()):
        raise ValueError('Measured gripper state exceeds numeric planning projection tolerance: nonfinite state')
    projected = {name: min(max(raw[name], update.limits[name][0]), update.limits[name][1])
                 for name in update.names}
    delta = {name: abs(projected[name] - raw[name]) for name in update.names}
    maximum = max(delta.values())
    report = {'raw_measured_m': raw, 'projected_model_m': projected,
              'usd_limits_m': update.limits, 'projection_m': delta,
              'maximum_allowed_projection_m': .00002,
              'existing_finger_collision_buffer_m': .002,
              'physical_usd_or_actuator_changes': False}
    destination = out / 'gripper_model_projection.json'
    if maximum > .00002:
        report['accepted'] = False
        destination.write_text(json.dumps(report, indent=2))
        raise ValueError('Measured gripper state exceeds numeric planning projection tolerance')
    # Independently rebuild the bounded ARX model and check the fast update.
    expected = update.load(projected)
    bounded = update.evaluate(projected)
    original = update.evaluate(raw)
    errors, offsets = {}, {}
    for name, value in bounded.items():
        reference = getattr(expected, name)
        assert torch.allclose(value, reference, atol=1e-6, rtol=1e-6), name
        errors[name] = float((value - reference).abs().max())
        offsets[name] = float((value - original[name]).abs().max())
    assert max(offsets.values()) <= .000021, 'Gripper model projection changed geometry beyond the numerical bound'
    report.update(accepted=True, independently_rebuilt_max_error=max(errors.values()),
                  maximum_model_tensor_change=max(offsets.values()))
    destination.write_text(json.dumps(report, indent=2))
    return {**measured, **projected}
