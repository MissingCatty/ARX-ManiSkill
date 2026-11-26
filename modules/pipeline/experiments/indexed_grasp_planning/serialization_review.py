"""Check selected physical endpoints and every generated robot YAML value."""
import json
from pathlib import Path

import yaml

from arx_project.layout import now, sha256, write_json
from .validation_review import validate


def raw(path): return json.loads(Path(path).read_text())


def review(root):
    reg = raw(root/'registration.json');accepted = raw(root/'acceptance.json')
    checked = validate(root)
    cache, pairs = {}, 0
    def value(path):
        digest = sha256(path)
        if digest not in cache:
            cache[digest] = json.dumps(yaml.load(path.read_text(), Loader=yaml.CSafeLoader),
                sort_keys=True, allow_nan=False, separators=(',', ':'))
        return cache[digest]
    for case in reg['cases']:
        repeats = [1, 2] if 'control_2' in reg['variants'] else [1]
        for repeat in repeats:
            control = root/'plans'/case['id']/f'control_{repeat}'
            fast = root/'plans'/case['id']/f'fast_binding_{repeat}'
            paths = {str(p.relative_to(control)) for p in control.rglob('*.yml')}
            if paths != {str(p.relative_to(fast)) for p in fast.rglob('*.yml')}:
                raise ValueError('Generated binding model receipts differ in coverage')
            for name in paths:
                if value(control/name) != value(fast/name):
                    raise ValueError('Generated binding configuration values differ: '+name)
                pairs += 1
            for directory, enabled in [(control, False), (fast, True)]:
                codec = raw(directory/'binding_serialization.json')
                if (codec['dump_calls'] > 0) != enabled:
                    raise ValueError('The registered input codec was not actually selected')
                if not codec['global_PyYAML_unchanged']:
                    raise ValueError('The input codec changed global PyYAML')
    checked.update(at=now(), generated_robot_YAML_pairs=pairs,
        all_generated_configuration_values_equal=True,
        primary_time_metric=reg['primary_time_metric'], timing_gates=accepted['timing_gates'])
    write_json(root/'independent_review.json', checked)
    return checked
