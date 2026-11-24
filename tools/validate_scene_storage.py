"""Load the shared library in SAPIEN, without executing a grasp."""
import argparse
import importlib
from pathlib import Path
import sys

from arx_project.assets import load_object_catalog
from arx_project.layout import ROOT, read_json, write_json, now
from arx_project.scenes import current_dataset


def validate(output):
    sys.path.insert(0, str(ROOT / 'modules/pipeline/experiments/current_configuration'))
    validation = importlib.import_module('validate_runtime')
    validation.OUT = output / 'simulation'
    validation.simulation()
    # The existing check verifies native robot loading, calibrated camera and
    # physics while holding the original reset joints. Load every object mesh
    # as well, in an isolated scene with no physics steps or grasp commands.
    from configuration import runtime
    from asset_environment import make_builder
    environment, baseline, run, sapien, torch, make, *_ = runtime()
    dataset = current_dataset()
    models = load_object_catalog(dataset / 'object_catalog.json')['models']
    entry = read_json(dataset / 'manifest.json')['schedule'][0]
    env = make(entry['model_id'], entry['scene'])
    records = []
    try:
        env.reset(seed=entry['seed'])
        for index, (name, record) in enumerate(models.items()):
            builder = make_builder(env.scene, record)
            builder.initial_pose = sapien.Pose([10 + index * 2, 10, 2])
            actor = builder.build_kinematic(name='storage-check-' + name)
            shapes = actor._bodies[0].get_collision_shapes()
            assert shapes, name
            assert all(abs(shape.contact_offset - .001) < 1e-8 for shape in shapes), name
            records.append({'model_id': name, 'collision_parts': len(shapes),
                            'visual_parts': len(record['visuals'])})
            print('ASSET_PASS', name, len(shapes), flush=True)
        env.get_obs()
    finally:
        env.close()
    report = {'passed': True, 'at': now(), 'scenes': 200, 'models_loaded': len(records),
              'robot_table_camera_physics_passed': True, 'assets': records,
              'physical_grasps': 0}
    write_json(output / 'runtime_validation.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    report = validate(parser.parse_args().output)
    print({key: value for key, value in report.items() if key != 'assets'})
