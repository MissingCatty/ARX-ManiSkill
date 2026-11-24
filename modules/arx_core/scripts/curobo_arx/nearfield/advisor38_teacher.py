"""Uniform advisor38 seed policy under the original ARX models and limits."""
from scripts.curobo_arx.data_factory.teacher import Teacher as GeometryTeacher
from .advisor37_teacher import Teacher as Previous
from .advisor37_geometry import clearance_bound


class Teacher(Previous):
    def authorize(self):
        if self.registration['stage'] not in ('advisor38_canonical_seed_preflight', 'advisor38_model_prefilter'):
            raise RuntimeError('Advisor38 explicit registration required')
        protocol = self.registration['advisor38_seed_protocol']
        if not protocol['remove_nominal_closed_preflight_gate'] or protocol['training_authorized']:
            raise RuntimeError('Advisor38 seed-only authorization mismatch')

    def visual_goal(self, request, world):
        result = GeometryTeacher.visual_goal(self, request, world)
        if result is None:
            raise RuntimeError('Every object requires the same camera clearance check')
        if clearance_bound(result) < .010:
            result = {**result, 'pass': False,
                      'reasons': [*result.get('reasons', []), 'uniform_camera_clearance_below_10mm']}
        return result


if __name__ == '__main__':
    import argparse
    from scripts.curobo_arx.data_factory.common import read
    from scripts.curobo_arx.seed_augmentation import worker
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', required=True)
    parser.add_argument('--registration', required=True)
    args = parser.parse_args()
    worker.Teacher = Teacher
    worker.serve(args.directory, read(args.registration))
