"""Compare instrumented and plain requests without changing their solutions."""
from pathlib import Path

from arx_project.layout import write_json
from .serialization_worker import SerializationPlanner
from .validation_worker import main as serve


class ProfilePlanner(SerializationPlanner):
    def request(self, directory, *, validation_variant, **options):
        if validation_variant not in ['control', 'profiled']:
            raise ValueError('Unknown IK profiling variant')
        trace = self.planner.pool.ik_trace
        trace.begin(Path(directory), validation_variant == 'profiled')
        try:
            return super().request(directory, validation_variant='fast_binding', **options)
        finally:
            write_json(Path(directory)/'ik_profile.json', trace.finish())


if __name__ == '__main__':
    from modules.curobo.indexed_planning.ik_profile_service import create
    serve(factory=create, planner_type=ProfilePlanner)
