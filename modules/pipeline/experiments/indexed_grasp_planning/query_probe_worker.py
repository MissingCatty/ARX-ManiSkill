"""Matched diagnostic requests; standalone component timings are separate."""
from pathlib import Path

from arx_project.layout import write_json
from .serialization_worker import SerializationPlanner
from .validation_worker import main as serve


class QueryPlanner(SerializationPlanner):
    def request(self, directory, *, validation_variant, **options):
        if validation_variant not in ['control', 'captured']:
            raise ValueError('Unknown query component diagnostic variant')
        capture = self.planner.pool.query_capture
        capture.begin(Path(directory), validation_variant == 'captured')
        try:
            return super().request(directory, validation_variant='fast_binding', **options)
        finally:
            write_json(Path(directory)/'query_probe.json', capture.finish())


if __name__ == '__main__':
    from modules.curobo.indexed_planning.query_probe_service import create
    serve(factory=create, planner_type=QueryPlanner)
