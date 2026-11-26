"""Toggle only the YAML input encoder through the common planning worker."""
from pathlib import Path

from arx_project.layout import read_json, write_json
from .validation_worker import ValidationPlanner, main as serve


class SerializationPlanner(ValidationPlanner):
    def request(self, directory, *, validation_variant, **options):
        if validation_variant not in ['control', 'fast_binding']:
            raise ValueError('Unknown binding serialization variant')
        pool = self.planner.pool
        pool.fast_binding_enabled = validation_variant == 'fast_binding'
        pool.binding_codec.reset()
        # Both groups retain the same scalar branch validator. Only YAML
        # preparation differs; native B16 IK and every later gate are shared.
        result = super().request(directory, validation_variant='control', **options)
        codec = pool.binding_codec.report()
        write_json(Path(directory)/'binding_serialization.json', codec)
        path = Path(directory)/'validation_metrics.json';metrics = read_json(path)
        metrics.update(binding_variant=validation_variant, binding_codec=codec)
        write_json(path, metrics)
        result['binding_variant'] = validation_variant
        return result


if __name__ == '__main__':
    from modules.curobo.indexed_planning.serialization_service import create
    serve(factory=create, planner_type=SerializationPlanner)
