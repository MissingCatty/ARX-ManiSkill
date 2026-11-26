"""Turn only exact ray-vote short circuit on/off in matched planning requests."""
from pathlib import Path

from arx_project.layout import write_json
from .serialization_worker import SerializationPlanner
from .validation_worker import main as serve


class MeshVotePlanner(SerializationPlanner):
    def request(self, directory, *, validation_variant, **options):
        if validation_variant not in ['control', 'early_vote']:
            raise ValueError('Unknown mesh-vote optimization variant')
        pool = self.planner.pool;enabled = validation_variant == 'early_vote'
        pool.mesh_vote_mode(enabled)
        try:
            return super().request(directory, validation_variant='fast_binding', **options)
        finally:
            pool.mesh_vote.assert_original_untouched()
            if not all(engine.full_solver is pool.mesh_solver_pairs[k][int(enabled)]
                       for k, engine in pool.batch_engines.items()):
                raise RuntimeError('Collision variant and full-IK CUDA graph no longer match')
            write_json(Path(directory)/'mesh_vote.json', {'variant': validation_variant,
                'geometry_and_thresholds_unchanged': True, 'separate_CUDA_graphs': True,
                'original_module_unchanged': True, 'contract': pool.mesh_vote.contract})


if __name__ == '__main__':
    from modules.curobo.indexed_planning.mesh_vote_service import create
    serve(factory=create, planner_type=MeshVotePlanner)
