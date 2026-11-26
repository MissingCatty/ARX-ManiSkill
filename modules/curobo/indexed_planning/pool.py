"""Provide batched endpoints through the unchanged local-contact gates."""
import time
from pathlib import Path

import numpy as np

from arx_project.layout import read_json, write_json
from modules.curobo.candidate_io.store import candidate_pool
from .batches import EndpointSequence, NativeBatchEngine


def batched_pool(base, capacities=(16,), selected_capacity=16):
    class BatchedPool(base):
        def __init__(self, root, warmup_trial):
            self.endpoint_sequence = self.endpoint_row = None
            self.ik_batch_size = selected_capacity
            self.batch_engines = {}
            super().__init__(root, warmup_trial)
            state = read_json(Path(warmup_trial)/'state.json')
            context = self.contexts['open']
            start = context.measured(state)
            goal = np.asarray(state['links']['right_eef_link'])
            reports = []
            import torch
            for capacity in capacities:
                began = time.perf_counter()
                engine = NativeBatchEngine(self, capacity)
                engine.bind(context)
                native = engine.native([goal]*capacity, start)
                pure = engine.pure([goal]*capacity, start)
                if any(q is None for q in native) or any(not len(q) for q in pure):
                    raise RuntimeError('Batched IK warmup failed at the recorded measured pose')
                torch.cuda.synchronize()
                reports.append({'capacity': capacity, 'warmup_seconds': time.perf_counter()-began,
                    'native_measured_pose_passes': len(native), 'pure_measured_pose_passes': len(pure),
                    'calls': engine.calls})
                engine.reset_seed()
                self.batch_engines[capacity] = engine
            write_json(Path(root)/'batch_ik_readiness.json', {'passed': True,
                'reports': reports, 'physical_motions': 0, 'scene_shared_with_original': True,
                'original_solver_configuration_except_batch_capacity': True})

        def select_contact(self, context, state, directory, validator):
            if self.ik_batch_size == 0:
                return super().select_contact(context, state, directory, validator)
            if self.endpoint_sequence is not None:
                raise RuntimeError('Endpoint batch requests cannot overlap')
            engine = self.batch_engines[self.ik_batch_size]
            engine.bind(context)
            sequence = EndpointSequence(candidate_pool(directory)['candidates'], context.measured(state), engine)
            self.endpoint_sequence = sequence
            try:
                result = super().select_contact(context, state, directory, validator)
                result['batch_ik'] = sequence.summary()
                return result
            finally:
                write_json(context.out/'batch_ik.json', sequence.summary())
                self.endpoint_sequence = self.endpoint_row = None

        def grasp_endpoint(self, context, state, grasp, start, validator):
            if self.endpoint_sequence is None or state.get('kinematic_endpoint_witness') is not None:
                return super().grasp_endpoint(context, state, grasp, start, validator)
            self.endpoint_row = self.endpoint_sequence.next(grasp)
            try:
                return super().grasp_endpoint(context, state, grasp, start, validator)
            finally:
                self.endpoint_row = None

        def native_grasp_ik(self, context, grasp, start):
            if self.endpoint_row is not None:
                return self.endpoint_row['native']
            return super().native_grasp_ik(context, grasp, start)

        def unconstrained_ik(self, context, world, start):
            if self.endpoint_row is not None:
                value = self.endpoint_row['pure']
                if value is None:
                    raise RuntimeError('Missing prefetched kinematic fallback')
                return value
            return super().unconstrained_ik(context, world, start)
    return BatchedPool
