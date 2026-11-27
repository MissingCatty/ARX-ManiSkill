"""An optional state-update adapter; original selection and planning stay intact."""
from pathlib import Path
import time

from arx_project.layout import write_json
from modules.curobo.gripper_sweep.adapter import narrowing_canonical
from .binding import MemoryBindings, MeasuredStartCollision
from .opening import opening_sweep


def memory_pool(base):
    class MemoryPool(base):
        def __init__(self, *args, **kwargs):
            self.memory_updates_enabled = False
            super().__init__(*args, **kwargs)
            self.memory_bindings = MemoryBindings(self)

        def bind(self, *args, **kwargs):
            if not self.memory_updates_enabled:
                return super().bind(*args, **kwargs)
            with self.stage_timing.stage('world_and_robot_bind'):
                return self.memory_bindings.bind(*args, **kwargs)

        def contact_view(self, *args, **kwargs):
            if not self.memory_updates_enabled:
                return super().contact_view(*args, **kwargs)
            with self.stage_timing.stage('contact_validator_update'):
                return self.memory_bindings.contact_view(*args, **kwargs)

        def validate_narrowing(self, work, state, width):
            if not self.memory_updates_enabled:
                return super().validate_narrowing(work, state, width)
            with self.stage_timing.stage('gripper_narrowing'):
                canonical = narrowing_canonical(state, width)
                result = opening_sweep(self, work / 'narrowing_batch', canonical)
                from modules.curobo.dual_arm.accelerated_adapter import opening_samples
                fractions, target, observed = opening_samples(canonical)
                samples = [float(.044 + f * (target - .044)) for f in fractions]
                first = result.get('first_invalid_sample')
                return {'passed': result['passed'], 'samples': samples if result['passed'] else samples[:first],
                        'first_invalid_sample': first,
                        'batched_receipt': str(work / 'narrowing_batch/opening_sweep.json')}

        def request(self, directory, observe=False, approach=False, lift=False, selector='graspcl'):
            began = time.perf_counter()
            try:
                return super().request(directory, observe=observe, approach=approach, lift=lift, selector=selector)
            except MeasuredStartCollision as error:
                directory = Path(directory)
                name = 'observation_plan' if observe else 'lift_plan' if lift else 'approach_plan' if approach else 'plan'
                result = {'success': False, 'reason': 'measured_start_state_in_collision',
                    'rejection_type': 'planning_constraint', 'original_collision_gate_preserved': True,
                    'wall_seconds': time.perf_counter() - began}
                write_json(directory / (name + '_result.json'), result)
                write_json(directory / (name + '_constraint_rejection.json'), {
                    'assertion': str(error), 'gate_relaxed': False, 'motion_commanded_by_request': False})
                return result
    return MemoryPool


def create(args):
    from modules.curobo.gripper_sweep.runtime import create as original_create
    from modules.curobo.dual_arm.interleaved_adapter import configure
    original = original_create(args, pool_extension=memory_pool)

    class MemoryPlanner(type(original)):
        def opening_sweep(self, work, canonical):
            if not self.pool.memory_updates_enabled:
                return super().opening_sweep(work, canonical)
            began = time.perf_counter()
            with self.pool.stage_timing.stage('gripper_opening'):
                result = opening_sweep(self.pool, work, canonical)
            result['wall_seconds'] = time.perf_counter() - began
            write_json(Path(work) / 'opening_sweep.json', result)
            return result

    return configure(MemoryPlanner(original.pool, original.model, original.root))
