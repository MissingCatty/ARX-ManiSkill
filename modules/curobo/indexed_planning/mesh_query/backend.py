"""Instance-local dispatch; original and fast IK keep separate CUDA graphs."""
from functools import wraps
from types import FunctionType, MethodType


class MeshVoteSwitch:
    def __init__(self, scene):
        import torch
        import warp as wp
        from curobo._src.geom.collision.wp_autograd import SphereObstacleCollision
        from curobo._src.geom.collision.wp_collision_kernel import sphere_obstacle_collision_kernel
        from .kernels import exact_vote_collision, CONTRACT
        self.enabled = False
        self.contract = CONTRACT
        self.original_forward = SphereObstacleCollision
        original_forward = SphereObstacleCollision.forward

        class WarpProxy:
            def __getattr__(self, key): return getattr(wp, key)

            @staticmethod
            def launch(*args, **kwargs):
                if (kwargs.get('kernel') is sphere_obstacle_collision_kernel
                        and hasattr(kwargs['inputs'][0], 'mesh_ids')):
                    kwargs = dict(kwargs, kernel=exact_vote_collision)
                return wp.launch(*args, **kwargs)

        namespace = dict(original_forward.__globals__, wp=WarpProxy())
        forward_func = FunctionType(original_forward.__code__, namespace, original_forward.__name__,
                                    original_forward.__defaults__, original_forward.__closure__)

        class FastCollision(torch.autograd.Function):
            forward = staticmethod(forward_func)
            backward = staticmethod(SphereObstacleCollision.backward)

        switch = self

        class Choice:
            @staticmethod
            def apply(*args, **kwargs):
                implementation = FastCollision if switch.enabled else SphereObstacleCollision
                return implementation.apply(*args, **kwargs)

        checker = scene.checker;original = checker.get_sphere_distance.__func__
        if original.__globals__['SphereObstacleCollision'] is not SphereObstacleCollision:
            raise ValueError('Original collision backend was already modified')
        namespace = dict(original.__globals__, SphereObstacleCollision=Choice)
        function = FunctionType(original.__code__, namespace, original.__name__,
                                original.__defaults__, original.__closure__)
        checker.get_sphere_distance = MethodType(function, checker)
        self.original_method = original

    def assert_original_untouched(self):
        if self.original_method.__globals__['SphereObstacleCollision'] is not self.original_forward:
            raise RuntimeError('Original collision module was modified')


def mesh_vote_pool(base):
    class MeshVotePool(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            from curobo.inverse_kinematics import InverseKinematics
            from resident_planner import all_kinematics_params
            self.mesh_vote = MeshVoteSwitch(self.contexts['open'].planner.scene_collision_checker)
            self.mesh_solver_pairs = {}
            for capacity, engine in self.batch_engines.items():
                original = engine.full_solver
                config = engine.fresh_config(original.config, capacity, pure=False)
                fast = InverseKinematics(config, scene_collision_checker=original.scene_collision_checker)
                engine.parameters.extend(all_kinematics_params(fast))
                self.mesh_solver_pairs[capacity] = (original, fast)
            self.mesh_vote_mode(False)

        def mesh_vote_mode(self, enabled):
            if self.endpoint_sequence is not None:
                raise RuntimeError('Cannot change collision graphs during a candidate search')
            self.mesh_vote.assert_original_untouched()
            self.mesh_vote.enabled = bool(enabled)
            for capacity, engine in self.batch_engines.items():
                engine.full_solver = self.mesh_solver_pairs[capacity][int(enabled)]

    return MeshVotePool
