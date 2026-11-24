"""Benchmark selection only. Each model owns its reconstruction implementation."""
from arx_project.registry import load_component

SUPPORTED_METHODS = ('zerograsp', 'rayst3r')


def load_adapter(method):
    names={'zerograsp':'ZeroGraspAdapter','rayst3r':'RaySt3RAdapter'}
    if method not in names:raise ValueError('Unsupported reconstruction method: '+method)
    return getattr(load_component(method,'reconstruction'),names[method])()
