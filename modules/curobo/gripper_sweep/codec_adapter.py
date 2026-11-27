"""Reuse V8's value-preserving YAML codec on this width-aware planner."""
from functools import wraps
from .adapter import create as original_create
from modules.curobo.indexed_planning.binding_serialization import BindingCodec,BindingInputScope


def attach_codec(pool,module=None):
    if hasattr(pool,'binding_input_scope'):raise RuntimeError('Binding codec already attached')
    if module is None:
        import resident_planner as module
    pool.fast_binding_enabled=False
    pool.binding_codec=BindingCodec(pool.yaml_cache)
    pool.binding_input_scope=BindingInputScope(module,pool.binding_codec)
    original=pool.bind
    @wraps(original)
    def bind(*args,**kwargs):
        if not pool.fast_binding_enabled:return original(*args,**kwargs)
        with pool.binding_input_scope.activate():return original(*args,**kwargs)
    pool.bind=bind
    return pool


def create(args,*,pool_extension=None):
    planner=original_create(args,**({} if pool_extension is None else {'pool_extension':pool_extension}));attach_codec(planner.pool)
    from modules.curobo.indexed_planning.stage_timing import wrap_method
    wrap_method(planner,'prepare_arm',planner.pool.stage_timing,'physical_arm_mapping')
    return planner
