"""Optional C-backed YAML receipts; original binding/physics code still runs."""
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
import threading
import time
from types import FunctionType, SimpleNamespace

import yaml


class BindingCodec:
    def __init__(self, cache=None):
        if not getattr(yaml, '__with_libyaml__', False):
            raise RuntimeError('The optional binding codec requires installed LibYAML')
        self.cache = {} if cache is None else cache
        self.reset()

    def reset(self):
        self.events = []

    def safe_load(self, text):
        began = time.perf_counter()
        hit = isinstance(text, str) and text in self.cache
        value = deepcopy(self.cache[text]) if hit else yaml.load(text, Loader=yaml.CSafeLoader)
        self.events.append({'operation': 'load', 'cache_hit': hit,
                            'wall_seconds': time.perf_counter()-began})
        return value

    def safe_dump(self, value, **options):
        if 'Dumper' in options:
            raise ValueError('The binding codec only permits CSafeDumper')
        began = time.perf_counter()
        text = yaml.dump(value, Dumper=yaml.CSafeDumper, **options)
        self.events.append({'operation': 'dump', 'bytes': len(text.encode()),
                            'wall_seconds': time.perf_counter()-began})
        return text

    def function(self, original):
        # Only this function's YAML dependency changes. The global PyYAML
        # module and all its constructors/representers remain untouched.
        namespace = {**original.__globals__, 'yaml': SimpleNamespace(
            safe_load=self.safe_load, safe_dump=self.safe_dump)}
        result = FunctionType(original.__code__, namespace, original.__name__,
                              original.__defaults__, original.__closure__)
        result.__kwdefaults__ = original.__kwdefaults__
        return result

    def report(self):
        return {'events': list(self.events), 'load_calls': sum(e['operation'] == 'load' for e in self.events),
            'dump_calls': sum(e['operation'] == 'dump' for e in self.events),
            'cache_hits': sum(e.get('cache_hit', False) for e in self.events),
            'wall_seconds': sum(e['wall_seconds'] for e in self.events),
            'global_PyYAML_unchanged': True}


class BindingInputScope:
    """Temporarily select the input encoder inside one serial resident bind."""
    def __init__(self, module, codec):
        self.module = module
        self.original = module.bind_inputs
        self.fast = codec.function(self.original)
        self.owner = None

        @wraps(self.original)
        def guarded(*args, **kwargs):
            if self.owner != threading.get_ident():
                raise RuntimeError('Binding serialization was invoked outside its owner thread')
            return self.fast(*args, **kwargs)
        self.guarded = guarded

    @contextmanager
    def activate(self):
        if self.owner is not None or self.module.bind_inputs is not self.original:
            raise RuntimeError('Nested binding or changed original input policy')
        self.owner = threading.get_ident()
        self.module.bind_inputs = self.guarded
        try:
            yield
        finally:
            unchanged = self.module.bind_inputs is self.guarded
            self.module.bind_inputs = self.original
            self.owner = None
            if not unchanged:
                raise RuntimeError('Another writer changed the active input binding function')


def serialization_pool(base):
    class SerializationPool(base):
        def __init__(self, *args, **kwargs):
            self.fast_binding_enabled = False
            super().__init__(*args, **kwargs)
            import resident_planner
            # Retain the same warm reference cache and copy-on-read behavior.
            self.binding_codec = BindingCodec(self.yaml_cache)
            self.binding_input_scope = BindingInputScope(resident_planner, self.binding_codec)

        def bind(self, *args, **kwargs):
            if not self.fast_binding_enabled:
                return super().bind(*args, **kwargs)
            with self.binding_input_scope.activate():
                return super().bind(*args, **kwargs)
    return SerializationPool
