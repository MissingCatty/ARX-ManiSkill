"""CUDA-event stage measurement without synchronizing inside model layers."""
from collections import defaultdict
from contextlib import contextmanager
from types import MethodType
import time
import torch


@contextmanager
def stages(adapter):
    records = defaultdict(list)
    changed = []
    hooks = []
    pending = {}
    def begin(name):
        first, last = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        first.record()
        return (time.perf_counter(), first, last)
    def end(name, started):
        wall, first, last = started
        last.record()
        records[name].append((time.perf_counter()-wall, first, last))
    def wrap(obj, attribute, name):
        original = getattr(obj, attribute)
        def measured(self, *args, **kwargs):
            started = begin(name)
            try:return original(*args, **kwargs)
            finally:end(name, started)
        changed.append((obj, attribute, original))
        setattr(obj, attribute, MethodType(measured, obj))
    wrapper = getattr(adapter, 'native_model', adapter.model)
    network = wrapper.model
    wrap(adapter.dino, 'get_intermediate_layers', 'dino_features')
    wrap(network, 'forward_encoders', 'ray_and_point_encoders')
    wrap(network, 'forward_decoder', 'ray_decoder')
    for label, module in [('geometry_head',network.pts_head), ('mask_head',network.classifier_head)]:
        def before(module, args, label=label):pending[label] = begin(label)
        def after(module, args, output, label=label):end(label, pending.pop(label))
        hooks.extend([module.register_forward_pre_hook(before), module.register_forward_hook(after)])
    result = {}
    try:
        yield result
    finally:
        torch.cuda.synchronize()
        for hook in hooks:hook.remove()
        for obj, attribute, original in reversed(changed):setattr(obj, attribute, original)
        result.update({name:{'calls':len(rows),'cpu_seconds':sum(row[0] for row in rows),
            'gpu_seconds':sum(row[1].elapsed_time(row[2])/1000 for row in rows)} for name,rows in records.items()})
