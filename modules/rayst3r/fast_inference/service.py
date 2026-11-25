"""Drop-in worker for an independently validated acceleration profile."""
from pathlib import Path

from arx_project.layout import ROOT,sha256
from arx_project.json_io import read_json

PROFILE=ROOT/'configs/experiments/rayst3r_fast_inference_profile.json'


def load_profile(path=PROFILE, *, project_root=ROOT):
    record=read_json(path)
    if (record.get('validated_cases')!=50 or record.get('record_integrity') is not True
            or record.get('geometry_quality_stable') is not True):
        raise RuntimeError('Acceleration profile is not fully validated')
    for name,digest in record['source_hashes'].items():
        if sha256(Path(project_root)/name)!=digest:raise RuntimeError('Validated acceleration source changed: '+name)
    for name,entry in record['weights'].items():
        file=Path(project_root)/name
        if file.stat().st_size!=entry['size'] or sha256(file)!=entry['sha256']:
            raise RuntimeError('Validated reconstruction weights changed: '+name)
    return record


def create(args):
    from modules.rayst3r.memory_efficient.adapter import MemoryEfficientWorker
    from modules.rayst3r.fast_inference.adapter import FastAdapter
    profile=load_profile()
    worker=MemoryEfficientWorker(args)
    worker.model=FastAdapter(baseline=worker.model,variant=profile['variant'])
    worker.model.configuration.pop('native_precision_views_points_weights_and_math_unchanged',None)
    worker.model.configuration.pop('attention_buffer_reuse',None)
    worker.model.configuration['validated_acceleration_profile']=profile['profile_id']
    return worker


class RaySt3RAdapter:
    """Benchmark/client constructor using the selected, validated variant."""
    def __new__(cls):
        from modules.rayst3r.fast_inference.adapter import FastAdapter
        profile=load_profile()
        return FastAdapter(variant=profile['variant'])
