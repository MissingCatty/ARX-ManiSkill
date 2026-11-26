"""Resident CPU NMS and shared-memory delivery to the physical planner."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from arx_project.layout import ROOT, now, write_json
from arx_project.client import Client
from arx_project.artifacts import publish_candidate_pool, finish_candidate_transport
from arx_project.service import serve
from modules.graspgenx.postprocessing.ordering import make_queue


class OrderingProxy:
    def __init__(self, campaign, planner_socket):
        self.root = Path(campaign)
        self.reg = json.loads((self.root/'registration.json').read_text())
        self.cases = {c['id']: c for c in self.reg['cases']}
        self.client = Client(planner_socket, timeout=1800, candidate_transport=True)

    def request(self, directory, *, execution_variant, case_id, **flags):
        directory = Path(directory)
        if execution_variant not in ['baseline', 'optimized'] or case_id not in self.cases:
            raise ValueError('Request is outside the fixed physical schedule')
        if not directory.resolve().is_relative_to(self.root.resolve()): raise ValueError('Wrong output root')
        stage = 'lift' if flags.get('lift') else 'approach' if flags.get('approach') else 'initial'
        if stage != 'initial':
            return self.forward(directory, stage, execution_variant, flags)
        source = ROOT/self.cases[case_id]['input_directory']/'candidates/candidate_pool.json'
        begin = time.perf_counter();data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != self.reg['input_sha256'][str(source)]:
            raise ValueError('Frozen candidate pool changed')
        payload = json.loads(data);loaded = time.perf_counter()-begin
        begin = time.perf_counter();ordered, manifest = make_queue(payload,
            'control' if execution_variant == 'baseline' else 'nms15')
        post_seconds = time.perf_counter()-begin
        write_json(directory/'candidate_order.json', manifest)
        descriptor = publish_candidate_pool(directory, ordered)
        result = self.forward(directory, stage, execution_variant, {**flags, 'candidate_transport': descriptor})
        finish_candidate_transport(directory)
        write_json(directory/'candidate_preparation.json', {'source_read_seconds': loaded,
            'postprocessing_seconds': post_seconds, 'variant': execution_variant,
            'all_original_candidates_retained': True, 'candidate_source': str(source.relative_to(ROOT))})
        return result

    def forward(self, directory, stage, variant, flags):
        begin = time.perf_counter()
        response = self.client.request(directory, execution_variant=variant, **flags)
        elapsed = time.perf_counter()-begin
        write_json(directory/('planner_rpc_'+stage+'.json'), {'at': now(), 'variant': variant,
            'seconds': elapsed, 'response': response})
        if response['status'] != 'ok': raise RuntimeError('Physical planner RPC failed: '+str(response))
        return response['result']


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ['root', 'socket', 'warmup-trial', 'descriptor', 'campaign', 'planner-socket']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--kind', default='curobo_indexed_planning');args = p.parse_args()
    serve(args, worker=OrderingProxy(args.campaign, args.planner_socket))
