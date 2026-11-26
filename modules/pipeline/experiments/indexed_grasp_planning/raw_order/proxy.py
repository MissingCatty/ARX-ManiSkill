"""Share the unchanged original candidate sequence in both comparison groups."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from arx_project.layout import ROOT, write_json
from arx_project.artifacts import publish_candidate_pool, finish_candidate_transport
from arx_project.service import serve
from modules.graspgenx.postprocessing.ordering import make_queue
from ..physical.proxy import OrderingProxy


def original_queue(payload):
    queue, manifest = make_queue(payload, 'control')
    check_original_order(manifest)
    if queue['candidates'] != payload['candidates']:
        raise ValueError('Original candidate values changed')
    return queue, manifest


def check_original_order(manifest):
    count = manifest['source_count']
    if (manifest['variant'] != 'control' or manifest['NMS'] is not None or manifest['deferred']
            or manifest['representatives'] != list(range(count))
            or manifest['priority_to_original_source'] != list(range(count))
            or manifest['candidate_pose_score_width_and_metadata_changes']
            or manifest['geometry_hard_filter_enabled']):
        raise ValueError('The raw-order comparison changed candidate order or geometry')


class OriginalOrderProxy(OrderingProxy):
    def request(self, directory, *, execution_variant, case_id, **flags):
        directory = Path(directory)
        if execution_variant not in ['baseline', 'optimized'] or case_id not in self.cases:
            raise ValueError('Request is outside the registered raw-order comparison')
        if not directory.resolve().is_relative_to(self.root.resolve()):
            raise ValueError('Wrong output root')
        stage = 'lift' if flags.get('lift') else 'approach' if flags.get('approach') else 'initial'
        if stage != 'initial':
            return self.forward(directory, stage, execution_variant, flags)
        source = ROOT/self.cases[case_id]['input_directory']/'candidates/candidate_pool.json'
        begin = time.perf_counter(); data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != self.reg['input_sha256'][str(source)]:
            raise ValueError('Frozen original candidate pool changed')
        payload = json.loads(data); loaded = time.perf_counter()-begin
        begin = time.perf_counter(); queue, manifest = original_queue(payload)
        preparation = time.perf_counter()-begin
        write_json(directory/'candidate_order.json', manifest)
        descriptor = publish_candidate_pool(directory, queue)
        result = self.forward(directory, stage, execution_variant,
                              {**flags, 'candidate_transport': descriptor})
        finish_candidate_transport(directory)
        if self.reg.get('campaign_stage') == 'physical' and result['success']:
            mode = 'control_1' if execution_variant == 'baseline' else 'optimized_1'
            expected = json.loads((ROOT/self.reg['planning_gate_source']/'plans'/case_id/mode/'plan_result.json').read_text())['selected']
            actual = result['selected']
            same = all(expected.get(k)==actual.get(k) for k in ['arm','original_source_index','contact_pose_world'])
            write_json(directory/'raw_order_handoff.json',{'passed':same,
                'expected_original_source_index':expected['original_source_index'],
                'actual_original_source_index':actual['original_source_index'],
                'same_original_grasp_and_arm':same})
            if not same:
                raise ValueError('Physical initial plan differs from the accepted original-order replay')
        write_json(directory/'candidate_preparation.json', {'source_read_seconds': loaded,
            'postprocessing_seconds': preparation, 'variant': execution_variant,
            'all_original_candidates_retained': True, 'original_order_unchanged': True,
            'NMS_enabled': False, 'candidate_source': str(source.relative_to(ROOT))})
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['root', 'socket', 'warmup-trial', 'descriptor', 'campaign', 'planner-socket']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--kind', default='curobo_indexed_planning'); args = parser.parse_args()
    serve(args, worker=OriginalOrderProxy(args.campaign, args.planner_socket))
