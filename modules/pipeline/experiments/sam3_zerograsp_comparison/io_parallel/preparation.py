"""Same measured handoff, fresh inference, then shared-memory planning."""
from pathlib import Path
import shutil
import time

from PIL import Image

from arx_project.artifacts import finish_candidate_transport
from arx_project.layout import ROOT, read_json, sha256, write_json
from ..cohort import check_pair
from ..replay import restore
from .contracts import verify_io


class Preparation:
    def __init__(self, method, pair, services, executor):
        self.method, self.pair, self.services, self.executor = method, pair, services, executor
        self.generation_started = self.returned = None
        self.timings = {}

    def prepare(self, env, out, entry):
        try:
            result = self._prepare(env, out, entry)
            self.returned = time.perf_counter()
            return result
        except BaseException:
            import traceback
            write_json(out/'infrastructure_error.json', {'error':traceback.format_exc()})
            raise

    def _prepare(self, env, out, entry):
        begun = time.perf_counter()
        frozen = check_pair(self.pair)
        for name in self.pair['input_hashes']:
            target = out/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(frozen/name, target)
        if self.pair.get('common_precondition_failure'):
            write_json(out/'paired_common_failure.json', {'source_result_sha256':self.pair['baseline_result_sha256'],
                'reason':'unchanged_high_workspace_stationing_prerequisite', 'native_grasp_planning_called':False})
            return {'success':False, 'reason':'unable_to_reach_high_workspace', 'motion_started':False}
        state = read_json(out/'state.json')
        restore(env, self.executor, out, state)
        self.timings['input_restore_seconds'] = time.perf_counter()-begun
        Image.fromarray(env.render().cpu().numpy()[0]).save(out/'initial_overview.png')
        write_json(out/'initial_conditions.json', {'spawn':env.spawn,'base_pose':entry['base_pose'],
            'initial_q':env.initial_q.tolist(),'object_bbox':env.object_bbox,'paired_station_world':state['links']['base_link']})
        write_json(out/'capture.json', {'camera':'head','mask_source':'frozen successful SAM3',
            'paired_source':self.pair['baseline_trial'],'input_hashes':self.pair['input_hashes']})
        self.generation_started = time.perf_counter()
        if self.method == 'zerograsp':
            t = time.perf_counter()
            response = self.services.request('zerograsp', out, seed=2026112500+self.pair['schedule_index'])
            self.timings['zerograsp_generation_postprocessing_rpc_seconds'] = time.perf_counter()-t
            count = response['result']['candidate_count']
        else:
            t = time.perf_counter()
            response = self.services.request('rayst3r', out)
            self.timings['rayst3r_rpc_seconds'] = time.perf_counter()-t
            if response['result'].get('point_count', 0) < 20:
                elapsed=time.perf_counter()-self.generation_started
                self.timings.update(generation_delivery_seconds=elapsed,generation_and_initial_planning_seconds=elapsed)
                write_json(out/'plan_result.json', {'success':False,'reason':'rayst3r_empty_pointcloud'})
                return {'success':False,'reason':'rayst3r_empty_pointcloud','motion_started':False}
            original = read_json(ROOT/self.pair['observation']/'graspgenx_raw/inference.json')
            t = time.perf_counter()
            self.services.request('graspgenx', out, seed=original['seed'])
            self.timings['graspgenx_rpc_seconds'] = time.perf_counter()-t
            from repair_contract import make_candidates
            t = time.perf_counter()
            count = make_candidates(out, background=True)
            self.timings['candidate_conversion_publish_seconds'] = time.perf_counter()-t
            if not count:finish_candidate_transport(out)
        self.timings['generation_delivery_seconds'] = time.perf_counter()-self.generation_started
        if count:
            options = {} if self.pair['workspace']['preferred_arm'] is None else {'arm':self.pair['workspace']['preferred_arm']}
            t = time.perf_counter()
            response = self.services.request('curobo_dual_arm', out, **options)
            self.timings['initial_planning_rpc_seconds'] = time.perf_counter()-t
            write_json(out/'initial_plan_response.json',response)
            plan = response['result']
            self.timings['candidate_decode_seconds'] = verify_io(out, require_decode=True)['load_seconds']
        else:
            verify_io(out, require_decode=False)
            plan = {'success':False, 'reason':self.method+'_no_postprocessed_candidate', 'count':0}
            write_json(out/'plan_result.json',plan)
        self.timings['generation_and_initial_planning_seconds'] = time.perf_counter()-self.generation_started
        if not plan['success']:return {'success':False,'reason':plan['reason'],'motion_started':False}
        names = ['state.json','plan_result.json','camera/color.png','camera/depth_m.npy','camera/mask.png','candidates/candidate_pool.json']
        return {'success':True,'prepared_plan':True,'motion_started':False,**self.pair['workspace'],
            'prepared_artifacts':{name:sha256(out/name) for name in names},'paired_source':self.pair['baseline_trial']}
