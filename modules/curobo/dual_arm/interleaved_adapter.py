"""Resume each arm at its next native candidate, alternating bounded windows."""
from copy import deepcopy
from pathlib import Path
import time

from arx_project.layout import ROOT, read_json, write_json
from modules.curobo.candidate_io.store import candidate_pool, candidate_window
from .adapter import DualArmPlanner
from .placement_service import PlacementPlanner


class InterleavedSearch:
    candidate_batch_size = 16
    subsequent_candidate_batch_size = 64

    def request(self, directory, *, station_probe=False, **flags):
        if station_probe:
            return super().request(directory, station_probe=True, **flags)
        if any(flags.get(name) for name in ['approach','lift','observe']):
            return super().request(directory, **flags)
        # The old prefix wrapper only learns a bound after an entire arm
        # request. Window scheduling owns the equivalent global bound here.
        return DualArmPlanner.request(self, directory, **flags)

    def search_arms(self, directory, state, sides, filename, selector):
        for width in [self.candidate_batch_size,self.subsequent_candidate_batch_size]:
            if type(width) is not int or width < 1:
                raise ValueError('Candidate batch size must be a positive integer')
        total = len(candidate_pool(directory)['candidates'])
        if len(sides) < 2 or total == 0:
            return super().search_arms(directory, state, sides, filename, selector)
        plans, searches = {}, {}
        schedule = []
        best = None
        for side in sides:
            work = self.prepare_arm(directory, state, side, filename, planned_open=True)
            sweep = self.opening_sweep(work, read_json(work/filename))
            if not sweep['passed']:
                plans[side] = {'success':False,'reason':sweep['reason'],'opening_sweep':sweep}
                write_json(work/'plan_result.json', plans[side])
            else:
                searches[side] = {'work':work,'cursor':0,'events':[],'rejected':{},'seconds':0.,'windows':[]}
        round_index = 0
        while searches:
            width = self.candidate_batch_size if round_index == 0 else self.subsequent_candidate_batch_size
            for side in list(searches):
                search = searches[side]
                start = search['cursor']
                bound = total if best is None else min(total,best+1)
                if start >= bound:
                    result = deepcopy(plans[side])
                    result.update(reason='no_candidate_in_prefix_can_improve_selected_arm',
                                  full_pool_exhausted=start >= total)
                    plans[side] = result
                    write_json(search['work']/'plan_result.json',result)
                    del searches[side]
                    continue
                # Once one arm finishes, the remaining arm can check its
                # entire competitive prefix without further switching.
                stop = bound if len(searches) == 1 else min(start+width,bound)
                started = time.perf_counter()
                with candidate_window(search['work'],start,stop):
                    result = deepcopy(self.pool.request(search['work'],selector=selector))
                seconds = time.perf_counter()-started
                events = result.get('candidate_events',[])
                for event in events:
                    if not 0 <= int(event['candidate_index']) < stop-start:
                        raise ValueError('Planner event is outside its candidate window')
                    event['candidate_index'] += start
                if result['success']:
                    index = int(result['selected']['index'])
                    if not 0 <= index < stop-start:
                        raise ValueError('Planner selection is outside its candidate window')
                    result['selected']['index'] = index+start
                    best = index+start if best is None else min(best,index+start)
                search['events'].extend(events)
                for name,count in result.get('rejected',{}).items():
                    search['rejected'][name] = search['rejected'].get(name,0)+count
                search['seconds'] += seconds
                search['cursor'] = start+len(events)
                receipt = {'arm':side,'start_index':start,'stop_index_exclusive':stop,
                    'candidate_checks':len(events),'success':result['success'],
                    'selected_index':result.get('selected',{}).get('index'),
                    'best_index_after_window':best,'wall_seconds':seconds}
                schedule.append(receipt)
                search['windows'].append(receipt)
                write_json(search['work']/'search_windows'/f'{start:06d}_{stop:06d}.json',
                           {'window':receipt,'result':result})
                terminal = result['success'] or result.get('reason') != 'no_feasible_native_post_or_recalled_candidate'
                if not terminal and len(events) != stop-start:
                    raise RuntimeError('Planner did not finish its admitted candidate window')
                result.update(candidate_events=deepcopy(search['events']),
                    rejected=dict(search['rejected']),wall_seconds=search['seconds'],
                    candidate_search_windows=list(search['windows']),
                    full_pool_exhausted=not result['success'] and search['cursor'] >= total,
                    full_candidate_count=total)
                if not result['success'] and not terminal:
                    result['count'] = search['cursor']
                    result['reason'] = ('no_feasible_native_post_or_recalled_candidate'
                        if search['cursor'] >= total else 'candidate_search_pending')
                plans[side] = result
                write_json(search['work']/'plan_result.json',result)
                if terminal or search['cursor'] >= total:
                    del searches[side]
            round_index += 1
        write_json(Path(directory)/'interleaved_search.json',{
            'batch_size':self.candidate_batch_size,'full_candidate_count':total,
            'subsequent_batch_size':self.subsequent_candidate_batch_size,
            'best_candidate_index':best,'schedule':schedule,
            'candidate_checks':sum(row['candidate_checks'] for row in schedule),
            'candidate_order_poses_and_scores_unchanged':True,
            'equal_rank_other_arm_checked':True,'repeated_candidate_checks':0})
        return plans


class InterleavedDualArmPlanner(InterleavedSearch, DualArmPlanner):
    pass


class InterleavedPlacementPlanner(InterleavedSearch, PlacementPlanner):
    pass


def create(args):
    from .adapter import create as original_create
    original = original_create(args)
    return configure(InterleavedDualArmPlanner(original.pool,original.model,original.root))


def create_map(args):
    from .placement_service import create as original_create
    original = original_create(args)
    return configure(InterleavedPlacementPlanner(original.pool,original.model,original.root))


def configure(planner):
    policy = read_json(ROOT/'configs/experiments/dual_arm_interleaving.json')
    planner.candidate_batch_size = policy['candidate_batch_size']
    planner.subsequent_candidate_batch_size = policy['subsequent_candidate_batch_size']
    return planner
