"""Station search with injectable collision/planning gates and no robot commands."""
from copy import deepcopy
from pathlib import Path
import time

from arx_project.layout import write_json
from .placement import SweepChecker, hypothetical_state, station_proposals


class StationSearch:
    def __init__(self, model, settings, backend, *, generator=None, options=None, precheck_factory=None):
        self.model, self.settings, self.backend = model, dict(settings), backend
        self.generator, self.options = generator, dict(options or {})
        self.precheck_factory = precheck_factory

    def select(self, state, pool, measured_target, directory):
        """Return a verified proposal and its checked transit; do not execute it."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        timings = dict.fromkeys(['query', 'hand_table_precheck', 'destination', 'transit', 'endpoint', 'planning'], 0.)
        events = []
        full_calls = endpoint_calls = 0
        checker = SweepChecker(self.model, state, self.settings['transit_clearance_m'])
        precheck = self.precheck_factory(state) if self.generator and self.precheck_factory else None
        budget = self.settings['maximum_full_planning_stations']

        def attempt(proposal, is_map):
            nonlocal full_calls, endpoint_calls
            index = len(events)
            work = directory/f'candidate_{index:04d}'
            event = {**proposal, 'receipt_directory': str(work)}
            events.append(event)
            if is_map and precheck is not None:
                tick = time.perf_counter()
                early = precheck.check(proposal)
                timings['hand_table_precheck'] += time.perf_counter()-tick
                event['hand_table_precheck'] = early
                if not early['passed']:
                    return None
            tick = time.perf_counter()
            gate = checker.check(proposal['base_world'], proposal['lift_m'])
            timings['destination'] += time.perf_counter()-tick
            event['destination_geometry'] = gate
            if not gate['passed']:
                return None
            prospective = hypothetical_state(self.model, state, proposal['base_world'], proposal['lift_m'])
            if is_map:
                candidate = pool['candidates'][proposal['candidate_index']]
                view = {**deepcopy(pool), 'candidates': [deepcopy(candidate)],
                        'source_candidate_indices': [proposal['candidate_index']],
                        'search_view': 'one_unchanged_map_matched_candidate'}
            else:
                view = pool
            write_json(work/'state.json', prospective)
            write_json(work/'candidates/candidate_pool.json', view)
            if is_map:
                write_json(work/'placement_witness.json', proposal)
                endpoint_calls += 1
                tick = time.perf_counter()
                endpoint = self.backend.endpoint(work, proposal['arm'])
                timings['endpoint'] += time.perf_counter()-tick
                event['endpoint'] = endpoint
                if not endpoint['passed']:
                    return None
            tick = time.perf_counter()
            gate, path = checker.path(proposal, self.settings)
            timings['transit'] += time.perf_counter()-tick
            event['transit_geometry'] = gate
            write_json(work/'transit_check.json', gate)
            if not gate['passed']:
                return None
            if full_calls >= budget:
                return None
            full_calls += 1
            tick = time.perf_counter()
            result = self.backend.plan(work, arm=proposal['arm'] if is_map else None)
            timings['planning'] += time.perf_counter()-tick
            event['planning_success'] = bool(result['success'])
            event['planning_reason'] = result.get('reason')
            if not result['success']:
                return None
            if is_map:
                selected = result['selected']
                if selected['index'] != 0 or selected['arm'] != proposal['arm']:
                    raise ValueError('Planner returned a candidate outside the map proposal')
                import numpy as np
                if not np.array_equal(np.asarray(selected['contact_pose_world']),
                                      np.asarray(view['candidates'][0]['pose'])):
                    raise ValueError('Planner modified the native world grasp')
                # Preserve native arm-local receipt indices; expose their global mapping.
                result['selected']['planner_local_index'] = 0
                result['selected']['index'] = proposal['candidate_index']
                result['source_candidate_indices'] = view['source_candidate_indices']
                write_json(work/'plan_result.json', result)
            return {'success': True, 'proposal': proposal, 'path': path, 'transit': gate,
                    'plan_result': result, 'work': str(work)}

        chosen = None
        queries = []
        if self.generator is not None:
            batch_size = self.options.get('candidate_query_limit', len(pool['candidates']))
            scan_limit = min(len(pool['candidates']), self.options.get('candidate_scan_limit', batch_size))
            destination_checks = 0
            for start in range(0, scan_limit, batch_size):
                proposals, query = self.generator.generate(state, pool['candidates'], self.settings, start=start)
                timings['query'] += query['query_seconds']
                queries.append(query)
                write_json(directory/'map_query.json', {'batches': queries, 'query_seconds': timings['query']})
                for proposal in proposals:
                    if (full_calls >= budget or endpoint_calls >= self.options['maximum_endpoint_checks'] or
                            destination_checks >= self.options['maximum_destination_checks']):
                        break
                    before = len(events)
                    chosen = attempt(proposal, True)
                    if 'destination_geometry' in events[before]:
                        destination_checks += 1
                    if chosen is not None:
                        break
                if (chosen is not None or full_calls >= budget or endpoint_calls >= self.options['maximum_endpoint_checks'] or
                        destination_checks >= self.options['maximum_destination_checks']):
                    break
        fallback = chosen is None and full_calls < budget
        if fallback:
            proposals = station_proposals(state, measured_target, self.settings)
            # Retain the baseline's cost order and complete original candidate pool.
            for proposal in proposals:
                if full_calls >= budget:
                    break
                chosen = attempt({**proposal, 'source': 'original_geometry_grid'}, False)
                if chosen is not None:
                    break
        report = {'success': chosen is not None, 'method': 'inverse_reachability' if self.generator else 'geometry_grid',
                  'timing_seconds': timings, 'total_seconds': time.perf_counter()-started,
                  'full_planning_calls': full_calls, 'maximum_full_planning_calls': budget,
                  'endpoint_checks': endpoint_calls, 'fallback_used': fallback and self.generator is not None,
                  'original_candidate_count': len(pool['candidates']), 'events': events,
                  'physical_grasps': 0}
        write_json(directory/'search_result.json', report)
        if chosen:
            write_json(directory/'selected.json', {**chosen['proposal'], 'coarse_plan': chosen['work']+'/plan_result.json',
                       'transit': chosen['transit'], 'grasp_will_be_reobserved_and_replanned': True})
            write_json(directory/'command.json', {'path': chosen['path'], 'goal': chosen['proposal']})
        return chosen, report
