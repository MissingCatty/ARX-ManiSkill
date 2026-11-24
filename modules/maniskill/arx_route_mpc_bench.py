"""Registered comparison of global route tracking plus local vision MPC."""
import argparse
import inspect
from pathlib import Path
import shutil
import sys

import arx_route_mpc as route
import arx_mpc_approach_bench as bench

SELF = Path(__file__).resolve()
CODE = SELF.parent
PREVIOUS = CODE / 'outputs/arx_graspcl_mpc_approach_20261118_02'
read, save, sha = bench.read, bench.save, bench.sha


def prepare(out, backend_gpu, scene_gpu, cases):
    source = inspect.getsource(bench.prepare)
    old = "slot = canonical + '_MPC01'"
    assert source.count(old) == 1
    namespace = {**bench.__dict__, 'SELF': SELF}
    exec(compile(source.replace(old, "slot = canonical + '_ROUTE01'"), str(SELF) + ':prepare', 'exec'), namespace)
    namespace['prepare'](out, backend_gpu, scene_gpu, cases)
    sources = [Path(module.__file__).resolve() for module in (bench, route, route.guarded)]
    for path in sources:
        shutil.copy2(path, out / 'source' / path.name)
    prior_result = PREVIOUS / 'mpc_results.json'
    assert read(prior_result)['verified']
    for file in (out / 'registration.json', out / bench.ARM / 'registration.json'):
        reg = read(file)
        reg['source_hashes'].update({str(p): sha(p) for p in [*sources, prior_result]})
        reg['frozen_algorithm'] = (
            'One global observation route before capture. After fresh perception, the accepted global plan '
            'supplies pregrasp and approach routes together. MPC tracks both from measured joints at 10 Hz; '
            'the approach enters a TCP neighborhood for wrist-guided refinement. No pregrasp-to-approach '
            'replan. Retain existing candidate gates, measured closed lift, independent physical evaluation.')
        reg['test_design_scope'] = 'Engineering comparison on three fixed reset recipes from the local MPC pilot.'
        reg['mpc'].update(route_stages=['observe', 'pregrasp', 'approach'],
                         route_lookahead_s=route.LOOKAHEAD_S, route_speed_rad_s=route.ROUTE_SPEED_RAD_S,
                         route_corridor_rad=route.ROUTE_CORRIDOR_RAD, route_handoff_tcp_m=route.HANDOFF_TCP_M,
                         local_speed_rad_s=route.guarded.MAX_COMMAND_SPEED_RAD_S,
                         local_max_rolling_solves=route.guarded.MAX_ROLLING_SOLVES)
        save(file, reg)


def summarize(out):
    bench.summarize(out)
    summary = read(out / 'mpc_results.json')
    prev = {case['canonical_case_id']: case for case in read(PREVIOUS / 'mpc_results.json')['cases']}
    for row in summary['cases']:
        trial = Path(row['trial'])
        result = read(trial / 'result.json')
        stages = {}
        for phase in ('observe', 'pregrasp', 'approach'):
            proof = trial / f'route_mpc_{phase}.json'
            if proof.exists():
                stages[phase] = read(proof)
        old = prev.get(row['canonical_case_id'])
        row.update(route_mpc=stages, trial_wall_seconds=result['wall_seconds'],
                   route_mpc_solves=sum(x['mpc_solves'] for x in stages.values()),
                   route_mpc_round_trip_seconds=sum(x['round_trip_seconds'] for x in stages.values()),
                   local_mpc_reference={'success': old['success'], 'reason': old['reason'],
                                        'trial': old['trial']} if old else None)
    save(out / 'route_mpc_results.json', summary)


def configure(reg):
    bench.SELF = SELF
    base = bench.configure(reg)
    base.report = summarize
    verify = base.verify

    def verify_routes(out):
        checks = {phase: 0 for phase in route.ROUTE_NAMES}
        for trial in (out / bench.ARM / 'trials').iterdir():
            result = read(trial / 'result.json')
            assert not (trial / 'approach_plan_result.json').exists(), 'Unexpected approach replan'
            for phase in route.ROUTE_NAMES:
                file = trial / f'route_mpc_{phase}.json'
                if not file.exists():
                    continue
                proof = read(file)
                if proof['settled']:
                    assert proof['mpc_solves'] > 0
                    assert proof['commands'] == proof['mpc_solves'] * route.mpc.STEPS_PER_SOLVE
                    assert proof['joint_error_rad'] < route.SETTLED_JOINT_ERROR_RAD
                    checks[phase] += 1
            if result['success']:
                assert all(read(trial / f'route_mpc_{phase}.json')['settled'] for phase in route.ROUTE_NAMES)
                assert read(trial / 'mpc_approach.json')['global_prefix_ticks'] == 0
        verify(out)
        proof = read(out / 'final_verification.json')
        proof.update(verified_mpc_route_stages=checks, approach_replans=0)
        save(out / 'final_verification.json', proof)

    base.verify = verify_routes
    return base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'run', 'execute', 'serve', 'status'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--entry', type=Path)
    parser.add_argument('--backend-gpu')
    parser.add_argument('--scene-gpu')
    parser.add_argument('--case', action='append')
    args, remaining = parser.parse_known_args()
    if args.stage == 'serve':
        if remaining[remaining.index('--kind') + 1] == 'curobo':
            bench.relaxed.runtime.load_runtime().policy().install_collision_policy()
            bench.relaxed.install_trace()
            route.guarded.install_variant()
            route.install_resident()
        from short_lift_services import main as serve

        sys.argv = [str(SELF), *remaining]
        serve()
        return
    assert not remaining and args.output is not None
    out = args.output.resolve()
    if args.stage == 'prepare':
        assert args.backend_gpu and args.scene_gpu
        prepare(out, args.backend_gpu, args.scene_gpu, args.case or bench.PILOT)
    elif args.stage == 'status':
        summarize(out)
    elif args.stage == 'run':
        configure(read(out / 'registration.json')).run(out)
    else:
        base = configure(read(read(args.entry)['campaign_registration']))
        run, baseline = base.core.batch.original_modules()
        base.lift_policy.install_execution_policy(run, baseline)
        route.guarded.install_variant()
        with route.execution_hook(run):
            base.execute(out, args.entry)


if __name__ == '__main__':
    main()
