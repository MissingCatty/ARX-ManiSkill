"""Add the verified original training plan to a new, immutable warmup snapshot."""
import argparse
from pathlib import Path
import shutil

import arx_direct_pilot_20261120 as pilot

SELF = Path(__file__).resolve()
SOURCE = pilot.CODE/'outputs/arx_zerograsp_graspcl_physical_20261117_01/trials/B0069_S3_sparse_014_lemon'


def prepare(root):
    # Check before any model is launched. The earlier prepared warmup copied
    # state/images only; use its exact original development execution record.
    old = pilot.common.read(pilot.PRIOR/'B_graspcl_no_experience/registration.json')
    warm = Path(old['warmup_trial'])
    assert (SOURCE/'plan_result.json').is_file()
    assert pilot.common.read(SOURCE/'plan_result.json')['success']
    assert (SOURCE/'state.json').read_bytes() == (warm/'state.json').read_bytes()
    pilot.prepare(root)
    copied = root/'warmup'/warm.name
    shutil.copytree(warm, copied)
    shutil.copy2(SOURCE/'plan_result.json', copied/'plan_result.json')
    hashes = {str(p): pilot.common.sha(p) for p in copied.rglob('*') if p.is_file()}
    hashes.update({str(SELF): pilot.common.sha(SELF), str(SOURCE/'plan_result.json'): pilot.common.sha(SOURCE/'plan_result.json')})
    for directory in [root, *[root/n for n in pilot.GROUPS], *[root/n/'fixed_graspcl' for n in pilot.GROUPS]]:
        path = directory/'registration.json';reg = pilot.common.read(path)
        reg['source_hashes'].update(hashes)
        reg.update(warmup_trial=str(copied), actual_run_entry=str(SELF),
                   training_plan_source=str(SOURCE/'plan_result.json'),
                   training_state_matches_original_source=True,
                   prior_prestart_output=str(pilot.CODE/'outputs/arx_direct_grasp_pilot_20261120_01'),
                   prior_prestart_actual_claims=0)
        pilot.common.save(path, reg)
    receipt = pilot.common.read(root/pilot.EPOCH/'registration.json')
    receipt.update(root_registration_sha256=pilot.common.sha(root/'registration.json'),
                   warmup_plan_validated_before_loading=True)
    pilot.common.save(root/pilot.EPOCH/'registration.json', receipt)
    shutil.copy2(SELF, root/'source'/SELF.name)
    pilot.guard = pilot.parent.guard
    pilot.parent.guard(pilot.common.read(root/'registration.json'))
    print('READY PREPARED', root, 'training state matches exact original source; 0 previous test claims', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'run'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.stage == 'prepare' else pilot.run)(args.output.resolve())
