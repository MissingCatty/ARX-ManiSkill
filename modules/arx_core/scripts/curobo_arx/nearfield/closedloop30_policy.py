"""Both advisor27 checkpoints, original prompts, unchanged bounded decoder."""
import argparse
from pathlib import Path
from scripts.curobo_arx.data_factory.common import read, sha, verify_files
from training.nearfield_arx import stage_policy, lateral_recovery


def validate_checkpoints(reg):
    for method in ('C', 'D'):
        binding = reg['learned_policy']['training_bindings'][method]
        folder = Path(reg['learned_policy']['checkpoints'][method])
        if sha(binding['registration']) != binding['sha256']: raise ValueError('Training binding changed')
        contract = read(folder/'arx_policy_contract.json')
        if (contract['method'] != method or contract['experiment'] != lateral_recovery.VERSION or
            contract['registration_sha256'] != binding['sha256']):
            raise ValueError('Wrong unweighted nearfield checkpoint')
        verify_files({str(folder/n): h for n, h in read(folder/'checkpoint_receipt.json')['files'].items()})


def run(campaign):
    stage_policy.validate_checkpoints = validate_checkpoints
    return stage_policy.run(campaign)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('campaign'); run(p.parse_args().campaign)
