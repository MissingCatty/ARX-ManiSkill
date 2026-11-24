"""Same advisor30 execution for all arms, with one fixed matched entrance."""
import argparse
from pathlib import Path
from scripts.curobo_arx.data_factory.common import read
from . import runtime_closedloop30, runtime_learned, closedloop30_rules
from .attribution_spec import validate_trials, fixed_entry_candidates, verify_generation


class AttributionEnvironment(runtime_closedloop30.ClosedLoopEnvironment):
    def consume_prediction(self, snapshot):
        pending=self.pending_policy;request=pending['request']
        path=pending['path'].with_name(request['request_id']+'_reply.json')
        if path.exists():verify_generation(self.group.registration,request,read(path))
        return super().consume_prediction(snapshot)


def install(campaign):
    reg = read(Path(campaign)/'registration.json'); validate_trials(reg['seed_trials'])
    runtime_closedloop30.install(campaign)
    closedloop30_rules.entry_candidates = fixed_entry_candidates
    runtime_learned.LearnedEnvironment = AttributionEnvironment


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);p.add_argument('--launch',required=True)
    a=p.parse_args();install(a.campaign);runtime_learned.LearnedGroup(a.campaign,a.launch).run()
