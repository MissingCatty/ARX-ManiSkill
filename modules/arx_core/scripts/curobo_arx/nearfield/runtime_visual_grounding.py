"""Same physical executor, with explicit control-applicability termination."""
import argparse
from pathlib import Path
from scripts.curobo_arx.data_factory.common import read
from training.nearfield_arx.learned_policy import read_bound_reply
from . import runtime_closedloop30,runtime_learned,closedloop30_rules
from .attribution_spec import fixed_entry_candidates
from .visual_grounding_spec import validate_trials,verify_reply


class VisualEnvironment(runtime_closedloop30.ClosedLoopEnvironment):
    def consume_prediction(self,snapshot):
        pending=self.pending_policy;request=pending['request'];path=pending['path']
        reply_path=path.with_name(request['request_id']+'_reply.json')
        if reply_path.exists():
            _,reply=read_bound_reply(path,reply_path);verify_reply(self.group.registration,request,reply)
            if not reply['visual_control']['applicable']:
                self.pending_policy=None
                self.log_event({'kind':'visual_control_not_applicable','policy_request_id':request['request_id'],
                                'alignment':reply['visual_control']['alignment']})
                return self.finish('visual_control_pose_pair_not_comparable')
        return super().consume_prediction(snapshot)


def install(campaign):
    reg=read(Path(campaign)/'registration.json');validate_trials(reg['seed_trials'])
    runtime_closedloop30.install(campaign);closedloop30_rules.entry_candidates=fixed_entry_candidates
    runtime_learned.LearnedEnvironment=VisualEnvironment


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);p.add_argument('--launch',required=True)
    a=p.parse_args();install(a.campaign);runtime_learned.LearnedGroup(a.campaign,a.launch).run()
