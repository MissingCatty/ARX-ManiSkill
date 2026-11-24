"""New registered admission rules around the unchanged learned ARX executor."""
import argparse
from scripts.curobo_arx.data_factory.common import sha, digest, write
from training.nearfield_arx.entry_guard import DSession, decode, parse_action, validate_receipt
from training.nearfield_arx.learned_policy import read_bound_reply, dispatch_drift
from . import runtime_learned, runtime_v2


class GuardedEnvironment(runtime_learned.LearnedEnvironment):
    def consume_prediction(self, snapshot):
        pending = self.pending_policy; path = pending['path']; request = pending['request']
        reply_path = path.with_name(request['request_id'] + '_reply.json')
        if reply_path.exists():
            _, reply = read_bound_reply(path, reply_path)
            proof = validate_receipt(path, request, reply,
                self.group.registration['learned_policy']['visual_contract']['depth_scale']['near_m'])
            token = reply['prediction'].strip()
            try:
                parts = parse_action(request['method'], token)
            except ValueError:
                return super().consume_prediction(snapshot)
            if request['method'] == 'C' and parts is not None:
                goal = decode('C', request['inputs']['eef_world'], token)
                if proof.get('goal_sha256') != digest(goal.tolist()):
                    raise RuntimeError('Depth-admission receipt refers to another C target')
                if (not proof['pass'] and dispatch_drift(pending['snapshot'], snapshot)['pass']
                        and reply['projection']['target_visibility_pass']):
                    write(path.with_name(request['request_id'] + '_dispatch_check.json'),
                          dispatch_drift(pending['snapshot'], snapshot))
                    self.pending_policy = None
                    self.log_event({'kind': 'policy_action_guard_rejected',
                        'policy_request_id': request['request_id'], 'raw_prediction': reply['prediction'],
                        'reason': proof['reason'], 'action_admission': proof})
                    return self.finish('C_repeated_divergence_depth_guard_rejected')
        return super().consume_prediction(snapshot)


def install():
    runtime_learned.parse_action = parse_action
    runtime_learned.decode = decode
    runtime_v2.DSession = DSession
    runtime_learned.LearnedEnvironment = GuardedEnvironment


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--campaign', required=True); p.add_argument('--launch', required=True)
    args = p.parse_args(); install()
    runtime_learned.LearnedGroup(args.campaign, args.launch).run()
