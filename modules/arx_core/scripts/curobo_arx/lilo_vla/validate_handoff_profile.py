"""Offline admission checks using real qualified ARX entries and copied mutations.

Mutations only exercise rejection paths in memory; they are not demonstrations
or physical trials, and never change simulator/robot configuration.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from handoff_profile import admit,read


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();profile=read(args.profile)
    health=read(Path(profile['qualification_campaign'])/'policy_health.json')
    health['policy_artifact_sha256']=profile['policy_artifact_sha256']
    positive=[];sample=None
    for case in profile['cases'].values():
        for folder in case['qualification_trials']:
            trial=Path(folder)
            value={'health':health,'scenario':read(trial/'scenario.json'),'selected':read(trial/'selected_grasp.json'),
                'observation':read(trial/'handoff_actual_observation/observation.json'),
                'certificate':read(trial/'handoff_actual_entry.json'),'setup':read(trial/'setup.json'),
                'instruction':read(trial/'grasp_report.json')['instruction']}
            receipt=admit(profile,**value);positive.append({'trial':str(trial),'receipt':receipt})
            if sample is None:sample=value
    def change_goal(x):x['selected']['grasp_world'][0][3]+=.001
    def change_scene(x):x['scenario']['placement']['position'][0]+=.01
    def change_head(x):x['health']['policy_artifact_sha256']['action_head.pt']='0'*64
    def change_state(x):x['health']['state_convention']='object_relative_v1'
    def change_rule(x):x['selected']['handoff_rule']='grasp_backoff_v1'
    def change_instruction(x):x['instruction']='place the object'
    def stale(x):x['observation']['frame_time']-=.02
    def wrong_pose(x):x['observation']['snapshot']['link_world']['right_eef_link'][0][3]+=.01
    def closed_gripper(x):
        snapshot=x['observation']['snapshot'];snapshot['joint_positions'][snapshot['joint_names'].index('right_joint27')]-=.01
    def contact(x):x['certificate']['pregrasp_target_contact_events']=1
    def hidden(x):x['certificate']['frustum']['pass']=False
    def invalid_continuation(x):x['certificate']['continuation']['pass']=False
    def wrong_reset(x):x['setup']['initial_conditions']['joint_positions'][0]+=.01
    def wrong_renderer(x):x['setup']['fingerprint']['invariant']['configured']['simulation']['renderer']='offline_mutation'
    negative=[]
    for name,mutate in [('different_goal',change_goal),('different_scenario',change_scene),('different_weights',change_head),
        ('wrong_state_version',change_state),('different_entry_rule',change_rule),('different_skill_instruction',change_instruction),('stale_observation',stale),
        ('outside_pose_neighborhood',wrong_pose),('wrong_gripper_phase',closed_gripper),('target_precontact',contact),
        ('outside_wrist_view',hidden),('failed_continuation',invalid_continuation),('different_robot_reset',wrong_reset),
        ('different_renderer',wrong_renderer)]:
        altered=copy.deepcopy(sample);mutate(altered)
        try:admit(profile,**altered)
        except ValueError as error:negative.append({'guard':name,'rejected':True,'reason':str(error)})
        else:raise AssertionError('Admission guard accepted '+name)
    result={'pass':True,'recorded_actual_entries':positive,'copied_negative_mutations':negative,
        'profile_sha256':hashlib.sha256(args.profile.read_bytes()).hexdigest(),
        'scope':'Offline protocol checks, including registered artifact identity for recorded entries; not new physical success evidence'}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'recorded_entries':len(positive),'negative_mutations_rejected':len(negative),'pass':True}))


if __name__=='__main__':main()
