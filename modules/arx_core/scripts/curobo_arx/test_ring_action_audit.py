"""ARX-archive integrity checks; not physical validation of ring preshaping."""
import json,shutil,tempfile,unittest
from pathlib import Path
import numpy as np

from audit_ring_actions import audit


class RingActionAuditTests(unittest.TestCase):
    def setUp(self):
        self.source=Path(__file__).resolve().parents[2]/'runs/autoresearch_arx_20261019_round8/trials/02_mustard_bottle_feedback'
        self.temp=tempfile.TemporaryDirectory();self.folder=Path(self.temp.name)
        for name in ('capture_snapshot.json','initial_fingerprint.json','selected_grasp.json','pregrasp_execution.json','approach_execution.json'):
            shutil.copy2(self.source/name,self.folder/name)

    def tearDown(self):self.temp.cleanup()

    def write(self,name,data):(self.folder/name).write_text(json.dumps(data))

    def fake_preshape_artifacts(self):
        # Deliberately reuse a MOVING real ARX approach as adversarial evidence;
        # these artifacts must never be mistaken for an executed ring preshape.
        execution=json.loads((self.source/'pregrasp_execution.json').read_text());q=np.asarray(execution['records'][-1]['target'],dtype=np.float32).tolist()
        cfg=json.loads((self.source/'initial_fingerprint.json').read_text())['invariant']['configured']['controllers']['gripper']
        api=cfg.get('api_closed',0.)+(.03-cfg['closed_m'])/(cfg['open_m']-cfg['closed_m'])*(cfg.get('api_open',-3.2)-cfg.get('api_closed',0.))
        command={'joint_names':json.loads((self.source/'capture_snapshot.json').read_text())['right_arm'],'position':[q,q],'time_from_start':[0.,3.],'gripper_target':api}
        self.write('ring_preshape_command.json',command);self.write('ring_preshape_execution.json',execution);self.write('ring_preshape_report.json',{'accepted':True});self.write('ring_policy.json',{'applied':True})
        self.write('approach_command.json',command)
        return command,execution

    def test_real_nonsphere_fallback_requires_no_preshape(self):
        self.write('ring_policy.json',{'applied':False})
        self.assertTrue(audit(self.folder)['pass'])

    def test_ring_approach_without_preshape_evidence_is_rejected(self):
        self.write('ring_policy.json',{'applied':True})
        self.assertFalse(audit(self.folder)['guard_pass'])

    def test_moving_record_cannot_masquerade_as_stationary_preshape(self):
        self.fake_preshape_artifacts();result=audit(self.folder)
        self.assertIn('Recorded preshape targets do not match constant commanded arm target',result['guard_violations'])

    def test_wrong_opening_mapping_is_rejected(self):
        command,_=self.fake_preshape_artifacts();command['gripper_target']=-3.2;self.write('ring_preshape_command.json',command)
        self.assertIn('Preshape command does not use unchanged API mapping',audit(self.folder)['guard_violations'])

    def test_arm_limit_violation_is_rejected(self):
        command,_=self.fake_preshape_artifacts();command['position'][0][0]=99.;command['position'][1][0]=99.;self.write('ring_preshape_command.json',command)
        self.assertIn('Preshape arm command exceeds frozen limits',audit(self.folder)['guard_violations'])


if __name__=='__main__':unittest.main()
