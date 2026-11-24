"""ARX mailbox audit selection; actual Isaac execution is checked separately."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from client import Client


ARX_JOINTS=[f'right_joint{i}' for i in range(21,27)]
ARX_POSITION=[.8932561,1.6205081,1.7437296,-1.0992556,1.1164106,.8189713]


class AuditMode(unittest.TestCase):
    def dispatch(self,flag):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'ready.json').write_text('{}')
            environment={'ARX_CUROBO_VALIDATION_DIR':str(root),
                         'ARX_RESEARCH_GUARD_FILE':str(root/'guard.json')}
            if flag is not None:environment['ARX_RESEARCH_AUDIT']=flag
            result={'settled':True,'contacts':[],
                    'records':[{'position':ARX_POSITION}],
                    'joint_names':ARX_JOINTS}
            requests=[];replace=Path.replace
            def reply(source,destination):
                value=replace(source,destination)
                if Path(destination).name=='request.json':
                    request=json.loads(Path(destination).read_text());requests.append(request)
                    (root/'response.json').write_text(json.dumps(
                        {'id':request['id'],'ok':True,'result':result}))
                    Path(destination).unlink()
                return value
            def audit(*_):
                (root/'guard_checks').mkdir(exist_ok=True)
            with patch.dict(os.environ,environment,clear=True), \
                 patch.object(Path,'replace',reply), \
                 patch('research_guard.check',side_effect=audit) as checks:
                returned=Client(root).request('trajectory',joint_names=ARX_JOINTS,
                    position=[ARX_POSITION],time_from_start=[0.])
                self.assertEqual(returned,result)
                self.assertEqual(len(requests),1)
                self.assertEqual(requests[0]['joint_names'],ARX_JOINTS)
                self.assertEqual(requests[0]['position'],[ARX_POSITION])
                if flag=='0':
                    checks.assert_not_called()
                    self.assertFalse((root/'guard_checks').exists())
                else:
                    self.assertEqual(checks.call_count,2)
                    self.assertEqual([call.args[-1] for call in checks.call_args_list],
                                     ['before_trajectory','after_trajectory'])
                    saved=root/'guard_checks'/(requests[0]['id']+'_execution.json')
                    self.assertEqual(json.loads(saved.read_text()),result)

    def test_normal_mode_preserves_arx_command_and_result_without_audit(self):
        self.dispatch('0')

    def test_explicit_research_mode_keeps_both_checks(self):
        self.dispatch('1')

    def test_legacy_explicit_guard_keeps_research_behavior(self):
        self.dispatch(None)

    def test_normal_mode_keeps_configuration_writes_forbidden(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ,
                {'ARX_CUROBO_VALIDATION_DIR':directory,
                 'ARX_RESEARCH_GUARD_FILE':directory+'/guard.json',
                 'ARX_RESEARCH_AUDIT':'0'},clear=True):
            for action in ('grasp_arm_gains','grasp_solver_iterations','obstacle'):
                with self.assertRaisesRegex(RuntimeError,'forbidden'):
                    Client(directory).request(action)
            self.assertFalse((Path(directory)/'request.json').exists())

    def test_audit_rejection_prevents_motion_dispatch(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ,
                {'ARX_CUROBO_VALIDATION_DIR':directory,
                 'ARX_RESEARCH_GUARD_FILE':directory+'/guard.json',
                 'ARX_RESEARCH_AUDIT':'1'},clear=True):
            root=Path(directory);(root/'ready.json').write_text('{}')
            with patch('research_guard.check',side_effect=RuntimeError('ARX configuration changed')):
                with self.assertRaisesRegex(RuntimeError,'configuration changed'):
                    Client(root).request('trajectory',joint_names=ARX_JOINTS)
            self.assertFalse((root/'request.json').exists())

    def test_invalid_audit_setting_fails_before_dispatch(self):
        with patch.dict(os.environ,{'ARX_RESEARCH_AUDIT':'invalid'},clear=True):
            with self.assertRaisesRegex(RuntimeError,'must be 0 or 1'):
                Client('/unused').request('trajectory',joint_names=ARX_JOINTS)


if __name__=='__main__':unittest.main()
