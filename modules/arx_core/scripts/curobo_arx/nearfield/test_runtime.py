import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from .runtime import audit_actions,skip_arrival,LocalEnvironment
from .campaign import startup_expired


class FailureGates(unittest.TestCase):
    def test_feedback_diagnostic_uses_source_waypoint_not_integrated_oracle_token(self):
        from .diagnose import reference_step
        from training.nearfield_arx.contract import encode
        source=np.eye(4);waypoint=source.copy();waypoint[0,3]=.012
        actual=source.copy();actual[0,3]=.010
        trial={'C_waypoints':[waypoint.tolist()],'C_tokens':[encode('C',actual,waypoint)]}
        np.testing.assert_allclose(reference_step(trial,0,source),waypoint)
        legacy={'C_tokens':[encode('C',source,waypoint)]}
        np.testing.assert_allclose(reference_step(legacy,0,source),waypoint)

    def test_C_oracle_uses_current_state_and_only_final_closed_goal(self):
        from types import SimpleNamespace
        from training.nearfield_arx.contract import encode
        a=np.eye(4);b=a.copy();b[0,3]=.012;c=b.copy();c[0,3]=.024
        actual=a.copy();actual[0,3]=.010
        env=LocalEnvironment.__new__(LocalEnvironment);env.local_index=0
        env.bridge=SimpleNamespace(snapshot=lambda:{'link_world':{'right_eef_link':actual}})
        env.group=SimpleNamespace(registration={'C_truth_driver':'current_state_to_recorded_next_waypoint','C_intermediate_phase':'local'},
            active_trial={'method':'C','source_Gstar_world':c.tolist(),'C_tokens':[encode('C',a,b),encode('C',b,c)],
                          'C_waypoints':[b.tolist(),c.tolist()]})
        calls=[];env.log_event=lambda event:None;env.submit=lambda label,goal,phase:calls.append((label,goal,phase))
        env.decide();np.testing.assert_allclose(calls[0][1],b);self.assertEqual(calls[0][2],'local')
        actual[:]=b;env.decide();np.testing.assert_allclose(calls[1][1],c);self.assertEqual(calls[1][2],'approach')

    def test_ready_scene_qualification_does_not_consume_startup_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            self.assertTrue(startup_expired(p,601))
            (p/'initialization.json').write_text(json.dumps({'phase':'ready_for_uniform_warmup'}))
            self.assertFalse(startup_expired(p,1000))
            (p/'initialization.json').write_text(json.dumps({'phase':'starting_simulation_app'}))
            self.assertTrue(startup_expired(p,601))

    def test_executed_missing_command_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);(p/'close_execution.json').write_text('{}')
            with self.assertRaises(RuntimeError):audit_actions(p,[{'label':'close'}])

    def test_orphan_execution_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);(p/'close_execution.json').write_text('{}')
            with self.assertRaises(RuntimeError):audit_actions(p,[])

    def test_strict_settled_flag_cannot_skip_different_goal(self):
        a=np.eye(4);b=a.copy();b[0,3]=.02
        self.assertFalse(skip_arrival({'settled':True},a,b,{},None)['accepted'])

    def test_arrival_requires_fresh_unchanged_state(self):
        a=np.eye(4).tolist();records=[{'time':t/10,'position':[0.],'target':[0.],'ee_world':a} for t in range(11)]
        execution={'settled':True,'records':records,'final':{'sim_time':1.}}
        command={'joint_names':['arm'],'position':[[0.],[0.]],'position_tolerance_rad':.005}
        state={'sim_time':1.2,'joint_names':['arm'],'joint_positions':[0.],'link_world':{'right_eef_link':a}}
        self.assertTrue(skip_arrival(execution,a,a,state,command)['accepted'])
        state['joint_positions']=[.02]
        self.assertFalse(skip_arrival(execution,a,a,state,command)['accepted'])


if __name__=='__main__':unittest.main()
