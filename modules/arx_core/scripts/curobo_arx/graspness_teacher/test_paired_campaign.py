import json
from pathlib import Path
import tempfile
import unittest
from scripts.curobo_arx.data_factory.ledger import Ledger
from .paired_campaign import prepare,summarize


class PhysicalActivationTests(unittest.TestCase):
    def test_failed_or_unfinished_model_screen_does_not_prepare_physics(self):
        for status in ['running','complete']:
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder);(root/'M03').mkdir()
                (root/'registration.json').write_text('{}')
                (root/'M03/results.json').write_text(json.dumps({'status':status,'rows':[
                    {'object_id':'butter','mode':'baseline','qualified':0,'split':'development'},
                    {'object_id':'butter','mode':'variant','qualified':0,'split':'development'}]}))
                with self.assertRaises(ValueError):prepare(root,'M03')
                self.assertFalse((root/'physical_M03').exists())

    def summary_fixture(self,root):
        trials=[{'trial_id':'original','object_id':'butter','policy':'baseline'},
                {'trial_id':'alternative','object_id':'butter','policy':'variant'}]
        (root/'registration.json').write_text(json.dumps({'paired_trials':trials}))
        (root/'trial_ledgers').mkdir()
        for trial in trials:Ledger(root/'trial_ledgers'/(trial['trial_id']+'.sqlite3')).close()
        return trials

    def test_no_plan_is_failure_while_never_started_is_unexecuted(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);trials=self.summary_fixture(root);group=root/'groups/ARX_env';group.mkdir(parents=True)
            (group/'paired_results.json').write_text(json.dumps({'trials':[{**trials[0],
                'success':False,'admitted':False,'physical_dispatches':0,'outcome':'candidate_exhausted'}]}))
            result=summarize(root)
            self.assertEqual(result['completed_trials'],1);self.assertEqual(result['physical_dispatches'],0)
            self.assertEqual(result['unexecuted'],['alternative']);self.assertEqual(result['fault_or_unknown'],[])
            self.assertEqual(result['objects'][0]['baseline']['failure_reasons'],{'candidate_exhausted':1})

    def test_started_without_terminal_result_is_unknown_and_not_unexecuted(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.summary_fixture(root)
            (root/'groups/ARX_env/trials/original').mkdir(parents=True)
            result=summarize(root)
            self.assertEqual(result['fault_or_unknown'],['original'])
            self.assertEqual(result['unexecuted'],['alternative'])


if __name__=='__main__':unittest.main()
