"""Only offline stop/claim gates for the resource continuation."""
from pathlib import Path
import unittest
from unittest.mock import patch
from scripts.curobo_arx.data_factory.common import ROOT,read
from .main48_resource_continue import unissued,ledger_rows


class ResourceContinuationTests(unittest.TestCase):
    root=ROOT/'runs/nearfield_main48_20261109_02'

    def test_actual_preparation_without_commands_is_unissued(self):
        proof=unissued(self.root)
        self.assertEqual(proof['new_physical_attempts'],0)
        self.assertEqual(proof['remaining_slots'],46)

    def test_any_claim_including_before_dispatch_blocks_replay(self):
        with patch('scripts.curobo_arx.nearfield.main48_resource_continue.ledger_rows',return_value=[{'physical':0,'state':'claimed'}]):
            with self.assertRaises(ValueError):unissued(self.root)

    def test_manual_stop_is_not_a_resource_restart(self):
        def changed(path):
            result=read(path)
            return {'reason':'user_requested_stop'} if Path(path).name=='STOP' else result
        with patch('scripts.curobo_arx.nearfield.main48_resource_continue.read',side_effect=changed):
            with self.assertRaises(ValueError):unissued(self.root)


if __name__=='__main__':unittest.main()
