"""Offline selection checks: scheduling exclusion must not alter frozen physics."""
import copy
from pathlib import Path
import tempfile
import unittest

from .catalog import build, select_collection, collection_input
from .ledger import Ledger
from .common import write


class CollectionTests(unittest.TestCase):
    def fixture(self):
        return {'objects':[
            {'object_id':'egad_A1','available':True,'placement':{'position':[4.,2.,.7]},'mass_kg':.1},
            {'object_id':'mustard_bottle','available':True,'placement':{'position':[4.,2.,.8]},'mass_kg':.3},
            {'object_id':'renamed_mesh','source':'/asset/egad49_arx/model.usd','available':True},
            {'object_id':'lizard_figurine','available':True},
        ],'initial_count':4}

    def test_selection_preserves_all_scene_records_and_the_input(self):
        source=self.fixture();original=copy.deepcopy(source)
        result=select_collection(source,'common_household')
        self.assertEqual(source,original)
        self.assertEqual(result['initial_count'],source['initial_count'])
        self.assertEqual(len(result['objects']),len(source['objects']))
        for before,after in zip(source['objects'],result['objects']):
            self.assertEqual({k:after[k] for k in before},before)
        self.assertEqual(result['collection_selection']['selected_object_ids'],['mustard_bottle'])
        self.assertTrue(result['objects'][0]['available'])

    def test_scheduler_never_falls_back_to_excluded_objects(self):
        catalog=select_collection(self.fixture(),'common_household')
        with tempfile.TemporaryDirectory() as tmp:
            ledger=Ledger(Path(tmp)/'ledger.sqlite3')
            try:
                ledger.initialize(catalog,{'max_physical_attempts':16,'max_total_dispatches':32,
                    'goals_per_object':2,'successes_per_bin':2,'distances_mm':[10,40,60,80]})
                self.assertEqual(ledger.acquire_object('group')['object_id'],'mustard_bottle')
                self.assertIsNone(ledger.available_object())
                with self.assertRaisesRegex(ValueError,'Collection disabled'):
                    ledger.claim('egad_A1','group','env')
                self.assertEqual(ledger.rows(),[])
            finally:ledger.close()

    def test_policy_rejects_empty_and_unknown_selections(self):
        with self.assertRaisesRegex(ValueError,'no available'):
            select_collection({'objects':[self.fixture()['objects'][0]]},'common_household')
        with self.assertRaisesRegex(ValueError,'Unknown collection'):
            select_collection(self.fixture(),'typo')

    def test_explicit_legacy_policy_does_not_rewrite_old_records(self):
        catalog=self.fixture()
        self.assertEqual(select_collection(catalog,'all'),catalog)

    def test_real_ARX_catalog_keeps_all_historical_objects(self):
        original=build();result=select_collection(original,'common_household')
        self.assertEqual(len(original['objects']),67)
        self.assertEqual(result['initial_count'],67)
        self.assertEqual(sum('egad' in x['object_id'].lower() for x in result['objects']),49)
        self.assertEqual(len(result['collection_selection']['selected_object_ids']),18)
        for x in result['objects']:
            if x['object_id'].startswith('egad_'):
                self.assertFalse(x['collection_enabled'])
                self.assertTrue(x['available'])

    def test_changed_policy_requires_a_new_collection_not_old_budget_lineage(self):
        from scripts.curobo_arx.sdg_stage1.prepare import prepare
        with tempfile.TemporaryDirectory() as tmp:
            parent=Path(tmp)/'old'
            write(parent/'registration.json',{'stage':'sdg_stage1','object_policy':'all'})
            with self.assertRaisesRegex(ValueError,'new collection'):
                prepare(Path(tmp)/'new',source_fix_parent=parent,object_policy='common_household')

    def test_reporting_keeps_excluded_objects_out_of_collection_denominator(self):
        from .report import report
        catalog=select_collection(self.fixture(),'common_household')
        for obj in catalog['objects']:
            obj.update(split='train',issues=[])
        quotas={'max_physical_attempts':16,'max_total_dispatches':32,'goals_per_object':2,
                'successes_per_bin':2,'distances_mm':[10,40,60,80],'target_episodes':16}
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            write(folder/'registration.json',{'catalog':catalog,'quotas':quotas})
            write(folder/'state.json',{'status':'prepared'})
            ledger=Ledger(folder/'ledger.sqlite3');ledger.initialize(catalog,quotas);ledger.close()
            result=report(folder)
            self.assertEqual(result['catalog_objects'],4)
            self.assertEqual(result['collection_objects'],1)
            self.assertFalse(result['quota_complete'])
            for obj in result['objects']:
                if obj['object_id']!='mustard_bottle':
                    self.assertIsNone(obj['quota_complete'])
                    self.assertFalse(obj['quota_applicable'])

    def test_default_household_pack_is_hash_bound_without_silent_fallback(self):
        from unittest.mock import patch
        from .common import sha
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);extra=root/'asset/catalog.json'
            write(extra,[])
            profile=root/'configs/lilo_vla/common_household_collection.json'
            write(profile,{'object_policy':'common_household','extra_catalog':'asset/catalog.json',
                           'extra_catalog_sha256':sha(extra)})
            with patch('scripts.curobo_arx.data_factory.catalog.ROOT',root):
                self.assertEqual(collection_input(None,'common_household'),(extra,profile))
                self.assertEqual(collection_input(None,'all'),(None,None))
                self.assertEqual(collection_input('explicit.json','common_household'),('explicit.json',None))
                write(extra,[{'unexpected':'changed catalog'}])
                with self.assertRaisesRegex(ValueError,'hash mismatch'):
                    collection_input(None,'common_household')


if __name__=='__main__':
    unittest.main()
