"""A second physical batch keeps the first batch's standalone web snapshots."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

from arx_project.layout import ROOT, read_json, write_json


class PublicationTests(unittest.TestCase):
    def test_new_batch_has_distinct_media_and_preserves_history(self):
        (ROOT/'tmp/tests').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='map_publication_', dir=ROOT/'tmp/tests') as folder:
            base = Path(folder)
            spec = importlib.util.spec_from_file_location('test_map_publisher', ROOT/'website/tools/publish_mobile_v4.py')
            publisher = importlib.util.module_from_spec(spec); spec.loader.exec_module(publisher)
            publisher.ROOT = base
            media = base/'website/resources/media'; media.mkdir(parents=True)
            source = base/'website/dist/iterations/sam3-graspgenx/versions/V4/index.html'
            source.parent.mkdir(parents=True); source.write_text('original complete page snapshot')
            old = media/'sam3-v4-summary.json'; old.write_text('original results')
            write_json(base/'website/内容/网站.json', {'versions': [{'id': 'sam3-graspgenx'}]})
            write_json(base/'website/内容/experiments.json', {'experiments': [{
                'id': 'sam3-graspgenx', 'iterations': [{'code': 'V4', 'batches': [{'id': 'old100'}]}]}]})
            (base/'website/内容/迭代').mkdir()
            write_json(base/'configs/experiments/active_experiment.json', {})
            dataset = base/'simulation/scenes/mobile100'; entries = []
            for layout in publisher.LAYOUTS:
                for i in range(20):
                    identity = f'{layout}_{i:03d}'
                    entries.append({'id': identity, 'layout': layout, 'target_text': 'target'})
                    image = dataset/'cases'/identity/'head.jpg'
                    image.parent.mkdir(parents=True); image.write_bytes(b'frozen reference image')
            write_json(dataset/'verification.json', {'passed': True})
            root = base/'tmp/mobile_dual_arm/new100'; (root/'trials').mkdir(parents=True)
            reg = {'variant': 'map', 'dataset': str(dataset), 'schedule': entries,
                'publication_prefix': 'new100', 'user_authorization': 'explicit new100',
                'predecessor': {'completed': 88, 'interrupted': 1, 'not_started': 11, 'successes': 48}}
            write_json(root/'registration.json', reg)
            progress = {'phase': 'registered', 'completed': 0, 'not_completed': 100,
                'station_ready': 0, 'planning_passes': 0, 'physical_attempts': 0, 'successes': 0,
                'not_started': 100, 'interrupted': 0, 'failure_reasons': {},
                'by_layout': {k: {'completed': 0, 'planning_passes': 0, 'successes': 0} for k in publisher.LAYOUTS}}
            write_json(root/'progress.json', progress); write_json(root/'summary.json', {**progress, 'results': []})
            publisher.publish(root)
            source.write_text('later active page')
            publisher.publish(root)
            self.assertEqual(old.read_text(), 'original results')
            self.assertEqual((media/'sam3-v4-initial100-report.html').read_text(), 'original complete page snapshot')
            self.assertTrue((media/'new100-summary.json').is_file())
            self.assertFalse((media/'new100-summary.json').is_symlink())
            versions = read_json(base/'website/内容/experiments.json')
            self.assertEqual([b['id'] for b in versions['experiments'][0]['iterations'][0]['batches']], ['old100', 'new100'])
            page = (base/'website/内容/迭代/sam3-graspgenx-v4.md').read_text()
            self.assertIn('两轮批次记录', page)
            self.assertIn('地图主动选站位', page)
            self.assertIn('/media/new100-registration.json', page)


if __name__ == '__main__':
    unittest.main()
