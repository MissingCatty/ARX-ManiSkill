import json
from pathlib import Path
import tempfile
import unittest

from .artifacts import ReceiptWriter


class ReceiptTests(unittest.TestCase):
    def test_ordered_overwrites_are_complete_before_return(self):
        with tempfile.TemporaryDirectory() as temp:
            writer = ReceiptWriter()
            writer.submit(temp, {'state.json': {'width': .044}})
            writer.submit(temp, {'state.json': {'width': .021}})
            report = writer.finish()
            self.assertEqual(json.loads((Path(temp) / 'state.json').read_text()), {'width': .021})
            self.assertEqual(report['packets'], 2)

    def test_disk_failure_reaches_the_request(self):
        def fail(*args):
            raise OSError('injected write failure')
        writer = ReceiptWriter(fail)
        writer.submit('/unused', {'receipt.json': {'passed': True}})
        with self.assertRaisesRegex(OSError, 'injected'):
            writer.finish()
        self.assertTrue(writer.closed)

    def test_completed_writer_cannot_accept_more_receipts(self):
        writer = ReceiptWriter()
        writer.finish()
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            writer.submit('/unused', {})


if __name__ == '__main__':
    unittest.main()
