"""Compare completed ARX slots separately for each candidate protocol."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import graspcl_compare as compare


def save(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def report(root, cache):
    # Suppress the legacy aggregate write; never present mixed protocols as the
    # performance of the new candidate-generation method.
    compare.save = lambda path, data: None
    result = compare.report(root, cache)
    registration = compare.read(root / 'registration.json')
    current = set(registration['current_protocol_slot_ids'])
    phases = {}
    for label, is_current in [('prior_raw_custom_geometry', False), (registration['candidate_protocol'], True)]:
        rows = [r for r in result['results'] if (r['id'] in current) == is_current]
        phases[label] = {'registered': len(current) if is_current else 200-len(current),
            'overall': compare.tally(rows),
            'by_layout': {layout: compare.tally([r for r in rows if r['layout'] == layout])
                          for layout in registration['layouts']}, 'results': rows}
    output = {'updated_at': datetime.now().astimezone().isoformat(), 'phase': result['campaign_phase'],
        'original_cumulative_budget': 200, 'current_protocol': registration['candidate_protocol'],
        'scope': 'Report prior and current protocols separately. Current cohort comprises only previously unstarted slots, not 200 native-post tests. Same historical scenes, fixed ARX model and success gates.',
        'by_protocol': phases, 'pending_claims': result['pending_claims']}
    save(root / 'comparison_by_protocol.json', output)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    cache = {}
    while True:
        result = report(args.campaign.resolve(), cache)
        if not args.watch or result['phase'] in ['complete', 'stopped'] or (args.campaign / 'halt.json').exists():
            break
        time.sleep(20)
