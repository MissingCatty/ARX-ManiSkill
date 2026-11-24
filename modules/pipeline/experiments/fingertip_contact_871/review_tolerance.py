"""Review retained collision summaries and select the local fingertip allowance."""
import ast
import csv
import math
from pathlib import Path
import re

from arx_project.layout import ROOT, now, read_json, sha256, write_json
from modules.curobo.fingertip_contact import load_profile


def main():
    output = ROOT/'tmp/fingertip_contact/tolerance_review_20261124_01'
    output.mkdir(parents=True, exist_ok=True)
    historical = ROOT/'modules/pipeline/experiments/fingertip_table_tolerance'
    report_file = historical/'write_report.py'
    source = report_file.read_text()
    tree = ast.parse(source)
    labels = {
        '仅给指尖球模型0.5毫米余量': ('sphere', 0.5),
        '仅给指尖球模型1毫米余量': ('sphere', 1.0),
        '仅给指尖球模型2毫米余量': ('sphere', 2.0),
        '真实夹指网格复核，零穿入': ('convex_mesh', 0.0),
        '真实夹指网格复核，加2毫米指尖余量': ('convex_mesh', 2.0),
    }
    rows = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        cells = [part.strip() for part in node.value.split('|')]
        if len(cells) != 5 or cells[1] not in labels:
            continue
        model, tolerance = labels[cells[1]]
        counts = [re.match(r'(\d+)/(\d+)', cell) for cell in cells[2:4]]
        if any(count is None for count in counts):
            raise ValueError('Unrecognized retained historical summary')
        rows.append({'label': cells[1], 'collision_representation': model,
            'tolerance_mm': tolerance, 'original_recovered': int(counts[0][1]),
            'original_positions': int(counts[0][2]), 'near_recovered': int(counts[1][1]),
            'near_positions': int(counts[1][2]), 'source_line': node.lineno})
    if len(rows) != len(labels) or any(row['original_positions'] != 53 or row['near_positions'] != 14 for row in rows):
        raise ValueError('Historical cohort coverage differs from the retained report')

    known = re.search(r'香皂的真实指尖最低点低于名义桌面约([0-9.]+)毫米', source)
    if known is None:
        raise ValueError('Known real-mesh contact example is missing')
    known_depth = float(known[1])
    recommended = math.ceil(known_depth/0.5)*0.5
    profile = load_profile()
    if recommended > 2.0 or profile.maximum_penetration_m*1000 != recommended:
        raise ValueError('Current contact policy differs from the supported selection')

    cleanup = ROOT/'tmp/arx_configuration/records/cleanup_20261124_01/manifest.json'
    cleanup_data = read_json(cleanup)
    removed = [row for row in cleanup_data['entries']
               if 'fingertip_table_tolerance/runs/' in row['path'] or 'full_failure_distance_probe/runs/' in row['path']]
    protected = [report_file, historical/'AGENTS.md', cleanup,
        ROOT/'modules/pipeline/docs/mainline_versions/debug/fingertip_table_tolerance/假设.md',
        ROOT/'modules/pipeline/docs/mainline_versions/debug/fingertip_table_tolerance/定位记录.md',
        ROOT/'modules/maniskill/arx_pipeline/planner.py',
        ROOT/'configs/policies/fingertip_table.json']
    result = {'at': now(), 'sources': {str(p.relative_to(ROOT)): sha256(p) for p in protected},
        'evidence_type': 'Retained historical report constants and diagnostic notes; no recomputation of deleted runs',
        'hard_collision_tolerance_mm': 0., 'optimizer_activation_distance_mm': 3.,
        'historical_rows': rows, 'known_real_mesh_example': {'object': 'bar_soap', 'depth_mm': known_depth},
        'historical_raw_records_available': False,
        'deleted_run_entries': removed,
        'scene_minimum_depth_distribution_available': False,
        'historical_collision_model': 'Legacy collision configuration, predating the current 871-sphere profile',
        'cohort_definition': '53 original planning-failure positions; 14 relocated positions are repeated scenes',
        'mesh_coverage': '59 deterministic representative joint solutions; confirmed recovered positions',
        'selected_nominal_mesh_tolerance_mm': recommended,
        'selection_rule': 'Round the retained 1.818mm real-mesh witness upward to a 0.5mm engineering setting',
        'increment_above_known_example_mm': recommended-known_depth,
        'active_policy_id': profile.profile_id, 'active_policy_changed': False,
        'current_871_population_distribution_available': False,
        'reason_for_bound': 'A 2mm cap covers the retained real-tip witness; surviving results supply no basis for a larger allowance',
        'physical_grasp_attempts': 0}
    write_json(output/'review.json', result)
    with (output/'historical_thresholds.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print('Retained summary rows:', len(rows))
    print('Known real fingertip depth (mm):', known_depth)
    print('Selected nominal local mesh allowance (mm):', recommended)
    print('Complete historical depth distribution available:', False)


if __name__ == '__main__':
    main()
