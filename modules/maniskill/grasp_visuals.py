"""CPU-only visual evidence from recorded RGB-D and unchanged ZeroGrasp rows."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import time
import traceback

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, to_hex
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


RENDERER_SHA256 = sha(Path(__file__))
VISUALIZATION_VERSION = 'target_top10_rgbd_v1'


def save(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)


def gripper_lines(row):
    """GraspNet display convention: local X approach, Y opening, Z height.

    Centerlines follow graspnetAPI.plot_gripper_pro_max (4 mm rods,
    20 mm base depth, 40 mm tail). These are candidate glyphs, not ARX meshes.
    """
    width, depth = float(row[1]), float(row[3])
    half = width / 2 + .002
    local = np.array([
        [[-.022, -half, 0], [depth, -half, 0]],
        [[-.022, half, 0], [depth, half, 0]],
        [[-.022, -half, 0], [-.022, half, 0]],
        [[-.064, 0, 0], [-.022, 0, 0]],
    ])
    return local @ row[4:13].reshape(3, 3).T + row[13:16]


def figure_save(fig, path):
    tmp = path.with_suffix('.tmp.png')
    fig.savefig(tmp, dpi=160, facecolor='white')
    plt.close(fig)
    tmp.replace(path)


def render(trial, out):
    out.mkdir(parents=True, exist_ok=True)
    source = trial / 'raw_grasps.npy'
    ready = json.loads((trial / 'network_candidates.json').read_text())
    model_name = ready.get('model_name', 'ZeroGrasp')
    candidate_stage = ready.get('candidate_stage', 'SDK collision filtering + NMS' if ready.get('post_collision_and_nms') else 'raw network candidates')
    score_label = ready.get('score_label', 'Network score')
    rows = np.load(source, allow_pickle=False)
    if rows.ndim != 2 or rows.shape[1] != 17 or len(rows) != ready['count']:
        raise RuntimeError('Incomplete or unexpected ZeroGrasp row array')
    if not np.isfinite(rows).all():
        raise RuntimeError('Non-finite proposal values; do not silently filter the ranking')
    indices = np.argsort(-rows[:, 0], kind='stable')[:10]
    top = rows[indices]
    if not len(top):
        raise RuntimeError('No network proposals available')
    state = json.loads((trial / 'state.json').read_text())
    params = json.loads((trial / 'camera/params.json').read_text())
    intr = params['intrinsics']
    k = np.array([[intr['fx'], 0, intr['ppx']], [0, intr['fy'], intr['ppy']], [0, 0, 1.]])
    camera = np.array(state['camera_world_optical'])
    rgb = np.asarray(Image.open(trial / 'camera/color.png').convert('RGB'))
    depth = np.asarray(Image.open(trial / 'camera/depth.png'), dtype=np.float64) * params['depth_scale']
    mask = np.asarray(Image.open(trial / 'camera/mask.png')) > 0
    v, u = np.indices(depth.shape)
    z = depth
    points = np.stack([(u-k[0, 2])*z/k[0, 0],
                       (v-k[1, 2])*z/k[1, 1], z], axis=-1).reshape(-1, 3)
    colors = rgb.reshape(-1, 3) / 255.
    points_world = points @ camera[:3, :3].T + camera[:3, 3]
    # A display crop only; the original RGB-D and all proposals remain intact.
    valid = ((points[:, 2] > .12) & (points[:, 2] < 2.) &
             (points_world[:, 0] >= -.735) & (points_world[:, 0] <= .495) &
             (np.abs(points_world[:, 1]) <= 1.22) &
             (points_world[:, 2] >= -.012) & (points_world[:, 2] <= .65))
    points_world, colors = points_world[valid], colors[valid]
    if not len(points_world):
        raise RuntimeError('Empty tabletop RGB-D point cloud')
    if len(points_world) > 240000:
        sample = np.sort(np.random.default_rng(0).choice(len(points_world), 240000, replace=False))
        points_world, colors = points_world[sample], colors[sample]
    lo, hi = float(top[:, 0].min()), float(top[:, 0].max())
    norm = Normalize(lo if lo != hi else lo-1e-6, hi)
    cmap = plt.get_cmap('RdYlBu_r')
    glyph_colors = [cmap(norm(float(r[0]))) for r in top]
    camera_lines = [gripper_lines(r) for r in top]
    world_lines = [lines @ camera[:3, :3].T + camera[:3, 3] for lines in camera_lines]
    title = trial.name
    fig = plt.figure(figsize=(10, 8), facecolor='white')
    ax = fig.add_axes([.005, .065, .80, .88], projection='3d', computed_zorder=False)
    ax.set_proj_type('ortho')
    ax.scatter(*points_world.T, c=colors, s=2.2, marker='s', linewidths=0, depthshade=False, rasterized=True, zorder=1)
    # Draw the high scores last so dense coincident predictions remain visible.
    for lines, color in reversed(list(zip(world_lines, glyph_colors))):
        for line in lines:
            ax.plot(*line.T, color=color, linewidth=2.0, solid_capstyle='round', zorder=3)
    all_points = np.concatenate([points_world, np.concatenate(world_lines).reshape(-1, 3)])
    lower, upper = all_points.min(0), all_points.max(0)
    center = (lower + upper) / 2
    span = np.maximum(upper-lower, [.2, .2, .15])
    for setter, i in ((ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)):
        setter(center[i]-.54*span[i], center[i]+.54*span[i])
    ax.set_box_aspect(span, zoom=1.2)
    # Look from the observed side of the RGB-D surface, avoiding invented backs.
    toward_camera = camera[:3, 3] - np.array(state['object_center'])
    azim = np.degrees(np.arctan2(toward_camera[1], toward_camera[0]))
    elev = np.clip(np.degrees(np.arctan2(toward_camera[2], np.linalg.norm(toward_camera[:2])))-8, 35, 65)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    fig.text(.03, .97, f'{model_name} | target top 10', fontsize=17, weight='bold', va='top')
    fig.text(.03, .935, title, fontsize=9, color='#444444', va='top')
    handles = [Line2D([0], [0], color=c, lw=3, label=f'#{i+1:02d}  {r[0]:.6f}')
               for i, (r, c) in enumerate(zip(top, glyph_colors))]
    fig.legend(handles=handles, loc='center right', bbox_to_anchor=(.992, .56),
               frameon=False, title=f'{score_label}\nhigh to low', fontsize=10)
    fig.text(.03, .035, f'Recorded RGB-D point cloud | {candidate_stage} | before cuRobo',
             fontsize=9, color='#444444')
    figure_save(fig, out / 'grasp_top10.png')

    # Also retain a direct calibrated image projection for checking alignment.
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.subplots_adjust(left=.02, right=.81, bottom=.06, top=.91)
    ax.imshow(rgb)
    for lines, color in reversed(list(zip(camera_lines, glyph_colors))):
        for line in lines:
            if (line[:, 2] <= .001).any():
                continue
            uv = line @ k.T
            uv = uv[:, :2] / uv[:, 2:]
            ax.plot(uv[:, 0], uv[:, 1], color=color, linewidth=1.7)
    ax.set_xlim(0, rgb.shape[1]-1)
    ax.set_ylim(rgb.shape[0]-1, 0)
    ax.set_axis_off()
    fig.suptitle(f'{model_name} | target top 10 (RGB projection)', fontsize=15, weight='bold')
    fig.legend(handles=handles, loc='center right', bbox_to_anchor=(.995, .5), frameon=False,
               title=f'{score_label}\nhigh to low', fontsize=10)
    fig.text(.03, .025, title, fontsize=9)
    figure_save(fig, out / 'grasp_top10_rgb.png')
    source_files = [source, trial/'state.json', trial/'camera/params.json',
                    trial/'camera/color.png', trial/'camera/depth.png', trial/'camera/mask.png']
    metadata = {'trial': str(trial.resolve()), 'slot': trial.name, 'model_name': model_name,
        'candidate_stage': candidate_stage, 'score_label': score_label,
        'scope': 'This trial target only; renderer adds no inference, NMS, planner filtering, or physical execution',
        'selection': 'First min(10,N) input rows by stable descending network score; no extra suppression in renderer',
        'candidate_count': len(rows), 'displayed_count': len(top), 'target_mask_pixels': int(mask.sum()),
        'point_cloud': 'Backprojected recorded RGB-D, camera optical +X right +Y down +Z forward; displayed in world meters',
        'point_cloud_display_crop': {'camera_depth_m': [.12, 2.], 'world_x': [-.735, .495], 'world_y': [-1.22, 1.22], 'world_z': [-.012, .65]},
        'glyph': 'GraspNet candidate gripper centerline convention; not an ARX physical collision model',
        'wire_display': '3D candidate wires drawn over point cloud for readability; not an occlusion or collision certificate',
        'colors': 'RdYlBu_r normalized over these displayed model scores: high warm red, low cool blue',
        'source_sha256': {str(f.resolve()): sha(f) for f in source_files},
        'renderer_sha256': RENDERER_SHA256, 'visualization_version': VISUALIZATION_VERSION,
        'top10': [{'rank': i+1, 'raw_row_index': int(idx), 'score': float(row[0]),
                   'color': to_hex(color), 'graspnet_row': row.tolist()}
                  for i, (idx, row, color) in enumerate(zip(indices, top, glyph_colors))]}
    pool_path = trial / 'candidate_pool.json'
    if pool_path.exists():
        pool = json.loads(pool_path.read_text())['candidates']
        for item in metadata['top10']:
            item['candidate_metadata'] = pool[item['raw_row_index']]['metadata']
        metadata['source_sha256'][str(pool_path.resolve())] = sha(pool_path)
    save(out/'grasp_top10.json', metadata)
    return metadata


def campaign(campaign_path, watch):
    out = campaign_path/'grasp_visualizations'
    out.mkdir(exist_ok=True)
    save(out/'visualization_registration.json', {'created_at': datetime.now().astimezone().isoformat(),
        'renderer': str(Path(__file__).resolve()), 'renderer_sha256': RENDERER_SHA256,
        'scope': 'Derived visualization only; original campaign registration and physical pipeline unchanged',
        'per_trial_output': 'trials/<slot>/zg_outputs/',
        'max_candidates_per_target': 10, 'sorting': 'raw network score descending',
        'extra_inference': False, 'cpu_only': True})
    while True:
        records = []
        trials = sorted((campaign_path/'trials').iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        for trial in trials:
            if not trial.is_dir():
                continue
            destination = trial/'zg_outputs'
            receipt = destination/'grasp_top10.json'
            ready = trial/'network_candidates.json'
            if not ready.exists():
                records.append({'slot': trial.name, 'status': 'no_saved_inference' if (trial/'result.json').exists() else 'waiting_for_inference'})
                continue
            try:
                legacy = out/trial.name
                names = ('grasp_top10.png', 'grasp_top10_rgb.png', 'grasp_top10.json')
                if not receipt.exists() and all((legacy/name).exists() for name in names):
                    destination.mkdir(exist_ok=True)
                    for name in names[:2]:
                        shutil.copy2(legacy/name, destination/name)
                    migrated = json.loads((legacy/'grasp_top10.json').read_text())
                    migrated.update(visualization_version=VISUALIZATION_VERSION,
                                    copied_from=str(legacy.resolve()),
                                    output_layout='Per-trial zg_outputs; existing render copied unchanged')
                    save(receipt, migrated)
                    print('COPIED', trial.name, 'to', destination, flush=True)
                if (not receipt.exists()
                    or json.loads(receipt.read_text()).get('visualization_version') != VISUALIZATION_VERSION
                    or not all((destination/name).exists() for name in names)):
                    result = render(trial, destination)
                    print('RENDERED', trial.name, result['displayed_count'], flush=True)
                record = json.loads(receipt.read_text())
                records.append({'slot': trial.name, 'status': 'rendered', 'count': record['displayed_count'],
                                'image': str(destination/'grasp_top10.png')})
            except Exception:
                destination.mkdir(exist_ok=True)
                failure = {'slot': trial.name, 'status': 'visualization_error', 'error': traceback.format_exc()}
                save(destination/'visualization_error.json', failure)
                records.append(failure)
                print(failure['error'], flush=True)
        save(out/'index.json', {'updated_at': datetime.now().astimezone().isoformat(),
            'rendered': sum(r['status']=='rendered' for r in records), 'records': records})
        if not watch:
            break
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--campaign', type=Path)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    if args.trial:
        destination = args.output.resolve() if args.output else args.trial.resolve()/'zg_outputs'
        metadata = render(args.trial.resolve(), destination)
        print('RENDERED', args.trial.name, metadata['displayed_count'])
    else:
        campaign(args.campaign.resolve(), args.watch)


if __name__ == '__main__':
    main()
