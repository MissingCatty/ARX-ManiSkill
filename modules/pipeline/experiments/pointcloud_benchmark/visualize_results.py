"""Render inspectable figures from saved inputs and predictions, without inference."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree
import trimesh

from prepare_scenes import RUN, PROJECT, write
from geometry_evaluation import subsample

METHODS = {'zerograsp': 'ZeroGrasp', 'hunyuan-mini': 'HY Mini Turbo',
           'hunyuan-turbo': 'HY Turbo', 'spar3d': 'SPAR3D', 'rayst3r': 'RaySt3R'}
RESOURCES = PROJECT / 'docs/pointcloud_benchmark/resources'


def inputs():
    cases = json.loads((RUN / 'scene_selection.json').read_text())['cases']
    tile_w, tile_h = 320, 268
    sheet = Image.new('RGB', (tile_w * 5, tile_h * 10), '#f3f5f8')
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
    for i, case in enumerate(cases):
        source = RUN / 'segmentation_v03' / case['id'] / 'overlay.jpg'
        picture = Image.open(source).convert('RGB').resize((320, 240), Image.Resampling.LANCZOS)
        x, y = i % 5 * tile_w, i // 5 * tile_h
        sheet.paste(picture, (x, y))
        ImageDraw.Draw(sheet).text((x + 8, y + 245), case['id'] + '  ' + case['style_id'], fill='#223349', font=font)
    sheet.save(RESOURCES / 'inputs_contact_sheet.jpg', quality=87, optimize=True)
    sheet.resize((800, 1340), Image.Resampling.LANCZOS).save(RESOURCES / 'inputs_contact_sheet_thumb.jpg', quality=82, optimize=True)


def reference(case_id):
    mesh = trimesh.load(RUN / 'ground_truth' / case_id / 'surface_world.ply', force='mesh', process=False)
    old_state = np.random.get_state(); np.random.seed(20261122)
    points, _ = trimesh.sample.sample_surface(mesh, 50000)
    np.random.set_state(old_state)
    camera = json.loads((RUN / 'observations' / case_id / 'camera.json').read_text())
    transform = np.array(camera['camera_world_optical'])
    return (points - transform[:3, 3]) @ transform[:3, :3]


def failures():
    empty = []
    for method in METHODS:
        for file in sorted((RUN/'methods'/method).glob('C*/result.json')):
            result = json.loads(file.read_text())
            if result['status']=='model_output_failure': empty.append(result)
    if not empty: return
    width, height, columns = 320, 265, 4
    sheet = Image.new('RGB', (columns*width, ((len(empty)+columns-1)//columns)*height), '#f3f5f8')
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
    for i, result in enumerate(empty):
        cid = result['id']
        image = Image.open(RUN/'observations'/cid/'rgb.png').convert('RGB')
        mask = np.asarray(Image.open(RUN/'segmentation_v03'/cid/'mask.png'))>0
        y, x = np.nonzero(mask)
        image = image.crop((max(0, x.min()-30), max(0, y.min()-30), min(640, x.max()+31), min(480, y.max()+31)))
        image.thumbnail((width, height-50))
        left, top = (i%columns)*width, (i//columns)*height
        sheet.paste(image, (left+(width-image.width)//2, top))
        draw = ImageDraw.Draw(sheet)
        draw.text((left+8, top+height-43), f"{cid}  {result['style_id']}", fill='#223349', font=font)
        draw.text((left+8, top+height-23), METHODS[result['method']]+' : empty mesh', fill='#9b3030', font=font)
    sheet.save(RESOURCES/'empty_outputs.jpg', quality=85, optimize=True)


def comparison(case_id):
    truth = reference(case_id)
    center = (truth.min(0) + truth.max(0)) / 2
    half = np.ptp(truth, axis=0).max() * .7
    low, high = center - half, center + half
    figure = plt.figure(figsize=(21, 7))
    figure.suptitle(case_id + ' | fixed metric camera coordinates | color: distance to reference (0-20 mm)', fontsize=14)
    original = Image.open(RUN / 'observations' / case_id / 'rgb.png')
    mask = np.asarray(Image.open(RUN / 'segmentation_v03' / case_id / 'mask.png')) > 0
    yy, xx = np.nonzero(mask)
    crop = original.crop((max(0, xx.min()-35), max(0, yy.min()-35), min(640, xx.max()+36), min(480, yy.max()+36)))
    statuses = {}
    for row, azimuth in enumerate([-65, 115]):
        image_ax = figure.add_subplot(2, 7, row*7+1)
        image_ax.imshow(crop); image_ax.axis('off')
        image_ax.set_title('Shared head RGB' if row == 0 else 'Opposite 3D viewpoint', fontsize=10)
        for column, method in enumerate(['reference'] + list(METHODS), 2):
            ax = figure.add_subplot(2, 7, row*7+column, projection='3d')
            if method == 'reference':
                points = subsample(truth, 4000)
                ax.scatter(*points.T, s=.8, color='#63758a', rasterized=True)
                title = 'Rendered reference'
            else:
                folder = RUN / 'methods' / method / case_id
                result_path = folder / 'result.json'
                result = json.loads(result_path.read_text()) if result_path.exists() else {'status': 'pending'}
                statuses[method] = result['status']
                if result.get('success'):
                    full = np.load(folder / 'metric_points.npy')
                    points = subsample(full, 4000)
                    distances = cKDTree(truth).query(points)[0] * 1000
                    ax.scatter(*subsample(truth, 2000).T, s=.4, color='#aab1ba', alpha=.12, rasterized=True)
                    ax.scatter(*points.T, s=.9, c=distances, cmap='turbo', vmin=0, vmax=20, rasterized=True)
                    clipped = 100 * np.mean(np.any((full < low) | (full > high), axis=1))
                    title = f"{METHODS[method]}\nCD {result['geometry']['chamfer_l1_mm']:.1f} mm | F5 {result['geometry']['fscore_5mm']*100:.0f}%"
                    if clipped > .1: title += f'\n{clipped:.1f}% outside view'
                else:
                    title = METHODS[method] + '\n' + result['status'].replace('_', ' ')
            ax.set_title(title, fontsize=9)
            ax.set_xlim(low[0], high[0]); ax.set_ylim(low[1], high[1]); ax.set_zlim(low[2], high[2])
            ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=-25, azim=azimuth)
            ax.set_axis_off()
    color_ax = figure.add_axes([.32, .04, .38, .012])
    figure.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 20), cmap='turbo'), cax=color_ax, orientation='horizontal', label='prediction-to-reference distance (mm)')
    figure.subplots_adjust(left=.01, right=.99, top=.87, bottom=.11, wspace=0, hspace=.15)
    file = RESOURCES / f'comparison_{case_id.lower()}.jpg'
    figure.savefig(file, dpi=130, facecolor='white'); plt.close(figure)
    with Image.open(file) as image:
        image.thumbnail((1050, 350))
        image.save(RESOURCES / f'comparison_{case_id.lower()}_thumb.jpg', quality=82, optimize=True)
    return {'id': case_id, 'statuses': statuses, 'file': str(file)}


def distributions():
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    clouds, times, scores = [], [], []
    labels = []
    for method, name in METHODS.items():
        results = [json.loads(p.read_text()) for p in sorted((RUN/'methods'/method).glob('C*/result.json'))]
        valid = [r for r in results if r.get('success')]
        if not valid: continue
        labels.append(name)
        clouds.append([r['geometry']['chamfer_l1_mm'] for r in valid])
        times.append([r['generation_s'] for r in results if 'generation_s' in r])
        normal = [r for r in results if r['status'] in ['completed', 'model_output_failure']]
        scores.append(sum(r['geometry']['fscore_5mm'] for r in valid)/len(normal)*100)
    if labels:
        axes[0].boxplot(clouds, tick_labels=labels, showfliers=True)
        axes[0].set_ylabel('metric Chamfer distance (mm, lower is better)')
        axes[1].boxplot(times, tick_labels=labels, showfliers=True)
        axes[1].set_ylabel('resident reconstruction time (s)')
        axes[2].bar(labels, scores, color='#3975b9'); axes[2].set_ylim(0, 100)
        axes[2].set_ylabel('mean F-score at 5 mm (%)')
    for ax in axes:
        ax.tick_params(axis='x', labelrotation=30); ax.grid(axis='y', alpha=.2)
    figure.suptitle('Shared 50 scenes | output failures score zero; unexecuted slots excluded')
    figure.tight_layout()
    figure.savefig(RESOURCES / 'metric_distributions.jpg', dpi=150)
    plt.close(figure)
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname=str(PROJECT.parent/'claro/缓存/对比视频字体/NotoSansCJKsc-Regular.otf'))
    panels = [
        ('reconstruction_error', '重建误差分布', '形状误差（毫米，越小越好）'),
        ('reconstruction_time', '重建耗时分布', '每次生成所需时间（秒，越短越好）'),
        ('shape_match_score', '形状匹配得分', '平均形状匹配得分（%，越高越好）'),
    ]
    for index, (filename, title, ylabel) in enumerate(panels):
        figure, ax = plt.subplots(figsize=(5.5, 4.5))
        if index < 2:
            ax.boxplot(clouds if index == 0 else times, tick_labels=labels, showfliers=True)
        else:
            bars = ax.bar(labels, scores, color='#3975b9')
            ax.set_ylim(0, 108)
            ax.bar_label(bars, labels=[f'{score:.1f}%' for score in scores], padding=3, fontsize=9)
        ax.set_title(title, fontproperties=font, fontsize=15, pad=12)
        ax.set_ylabel(ylabel, fontproperties=font, fontsize=10)
        ax.tick_params(axis='x', labelrotation=30, labelsize=9)
        for label in ax.get_xticklabels(): label.set_horizontalalignment('right')
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
        figure.tight_layout()
        original = RESOURCES/f'{filename}.jpg'
        figure.savefig(original, dpi=180, facecolor='white'); plt.close(figure)
        with Image.open(original) as preview:
            preview.thumbnail((500, 420), Image.Resampling.LANCZOS)
            preview.save(RESOURCES/f'{filename}_thumb.jpg', quality=85, optimize=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--inputs-only', action='store_true'); args = parser.parse_args()
    RESOURCES.mkdir(parents=True, exist_ok=True); inputs()
    if not args.inputs_only:
        failures()
        gallery = [comparison(f'C{i:03d}') for i in range(1, 51, 5)]
        distributions()
        write(RUN / 'figure_manifest.json', {'selection': 'first registered style of each category; no metric-based selection', 'figures': gallery})
    print(RESOURCES)


if __name__ == '__main__': main()
