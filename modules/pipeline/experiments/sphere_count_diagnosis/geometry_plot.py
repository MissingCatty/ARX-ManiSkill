"""Exact fingertip cross-sections of native colliders and sphere unions."""
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
import trimesh

from arx_project.layout import ROOT, read_json
from arx_project.assets import resource_path
from .cooked_geometry_fit import loader
from modules.curobo.sphere_study.fit_metrics import ConvexUnion


def polygon_section(vertices, axis, value):
    centered = vertices-vertices.mean(0)
    if np.linalg.matrix_rank(centered) == 3:
        faces = ConvexHull(vertices).simplices
    else:
        faces = np.array([[i, j] for i in range(len(vertices)) for j in range(i+1, len(vertices))])
    edges = set()
    for face in faces:
        for i in face:
            for j in face:
                if i < j:
                    edges.add((i, j))
    points = list(vertices[np.abs(vertices[:, axis]-value) < 1e-9])
    for i, j in edges:
        a, b = vertices[i], vertices[j]
        if (a[axis]-value)*(b[axis]-value) < 0:
            points.append(a+(b-a)*(value-a[axis])/(b[axis]-a[axis]))
    return np.unique(np.asarray(points), axis=0) if points else np.empty((0, 3))


def plot(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon
    reg = read_json(root/'registration.json')
    load = loader(ROOT/reg['native_geometry_source'])
    meshes, _ = load(resource_path('arx_robot'), 'right')
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
    fig.set_constrained_layout_pads(h_pad=.10, w_pad=.06, hspace=.10, wspace=.04)
    for row, name in enumerate(['right_link27', 'right_link28']):
        directory = root/'refined/links/right'/name
        data = np.load(directory/'samples.npz', allow_pickle=False)
        transform = data['eef_from_link']
        model = ConvexUnion(meshes[name])
        added = data['additional_512_vs_871_m']
        point = data['boundary_512'][int(added.argmax())]
        nearest, _, _ = trimesh.proximity.closest_point(model.mesh, point[None])
        plane = float((nearest @ transform[:3, :3].T + transform[:3, 3])[0, 0])
        polygons = []
        for part in model.parts:
            vertices = part.vertices @ transform[:3, :3].T + transform[:3, 3]
            section = polygon_section(vertices, 0, plane)
            if len(section):
                polygons.append(section[:, [1, 2]]*1000)
        tip = float(data['tip_eef_z_m'])*1000
        low, high = tip-12, tip+12
        clouds = [p for p in polygons if p[:, 1].max() >= low]
        relevant = np.concatenate(clouds)
        xmin, xmax = relevant[:, 0].min()-12, relevant[:, 0].max()+12
        for col, count in enumerate([871, 512]):
            ax = axes[row, col]
            balls = data[f'spheres_{count}'].copy()
            balls[:, :3] = balls[:, :3] @ transform[:3, :3].T + transform[:3, 3]
            delta = balls[:, 0]-plane
            for ball, offset in zip(balls, delta):
                if abs(offset) <= ball[3]:
                    radius = np.sqrt(max(0, ball[3]**2-offset**2))*1000
                    ax.add_patch(Circle(ball[[1, 2]]*1000, radius, facecolor='#e69f00',
                        edgecolor='#c58400', alpha=.23, linewidth=.45))
            for p in polygons:
                if len(p) >= 3 and np.linalg.matrix_rank(p-p.mean(0)) == 2:
                    ax.add_patch(Polygon(p[ConvexHull(p).vertices], facecolor='#77838c', edgecolor='#26343d',
                        alpha=.88, linewidth=.9))
                elif len(p) > 1:
                    ax.plot(p[:, 0], p[:, 1], color='#26343d', linewidth=1.)
            ax.axhline(tip, color='#0072b2', linewidth=1.1, linestyle='--', label='Physical tip plane')
            ax.axhline(tip-10, color='#777777', linewidth=.8, linestyle=':')
            ax.set_xlim(xmin, xmax);ax.set_ylim(low, high);ax.set_aspect('equal');ax.grid(alpha=.16)
            ax.set_title(f'{count} total spheres: finger {name[-2:]} ({len(balls)} spheres)', fontsize=12)
            ax.set_xlabel('EEF closing direction Y (mm)');ax.set_ylabel('EEF approach direction Z (mm)')
            ax.text(.02, .97, f'Exact section at EEF X = {plane*1000:.2f} mm',
                transform=ax.transAxes, va='top', fontsize=9)
        data.close()
    fig.suptitle('Native physical geometry (grey) and collision spheres (orange)\nSame cross-section and scale in each row', fontsize=14)
    target = root/'fingertip_cross_sections.png';fig.savefig(target, dpi=155);plt.close(fig)
    return target
