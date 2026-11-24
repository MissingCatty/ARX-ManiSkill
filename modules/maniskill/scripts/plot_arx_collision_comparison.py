"""Make a matched-view comparison of cooked solids and the before/after sphere unions."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import ConvexHull
from check_arx_collision_spheres import LINKS, ROOT, SIM, fk


def unit(a):
    return np.array(a)/np.linalg.norm(a)


def camera(eye, up):
    right = unit(np.cross(up, eye))
    return np.stack([right, -unit(np.cross(eye, right)), unit(eye)], axis=1)


def bounds(projected, spheres, shape):
    v = np.vstack(projected)[:, :2]
    lo, hi = v.min(axis=0), v.max(axis=0)
    for balls in spheres:
        lo = np.minimum(lo, (balls[:, :2]-balls[:, 3:4]).min(axis=0))
        hi = np.maximum(hi, (balls[:, :2]+balls[:, 3:4]).max(axis=0))
    return (lo+hi)/2, min((shape[1]-28)/(hi-lo)[0], (shape[0]-26)/(hi-lo)[1])


def new_buffers(shape):
    return np.full(shape, -np.inf), np.zeros(shape)


def raster_bounds(xy, shape, pad=0):
    x0, y0 = np.maximum(0, np.floor(xy.min(axis=0)-pad).astype(int))
    x1, y1 = np.minimum(np.array(shape[::-1])-1, np.ceil(xy.max(axis=0)+pad).astype(int))
    ys, xs = np.ogrid[y0:y1+1, x0:x1+1]
    return np.s_[y0:y1+1, x0:x1+1], xs, ys


def triangles(parts, shape):
    depth, shade = new_buffers(shape)
    light = unit([-.4, -.52, .76])
    for verts in parts:
        hull = ConvexHull(verts)
        for tri, normal in zip(hull.simplices, hull.equations[:, :3]):
            if normal[2] <= 1e-8:
                continue
            a, b, c = verts[tri]
            ab, ac = b-a, c-a
            det = ab[0]*ac[1]-ab[1]*ac[0]
            if abs(det) < 1e-7:
                continue
            sl, x, y = raster_bounds(verts[tri, :2], shape)
            u = ((x-a[0])*ac[1]-(y-a[1])*ac[0])/det
            v = ((y-a[1])*ab[0]-(x-a[0])*ab[1])/det
            z = a[2]+u*ab[2]+v*ac[2]
            keep = (u >= -1e-8)&(v >= -1e-8)&(u+v <= 1+1e-8)&(z > depth[sl])
            depth[sl][keep] = z[keep]
            shade[sl][keep] = .64+.36*max(0, normal@light)
    return depth, shade


def spheres(balls, shape):
    depth, shade = new_buffers(shape)
    for x, y, z, r in balls:
        sl, xx, yy = raster_bounds(np.array([[x, y]]), shape, r)
        dx, dy = (xx-x)/r, (yy-y)/r
        dz2 = 1-dx**2-dy**2
        dz = np.sqrt(np.maximum(0, dz2))
        hit = z + r*dz
        keep = (dz2>=0)&(hit>depth[sl])
        depth[sl][keep] = hit[keep]
        shade[sl][keep] = (.52+.48*np.maximum(0, -.43*dx-.49*dy+.76*dz))[keep]
    return depth, shade


def render(parts, old, new, eye, up, size):
    cam = camera(eye, up)
    parts = [p@cam for p in parts]
    both = [np.column_stack([s[:, :3]@cam, s[:, 3]]) for s in (old, new)]
    center, scale = bounds(parts, both, size)
    offset = np.r_[np.array(size[::-1])/2-center*scale, 0.]
    parts = [p*scale+offset for p in parts]
    for s in both:
        s *= scale
        s[:, :3] += offset
    depth, shade = triangles(parts, size)
    physical = np.isfinite(depth)
    bg = np.array([247, 249, 250])
    grey = shade[..., None]*np.array([169, 182, 193])
    ref = np.where(physical[..., None], grey, bg)
    images = [ref]
    for s in both:
        depth_s, shade_s = spheres(s, size)
        hit = np.isfinite(depth_s)
        clr = shade_s[..., None]*np.array([248, 167, 70])
        image = np.where(hit[..., None], clr, ref)
        overlap = hit&physical
        image[overlap] = .65*clr[overlap]+.35*grey[overlap]
        outline = cv2.morphologyEx(physical.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3,3),np.uint8))>0
        image[outline] = [41, 67, 81]
        images.append(image)
    return [Image.fromarray(np.uint8(np.clip(im, 0, 255))) for im in images]


def run(work, orig, fontpath):
    geo=json.loads((work/'geometry.json').read_text())
    baseline=yaml.safe_load((orig/'model/arx.yml').read_text())['robot_cfg']['kinematics']
    case=sorted(orig.glob('full_pose_solutions.npz'))[0]
    with np.load(case) as z:
        candidate, branch = np.argwhere(z['cpu_pose_valid'])[0]
        q=z['q'][candidate,branch]
    frames={l:t[0] for l,t in fk(q[None],baseline['lock_joints'],SIM).items()}
    parts,old,new={},{},{}
    # Link envelope comparisons sample equal-area directions; centers, radii and mesh points are local mm.
    i=np.arange(8192)
    z=1-2*(i+.5)/len(i)
    th=i*np.pi*(3-np.sqrt(5))
    dirs=np.column_stack([np.sqrt(1-z*z)*np.cos(th),np.sqrt(1-z*z)*np.sin(th),z])
    support={}
    for l in LINKS:
        t=frames[l]
        def place(p):
            return p@t[:3,:3].T+t[:3,3]*1000
        with np.load(work/f'{l}.npz') as data:
            local=[data[k] for k in data.files if k.startswith('convex_')]
            parts[l]=[place(p) for p in local]
            after=data['balls']
        before=np.array([[*b['center'],b['radius']] for b in baseline['collision_spheres'][l]])*1000
        old[l]=np.column_stack([place(before[:,:3]),before[:,3]])
        new[l]=np.column_stack([place(after[:,:3]),after[:,3]])
        real=(np.vstack(local)@dirs.T).max(axis=0)
        support[l]={name: float(((balls[:,:3]@dirs.T+balls[:,3,None]).max(axis=0)-real).max())
                    for name, balls in [('old_mm',before),('new_mm',after)]}
    (work/'directional_fit.json').write_text(json.dumps({'directions_per_link':len(dirs),'max_outward_support':support,
        'figure_case':orig.name,'zero_based_candidate':int(candidate),'zero_based_ik_branch':int(branch)},indent=2))
    direction=frames[LINKS[5]][:3,3]-frames[LINKS[0]][:3,3]
    direction[2]=0.
    direction=unit(direction)
    up=np.array([0.,0.,1.])
    eye=unit(np.cross(direction,up)+.32*up+.16*direction)
    hand=frames[LINKS[5]][:3,:3]
    images=[]
    for names, angle, above, height in [(LINKS,eye,up,370),(LINKS[5:],hand@unit([.48,-.31,.82]),hand@np.array([1.,0,0]),340)]:
        images.append(render([p for l in names for p in parts[l]],np.vstack([old[l] for l in names]),
                             np.vstack([new[l] for l in names]),angle,above,(height,518)))
    board=Image.new('RGB',(1666,1046),'white')
    draw=ImageDraw.Draw(board)
    def label(x,y,s,size,color='#203640'):
        draw.text((x,y),s,font=ImageFont.truetype(str(fontpath),size),fill=color,anchor='lt')
    label(36,24,'ARX 右臂与夹爪  |  碰撞球改前与改后',39)
    label(38,82,'同一开爪抓取姿态。橙色为规划碰撞球，深色轮廓为仿真凸碰撞体。每行使用相同视角和比例。',21,'#48616d')
    xs=[38,574,1110]
    old_total=sum(len(old[l]) for l in LINKS)
    new_total=sum(len(new[l]) for l in LINKS)
    for x,ttl in zip(xs,['仿真碰撞体','改前  ·  '+str(old_total)+' 个球','改后  ·  '+str(new_total)+' 个球']):
        label(x,137,ttl,28)
    for y,row in zip([207,645],images):
        for x,im in zip(xs,row):
            board.paste(im,(x,y))
    label(38,181,'整条右臂',19,'#465f6a')
    label(38,609,'腕部与双指放大',19,'#465f6a')
    label(574,609,f'腕部 {len(old[LINKS[5]])}    双指 {len(old[LINKS[6]])+len(old[LINKS[7]])}',19,'#465f6a')
    label(1110,609,f'腕部 {len(new[LINKS[5]])}    双指 {len(new[LINKS[6]])+len(new[LINKS[7]])}',19,'#465f6a')
    label(38,1007,f'姿态取自 {orig.name}，候选 {candidate+1}，IK 分支 {branch+1}。',17,'#536b76')
    board.save(work/'ARX碰撞球改前改后对比.png')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('work','source_case','cjk_font'):
        p.add_argument(arg,type=Path)
    a=p.parse_args()
    run(a.work,a.source_case,a.cjk_font)
