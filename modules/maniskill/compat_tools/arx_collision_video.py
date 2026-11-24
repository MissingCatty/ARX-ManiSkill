"""Join original observation video and a synchronized, static grasp endpoint reenactment."""
import argparse
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'outputs/arx_mpc_grasp_ik_audit_20261119_01'
CHECK = ROOT / 'outputs/arx_collision_spheres_20261119_01'
OUT = CHECK / '双视角视频'
FONT = '/tmp/arx-fig-noto-sc.otf'
CASES = {
    'I0044': ('垒球', '木块', 'right_link27'),
    'I0003': ('乐高积木', '旁边的杯子', 'right_link28'),
    'I0048': ('一字螺丝刀', '罐头罐', 'right_link28'),
}
LEFT, RIGHT, HEIGHT = 960, 720, 540


def read(path):
    return json.loads(path.read_text())


def font(size):
    return ImageFont.truetype(FONT, size)


def photo(rgb):
    return Image.fromarray(np.asarray(rgb, dtype=np.uint8)[..., :3])


def capture_all():
    sys.path.insert(0, str(ROOT / 'arx_pipeline'))
    import sapien
    import trimesh
    from environment import ARM, make

    for case in read(CHECK / 'endpoint_contact_pairs.json')['case_results']:
        prefix = case['case'].split('_')[0]
        if prefix not in CASES:
            continue
        target, obstacle, link = CASES[prefix]
        witness = next(w for w in case['examples'] if w['link'] == link)
        group = AUDIT / case['case']
        trial = Path(read(group / 'audit.json')['trial'])
        state = read(trial / 'state.json')
        actors = [dict(name=k, **v) for k, v in state['objects'].items()]
        scene = {'actors': actors}
        env = make(state['objects']['target']['model_id'], perception_camera='right_wrist', scene_spec=scene)
        try:
            robot = env.agent.robot
            names = [j.name for j in robot.active_joints]
            assert names == state['joint_names']
            robot.set_pose(sapien.Pose(np.asarray(state['links']['base_link'])))
            robot.set_qpos(np.asarray(state['qpos'], dtype=np.float32)[None])
            for actor in actors:
                p = actor['pose']
                env.objects[actor['name']].set_pose(sapien.Pose(p[:3], p[3:]))
            initial = robot.links_map['right_eef_link'].pose.to_transformation_matrix().cpu().numpy()[0]
            np.testing.assert_allclose(initial, state['links']['right_eef_link'], atol=1e-5, rtol=0)
            with np.load(group/'full_pose_solutions.npz') as data:
                c, b = witness['candidate_1_based']-1, witness['ik_branch_1_based']-1
                assert data['cpu_pose_valid'][c, b]
                q = np.asarray(state['qpos'], dtype=np.float32)
                q[[names.index(n) for n in ARM]] = data['q'][c, b]
                goal = data['goals'][c]
            robot.set_qpos(q[None])
            np.testing.assert_allclose(robot.qpos.cpu().numpy()[0], q, atol=1e-6, rtol=0)
            eef = robot.links_map['right_eef_link'].pose.to_transformation_matrix().cpu().numpy()[0]
            assert np.linalg.norm(eef[:3, 3]-goal[:3, 3]) < .00201
            np.testing.assert_allclose(env.objects[witness['obstacle']].pose.raw_pose.cpu().numpy()[0],
                                       state['objects'][witness['obstacle']]['pose'], atol=1e-6, rtol=0)
            left = photo(env.render().cpu().numpy()[0])
            obs = env.get_obs()
            right = photo(obs['sensor_data']['right_wrist']['rgb'].cpu().numpy()[0])
            obstacle_meta = state['scene_meshes'][witness['obstacle']]
            mesh = trimesh.load_mesh(obstacle_meta['file_path'], process=False)
            center = np.mean(mesh.bounds, axis=0)
            world = np.asarray(obstacle_meta['world']) @ np.r_[center, 1.]
            cam = env.scene.human_render_cameras['overview'].get_params()
            xyz = (cam['extrinsic_cv'].cpu().numpy()[0] @ world)[:3]
            uv = (cam['intrinsic_cv'].cpu().numpy()[0] @ xyz)[:2]/xyz[2]
            assert xyz[2] > 0 and (uv > 0).all() and (uv < left.size).all(), uv
            width, height = 896, 504
            x = int(np.clip(uv[0] - width/2, 0, left.width-width))
            y = int(np.clip(uv[1] - height/2, 0, left.height-height))
            left = left.crop((x,y,x+width,y+height))
            left.save(OUT / (prefix+'_终点左前方.png'))
            right.save(OUT / (prefix+'_终点腕部.png'))
            metadata = {'case':case['case'], 'trial':str(trial), 'target':target, 'obstacle_label':obstacle,
                        'witness':witness,'obstacle_xy_in_overview_crop':(uv-[x,y]).tolist()}
            (OUT / (prefix+'_复现信息.json')).write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
            print(f'{prefix} captured', flush=True)
        finally:
            env.close()


def label(canvas, text, origin, size, fill):
    ImageDraw.Draw(canvas).text(origin, text, font=font(size), anchor='lt', fill=fill)


def board(info, endpoint=False):
    img = Image.new('RGB', (LEFT+RIGHT, HEIGHT+102), '#12232d')
    size = 29
    group = info['case'].split('_')[0]
    detail = '原始录像  ·  移动到腕部观察位'
    if endpoint:
        w = info['witness']
        detail = f'静态终点复现  ·  候选 {w["candidate_1_based"]}  ·  {w["link"]} 与{info["obstacle_label"]}相交'
    label(img, f'{group}   {info["target"]}', (20,17), size, 'white')
    label(img, detail, (406,18), 26, '#ffcf8a' if endpoint else '#d8ebf5')
    label(img, '第三人称 · 机器人左前方' + ('（局部）' if endpoint else ''), (20,66), 21, '#d8ebf5')
    label(img, '右腕相机', (LEFT+16,66), 21, '#d8ebf5')
    return img


def add_images(board, left, right):
    img = board.copy()
    img.paste(left.resize((LEFT,HEIGHT), Image.Resampling.LANCZOS), (0,102))
    img.paste(right.resize((RIGHT,HEIGHT), Image.Resampling.LANCZOS), (LEFT,102))
    return img


def encode():
    for prefix in CASES:
        info = read(OUT/(prefix+'_复现信息.json'))
        trial = Path(info['trial'])
        raw_header, endpoint_header = board(info), board(info, endpoint=True)
        left = Image.open(OUT/(prefix+'_终点左前方.png'))
        right = Image.open(OUT/(prefix+'_终点腕部.png'))
        snapshot = add_images(endpoint_header, left, right)
        draw = ImageDraw.Draw(snapshot)
        xy = np.array(info['obstacle_xy_in_overview_crop']) * [LEFT/left.width, HEIGHT/left.height] + [0,102]
        x, y = xy
        ink = '#ffcd64'
        draw.ellipse((x-13,y-13,x+13,y+13), outline=ink, width=4)
        draw.line((x+13,y,x+101,y), fill=ink, width=3)
        draw.rounded_rectangle((x+102,y-20,x+274,y+19), radius=6, fill='#12232d')
        label(snapshot, info['obstacle_label'], (x+114,y-12), 20, ink)
        snapshot.save(OUT/(prefix+'_双视角终点.png'))
        video = OUT/(f'{prefix}_{info["target"]}_双视角.mp4')
        with imageio.get_reader(trial/'execution.mp4', input_params=['-threads','2']) as a, \
             imageio.get_reader(trial/'right_wrist.mp4', input_params=['-threads','2']) as b, \
             imageio.get_writer(video, fps=20, codec='libx264', macro_block_size=2,
                                output_params=['-preset','fast','-crf','21','-movflags','+faststart','-threads','3']) as writer:
            assert a.get_meta_data()['fps'] == b.get_meta_data()['fps'] == 20
            assert a.count_frames() == b.count_frames() == 120
            for frame_left, frame_right in zip(a,b,strict=True):
                writer.append_data(np.asarray(add_images(raw_header, photo(frame_left), photo(frame_right))))
            for _ in range(80):
                writer.append_data(np.asarray(snapshot))
        print(video, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['capture','encode'])
    args = p.parse_args()
    (capture_all if args.phase == 'capture' else encode)()
