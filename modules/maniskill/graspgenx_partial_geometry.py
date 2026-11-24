"""Measured ARX geometry and visible-depth inputs for GraspGenX experiments."""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

CODE = Path(__file__).resolve().parent
PROTOCOL = 'arx_gt_partial_topdown_pair_v1'
TCP_M = .043
TOPDOWN_DEGREES = 30.
# GraspGenX +X is closing; ARX EEF +Y is closing. Both approach along +Z.
T_EEF_GENERATOR = np.eye(4)
T_EEF_GENERATOR[:3, :3] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]


def save(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    def encode(x):
        if isinstance(x, np.ndarray):return x.tolist()
        if isinstance(x, np.generic):return x.item()
        raise TypeError(type(x).__name__)
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=encode) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def origin(element):
    transform = np.eye(4)
    if element is not None:
        transform[:3, 3] = np.fromstring(element.get('xyz', '0 0 0'), sep=' ')
        transform[:3, :3] = Rotation.from_euler('xyz', np.fromstring(element.get('rpy', '0 0 0'), sep=' ')).as_matrix()
    return transform


def gripper_descriptor():
    """Derive the conditioning cavity from the unchanged simulation URDF."""
    import trimesh
    asset = CODE / 'assets/arx';urdf = asset / 'robot.urdf'
    tree = ET.parse(urdf).getroot()
    joints = {j.find('child').get('link'): j for j in tree.findall('joint')}
    links = {l.get('name'): l for l in tree.findall('link')}
    sources = {str(urdf): sha(urdf)}
    def vertices(opening):
        cache = {}
        def world(link):
            if link in cache:return cache[link]
            joint = joints.get(link)
            if joint is None:value = np.eye(4)
            else:
                value = world(joint.find('parent').get('link')) @ origin(joint.find('origin'))
                if joint.get('type') == 'prismatic' and joint.get('name') in ['right_joint27', 'right_joint28']:
                    shift = np.eye(4)
                    shift[:3, 3] = np.fromstring(joint.find('axis').get('xyz'), sep=' ') * opening
                    value = value @ shift
            cache[link] = value
            return value
        inverse = np.linalg.inv(world('right_eef_link') @ T_EEF_GENERATOR)
        result = []
        for name in ['right_link27', 'right_link28']:
            parts = []
            for collision in links[name].findall('collision'):
                mesh = collision.find('geometry/mesh');path = asset / mesh.get('filename')
                sources[str(path)] = sha(path)
                xyz = np.asarray(trimesh.load_mesh(path, process=False).vertices)
                xyz = xyz * np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
                transform = inverse @ world(name) @ origin(collision.find('origin'))
                parts.append(xyz @ transform[:3, :3].T + transform[:3, 3])
            result.append(np.concatenate(parts))
        return result
    opened, middle, closed = vertices(.044), vertices(.022), vertices(0.)
    # The full finger includes proximal mounting blocks that overlap in X.
    # Condition the model on the measured distal contact zone, not that block.
    pads_path = CODE.parents[1] / 'simulation/robots/arx/finger_pad_geometry.json'
    pads = json.loads(pads_path.read_text())['parts']
    sources[str(pads_path)] = sha(pads_path)
    contact_z_low = max(pads[name]['pad_neutral_EEF_bounds_m'][0][2]
                        for name in ['right_link27', 'right_link28'])
    def cavity(parts):
        distal = [x[x[:, 2] >= contact_z_low - 1.e-6] for x in parts]
        assert all(len(x) >= 4 for x in distal)
        ordered = sorted(distal, key=lambda x: x[:, 0].mean())
        negative, positive = ordered
        lower = np.maximum(negative.min(0), positive.min(0))
        upper = np.minimum(negative.max(0), positive.max(0))
        lower[0], upper[0] = negative[:, 0].max(), positive[:, 0].min()
        assert np.all(upper > lower), (lower, upper)
        return upper - lower, (upper + lower) / 2
    ext_open, off_open = cavity(opened)
    ext_mid, off_mid = cavity(middle)
    depth = max(x[:, 2].max() for x in opened)
    assert .02 < depth < .08 and .05 < ext_open[0] < .12
    params = {'extents_open': ext_open, 'offset_open': off_open,
              'extents_mid': ext_mid, 'offset_mid': off_mid, 'fingertip_depth': float(depth)}
    return {'protocol': PROTOCOL, 'conditioner': 'Actual unchanged ARX collision meshes in canonical EEF frame',
            'T_eef_generator': T_EEF_GENERATOR, 'parameters': params,
            'finger_bounds_open': [np.stack([x.min(0), x.max(0)]) for x in opened],
            'finger_bounds_closed': [np.stack([x.min(0), x.max(0)]) for x in closed],
            'source_sha256': sources, 'robot_or_physics_modified': False,
            'maximum_opening_m': .088, 'tcp_m': TCP_M}


def visible_points(directory):
    """Noiseless rendered depth + GT instance mask, never full mesh sampling."""
    directory = Path(directory)
    params = json.loads((directory / 'camera/params.json').read_text());k = params['intrinsics']
    depth = np.asarray(Image.open(directory / 'camera/depth.png'), dtype=np.float64) * params['depth_scale']
    mask = np.asarray(Image.open(directory / 'camera/mask.png')) > 0
    v, u = np.indices(depth.shape)
    keep = mask & np.isfinite(depth) & (depth > .12) & (depth < 2.)
    camera_points = np.stack([(u-k['ppx'])*depth/k['fx'], (v-k['ppy'])*depth/k['fy'], depth], -1)[keep]
    state = json.loads((directory / 'state.json').read_text())
    camera = np.asarray(state['camera_world_optical'])
    world_points = camera_points @ camera[:3, :3].T + camera[:3, 3]
    return camera_points.astype(np.float32), world_points.astype(np.float32)


def angle_from_down(pose):
    approach = np.asarray(pose)[:3, 2]
    return float(np.degrees(np.arccos(np.clip(-approach[2], -1., 1.))))


def generator_to_contact(grasp_world):
    eef = np.asarray(grasp_world) @ np.linalg.inv(T_EEF_GENERATOR)
    contact = eef.copy();contact[:3, 3] += TCP_M * eef[:3, 2]
    return contact


if __name__ == '__main__':
    import sys
    descriptor = gripper_descriptor()
    save(Path(sys.argv[1]), descriptor)
    print(json.dumps(descriptor['parameters'], default=lambda x: x.tolist(), indent=2))
