"""Same-render-product capture after physics, without another physics step."""
import copy
import numpy as np

VERSION = 'arx_nearfield_capture_v1'
AXES = np.diag([1., -1., -1., 1.])


class CaptureUnavailable(RuntimeError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def array_data(value):
    # Replicator annotators may include metadata alongside their data buffer.
    if isinstance(value, dict):value = value['data']
    return np.array(value, copy=True)


def buffer_info(value):
    a = np.asarray(value)
    return {'shape':list(a.shape), 'dtype':str(a.dtype), 'size':int(a.size),
            'nonzero':int(np.count_nonzero(a))}


def capture_trigger(camera, stage):
    if not camera.prim_path.startswith('/World/COOLASSensors/'):
        raise ValueError('Autotrigger requires the separate capture camera')
    prim=stage.GetPrimAtPath(camera.prim_path)
    rate=prim.GetAttribute('omni:sensor:tickRate')
    original=float(rate.Get());rate.Set(0.)
    if float(rate.Get())!=0.:raise RuntimeError('Capture render trigger was not applied')
    return {'camera_path':camera.prim_path,'previous_tick_rate_hz':original,'capture_tick_rate_hz':0.,
        'purpose':'Refresh same-time post-physics render buffers; sample/physics clocks unchanged'}


def plain(value):
    if isinstance(value, dict):return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):return [plain(v) for v in value]
    if hasattr(value, 'numpy'):value = value.numpy()
    if isinstance(value, np.ndarray):return value.tolist()
    if isinstance(value, np.generic):return value.item()
    return value


def reference_seconds(value):
    try:
        n = int(np.asarray(value['referenceTimeNumerator']).item())
        d = int(np.asarray(value['referenceTimeDenominator']).item())
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError('Missing render reference time') from exc
    if d <= 0:raise ValueError('Invalid render reference denominator')
    return n / d


def masks_from_ids(ids, mapping, roles, environment):
    ids = np.asarray(ids)
    if ids.ndim != 2 or ids.dtype != np.uint32 or not ids.size:
        raise ValueError('Missing integer instance image')
    if set(roles) != {'target', 'right_link27', 'right_link28'}:
        raise ValueError('Missing target/finger identities')
    if len(set(roles.values())) != 3 or any(not roles[n].endswith('/'+n) for n in ('right_link27','right_link28')):
        raise ValueError('Finger identities swapped or duplicated')
    if any(not p.startswith(environment + '/') for p in roles.values()):
        raise ValueError('Role belongs to a different environment')
    normalized = {}
    for key, path in mapping.items():
        if isinstance(path, dict):path = path.get('prim', path.get('path'))
        if not isinstance(path, str):raise ValueError('Invalid instance path mapping')
        normalized[str(int(key))] = path
    masks = {name: np.zeros(ids.shape, dtype=bool) for name in roles}
    for ident in np.unique(ids):
        path = normalized.get(str(int(ident)))
        if path is None:raise ValueError(f'Instance {ident} has no path mapping')
        if path in ('BACKGROUND', 'UNLABELLED', 'UNLABELED', ''):continue
        if not path.startswith('/'):raise ValueError('Instance path is not absolute')
        if '/Envs/' in path and not path.startswith(environment + '/'):
            raise ValueError('Instance pixels belong to a different environment')
        for name, root in roles.items():
            if path == root or path.startswith(root + '/'):
                masks[name] |= ids == ident
    # A fully occluded finger may legitimately have zero visible pixels.
    return masks, normalized


def validate_bundle(rgb, depth, ids, info, reference_before, reference_after,
                    sim_time, camera_params, expected_optical, roles, environment):
    if plain(reference_before) != plain(reference_after):
        raise ValueError('Render changed while reading camera channels')
    t = reference_seconds(reference_after)
    if not np.isfinite(t) or abs(t - sim_time) > 1e-4:
        raise ValueError(f'Stale rendered frame: render={t}, physics={sim_time}')
    rgb, depth, ids = np.asarray(rgb), np.asarray(depth), np.asarray(ids)
    if rgb.ndim != 3 or rgb.shape[2] not in (3, 4) or rgb.dtype != np.uint8 or not np.any(rgb):
        raise ValueError('Empty or invalid RGB buffer')
    if depth.shape != rgb.shape[:2] or ids.shape != depth.shape:
        raise ValueError('Camera channel shapes disagree')
    if not np.any(np.isfinite(depth) & (depth > 0)):
        raise ValueError('Empty linear depth buffer')
    resolution = np.asarray(camera_params.get('renderProductResolution', [])).tolist()
    if resolution != [depth.shape[1], depth.shape[0]]:
        raise ValueError('Render camera resolution mismatch')
    # CameraParams uses the USD row-vector world-to-camera view matrix.
    view = np.asarray(camera_params['cameraViewTransform'], dtype=float).reshape(4, 4).T
    optical = np.linalg.inv(view) @ AXES
    if not np.allclose(optical, expected_optical, atol=1e-5, rtol=0):
        raise ValueError('Rendered camera pose does not match post-step mounted camera')
    masks, mapping = masks_from_ids(ids, info.get('idToLabels', {}), roles, environment)
    return {'rgb': np.array(rgb[:, :, :3], copy=True), 'depth': np.array(depth, dtype=np.float32, copy=True),
            'arrays': {'instance_ids': np.array(ids, copy=True),
                       'depth_valid': np.isfinite(depth) & (depth > 0), **masks},
            'metadata': {'version': VERSION, 'reference_time': plain(reference_after),
                         'render_time_s': t, 'camera_params': plain(camera_params),
                         'instance_id_to_path': mapping, 'role_paths': roles,
                         'environment_root': environment,
                         'visible_pixels': {k: int(v.sum()) for k, v in masks.items()},
                         'render_camera_world_optical': optical.tolist()}}


class SynchronizedCapture:
    def __init__(self, bridge, *, render_every_update=False):
        import omni.replicator.core as rep
        self.bridge = bridge
        self.camera = bridge.context.cameras['right_wrist']
        self.product = self.camera._sensor._hydra_texture.path
        self.render_schedule = {'render_every_update':render_every_update}
        if render_every_update:
            # Isaac 6's positive sensor tick rate skips same-time renders.
            # Only the separate capture camera uses autotrigger. Its original
            # mounted camera, optics, physics and observation interval stay as-is.
            self.render_schedule.update(capture_trigger(self.camera,bridge.context.stage))
        self.annotators = {}
        for key, name in [('rgb', 'rgb'), ('depth', 'distance_to_image_plane'),
                          ('instance', 'instance_id_segmentation_fast'),
                          ('reference', 'ReferenceTime'), ('params', 'CameraParams')]:
            a = rep.AnnotatorRegistry.get_annotator(name, device='cpu')
            a.attach(self.product);self.annotators[key] = a
        self.sequence = 0
        self.last_time = None

    def capture(self, before, after, max_renders=6):
        world = self.bridge.context.world
        t = float(world.current_time)
        self.camera.sync_pose(force=True)
        expected = np.array(after['camera_world']['right_wrist']) @ AXES
        expected[:3, 3] += self.bridge.roles.origin
        target = after['objects']['test_object']
        roles = {'target': target.get('raw_path', target['path']),
                 **{n: self.bridge.links[n].paths[0] for n in ('right_link27', 'right_link28')}}
        error = None; diagnostics = []
        for count in range(1, max_renders + 1):
            world.render()
            if abs(float(world.current_time) - t) > 1e-9:
                raise RuntimeError('Capture refresh advanced frozen physics time')
            # Drain one possible queued render before publishing the new pose.
            if count < 2:continue
            try:
                ref0 = copy.deepcopy(self.annotators['reference'].get_data())
                rgb = array_data(self.annotators['rgb'].get_data())
                depth = array_data(self.annotators['depth'].get_data()).squeeze()
                instance = self.annotators['instance'].get_data()
                if not isinstance(instance, dict):raise ValueError('Missing instance mapping')
                ids = np.array(instance['data'], copy=True).squeeze()
                info = copy.deepcopy(instance['info'])
                params = copy.deepcopy(self.annotators['params'].get_data())
                ref1 = copy.deepcopy(self.annotators['reference'].get_data())
                bundle = validate_bundle(rgb, depth, ids, info, ref0, ref1, t, params,
                                         expected, roles, self.bridge.environment_root)
            except (ValueError, KeyError, TypeError, np.linalg.LinAlgError) as exc:
                error = str(exc)
                diagnostics.append({'render_update':count,'error':error,
                    'physics_time':t,'reference':plain(locals().get('ref0')),
                    'rgb':buffer_info(locals().get('rgb')),'depth':buffer_info(locals().get('depth')),
                    'ids':buffer_info(locals().get('ids')),
                    'camera_params':plain(locals().get('params')),
                    'expected_camera_world_optical':expected.tolist(),
                    'before_physics':before,'pose_sync_state':after})
                continue
            final = self.bridge.snapshot()
            if not np.allclose(final['joint_positions'], after['joint_positions'], atol=1e-8, rtol=0):
                raise RuntimeError('Robot changed during render-only capture')
            for name, obj in after['objects'].items():
                if not np.allclose(obj['world'], final['objects'][name]['world'], atol=1e-8, rtol=0):
                    raise RuntimeError('Object changed during render-only capture')
            if self.last_time is not None and t <= self.last_time:
                raise RuntimeError('Capture reused a published physics tick')
            self.sequence += 1;self.last_time = t
            bundle['metadata'].update(sequence=self.sequence, render_product=str(self.product),
                render_schedule=self.render_schedule,
                render_only_updates=count, physics_time_before_render=t, physics_time_after_render=float(world.current_time),
                before_physics=before, pose_sync_state=after, after_render=final)
            c = self.camera
            c._latest_rgb, c._latest_depth = bundle['rgb'], bundle['depth']
            c.frame_time = bundle['metadata']['render_time_s']
            c.capture_metadata, c.capture_arrays = bundle['metadata'], bundle['arrays']
            return bundle['metadata']
        raise CaptureUnavailable('Synchronized camera capture failed: ' + str(error), diagnostics)


def validate_saved_capture(episode, row):
    """Independent file check; no inference from a writer's pass flag."""
    from scripts.curobo_arx.data_factory.common import sha
    paths = row.get('capture_files', {})
    if set(paths) != {'instance_ids', 'depth_valid', 'target', 'right_link27', 'right_link28'}:
        raise ValueError('Incomplete saved capture channels')
    arrays = {k: np.load(episode / p, allow_pickle=False) for k, p in paths.items()}
    if any(arrays[k].dtype != np.bool_ for k in ('depth_valid','target','right_link27','right_link28')):
        raise ValueError('Saved binary channel has wrong dtype')
    metadata = row['capture']
    if metadata['version'] != VERSION or metadata['environment_root'] != row['environment_root']:
        raise ValueError('Wrong capture identity')
    if abs(reference_seconds(metadata['reference_time']) - row['frame_time']) > 1e-4:
        raise ValueError('Saved render/state times disagree')
    if metadata['environment_root'] != row['camera_owner_root']:
        raise ValueError('Capture camera owner mismatch')
    target = row['snapshot']['objects']['test_object']
    if metadata['role_paths']['target'] != target.get('raw_path',target['path']):
        raise ValueError('Saved target mask belongs to another object')
    if metadata['physics_time_before_render'] != metadata['physics_time_after_render']:
        raise ValueError('Capture advanced physics')
    if abs(metadata['after_render']['sim_time'] - row['frame_time']) > 1e-4:
        raise ValueError('Capture state timestamp mismatch')
    expected = np.array(row['snapshot']['camera_world']['right_wrist']) @ AXES
    expected[:3, 3] += row['env_origin']
    if not np.allclose(expected, metadata['render_camera_world_optical'], atol=1e-5, rtol=0):
        raise ValueError('Saved camera and robot state mismatch')
    depth = np.load(episode / row['depth_relative'], allow_pickle=False)
    if not np.array_equal(arrays['depth_valid'], np.isfinite(depth) & (depth > 0)):
        raise ValueError('Saved depth validity disagrees')
    masks, _ = masks_from_ids(arrays['instance_ids'], metadata['instance_id_to_path'],
                             metadata['role_paths'], row['environment_root'])
    for name, mask in masks.items():
        if not np.array_equal(mask, arrays[name]):raise ValueError('Saved role mask disagrees with instance IDs')
    return {p: sha(episode / p) for p in paths.values()}


def audit_episode(episode):
    import json
    from pathlib import Path
    episode = Path(episode); count = 0; last = None
    for line in (episode/'samples.jsonl').read_text().splitlines():
        row = json.loads(line)
        validate_saved_capture(episode, row)
        metadata = row['capture']
        if last is not None and metadata['sequence'] != last + 1:
            raise ValueError('Missing captured tick')
        if metadata['physics_time_before_render'] != metadata['physics_time_after_render']:
            raise ValueError('Capture changed physics time')
        if abs(metadata['after_render']['sim_time'] - row['frame_time']) > 1e-4:
            raise ValueError('Capture state time mismatch')
        last = metadata['sequence'];count += 1
    if not count:raise ValueError('No captured observations')
    return {'pass':True,'frames':count,'version':VERSION}
