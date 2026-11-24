"""Relocate a frozen ARX planning model, updating only its measured state."""
import copy
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml


def relocate(source, destination, snapshot, *, measured_gripper=True):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    data = yaml.safe_load((source / 'arx.yml').read_text())
    original = copy.deepcopy(data)
    spec = data['robot_cfg']['kinematics']
    urdf = source / 'arx.urdf'
    shutil.copy2(urdf, destination / 'arx.urdf')
    spec['urdf_path'] = str((destination / 'arx.urdf').resolve())
    positions = dict(zip(snapshot['joint_names'], snapshot['joint_positions']))
    raw_positions = positions.copy()
    for joint in ET.parse(urdf).getroot().findall('joint'):
        limit = joint.find('limit')
        name = joint.attrib['name']
        if limit is not None and name in positions:
            positions[name] = min(max(positions[name], float(limit.attrib['lower'])), float(limit.attrib['upper']))
    state_clamps = {name: positions[name]-value for name,value in raw_positions.items() if positions[name]!=value}
    assert all(abs(value)<1e-4 for value in state_clamps.values()),state_clamps
    if measured_gripper:
        spec['lock_joints'].update({name: positions[name] for name in spec['lock_joints']})
    cspace = spec['cspace']
    cspace['default_joint_position'] = [positions[name] for name in cspace['joint_names']]
    # No geometry, collision settings, transforms, bounds or dynamic limits change.
    normalized = copy.deepcopy(data)
    normalized['robot_cfg']['kinematics']['urdf_path'] = original['robot_cfg']['kinematics']['urdf_path']
    normalized['robot_cfg']['kinematics']['lock_joints'] = original['robot_cfg']['kinematics']['lock_joints']
    normalized['robot_cfg']['kinematics']['cspace']['default_joint_position'] = original['robot_cfg']['kinematics']['cspace']['default_joint_position']
    assert normalized == original
    (destination / 'arx.yml').write_text(yaml.safe_dump(data, sort_keys=False))
    manifest = source / 'model_manifest.json'
    if manifest.exists():
        shutil.copy2(manifest, destination / manifest.name)
    audit = {'source': str(source), 'urdf_sha256': hashlib.sha256(urdf.read_bytes()).hexdigest(),
             'source_yaml_sha256': hashlib.sha256((source / 'arx.yml').read_bytes()).hexdigest(),
             'mechanical_and_collision_configuration_equal': True,
             'changes': ['absolute URDF location', 'measured/default joint state'],
             'measured_state_limit_clamps': state_clamps,
             'historical_live_physics_equivalence': 'unverified; source is an archived planning model, checked against current live FK separately'}
    (destination / 'relocation_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
