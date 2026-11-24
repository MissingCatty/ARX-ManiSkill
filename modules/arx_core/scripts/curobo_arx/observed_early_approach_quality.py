"""Additional observed-target/CAD quality feature; no planner or physical model edits."""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp
from grasp_quality_features import observed

ROOT = Path(__file__).resolve().parents[2]
PROXY = ROOT / 'runs/research_contract_20261019/fixed_gripper_quality_proxy_2mm.npz'


class ObservedEarlyApproachQuality:
    def __init__(self, folder):
        folder = Path(folder)
        self.meta = json.loads(PROXY.with_suffix('.json').read_text())
        assert hashlib.sha256(PROXY.read_bytes()).hexdigest() == self.meta['data_sha256']
        with np.load(PROXY) as data:
            self.trees = {name: cKDTree(data[name].copy()) for name in
                          ('right_link26', 'right_link27', 'right_link28')}
        points = observed(folder, max_points=None)
        # Fixed resolution and seed, independent of physical outcome labels.
        _, ids = np.unique(np.floor(points / .002).astype(np.int64), axis=0, return_index=True)
        points = points[np.sort(ids)]
        if len(points) > 2000:
            points = points[np.random.default_rng(0).choice(len(points), 2000, replace=False)]
        self.points = points
        snap = json.loads((folder / 'execution_initial_snapshot.json').read_text())
        joints = dict(zip(snap['joint_names'], snap['joint_positions']))
        self.open_state = {link: float(joints[joint]) for link, joint in
                           (('right_link27', 'right_joint27'), ('right_link28', 'right_joint28'))}

    def evaluate(self, grasp, pregrasp):
        grasp, pregrasp = np.asarray(grasp), np.asarray(pregrasp)
        alpha = np.linspace(0, 1, 17)
        rotations = Slerp([0, 1], Rotation.from_matrix(np.stack([pregrasp[:3, :3], grasp[:3, :3]])))(alpha).as_matrix()
        translations = pregrasp[:3, 3][None] * (1 - alpha[:, None]) + grasp[:3, 3][None] * alpha[:, None]
        touched = {link: np.zeros(len(self.points), dtype=bool) for link in self.trees}
        early_touched = {link: np.zeros(len(self.points), dtype=bool) for link in self.trees}
        sample_profile = []
        goal_counts = {}
        palm_lower = float('inf')
        for rotation, translation in zip(rotations, translations):
            remaining = float(np.linalg.norm(translation - grasp[:3, 3]))
            counts = {}
            local = (self.points - translation) @ rotation
            for link, tree in self.trees.items():
                shift = np.asarray(self.meta['motion_axis_by_link'][link]) * self.open_state.get(link, 0.)
                distance = tree.query(local - shift, workers=1)[0]
                mask = distance <= self.meta['sample_cover_radius_m']
                touched[link] |= mask
                if remaining >= .010 - 1e-10:
                    early_touched[link] |= mask
                counts[link] = int(mask.sum())
                goal_counts[link] = int(mask.sum())
                if link == 'right_link26':
                    palm_lower = min(palm_lower, float(distance.min() - self.meta['sample_cover_radius_m']))
            sample_profile.append(dict(remaining_m=remaining, contact_point_counts=counts))
        full_mask = touched['right_link27'] | touched['right_link28']
        finger_mask = early_touched['right_link27'] | early_touched['right_link28']
        fraction = float(finger_mask.mean())
        return dict(terminal_contact_allowance_m=.010, full_sweep_fraction=float(full_mask.mean()), sample_profile=sample_profile, observed_points=len(self.points), nominal_approach_samples=17,
                    measured_open_finger_state_m=self.open_state,
                    finger_sweep_point_count=int(finger_mask.sum()),
                    finger_sweep_point_fraction=fraction,
                    potential_open_finger_contact=fraction >= .01,
                    goal_potential_contact_points=goal_counts,
                    observed_palm_gap_lower_estimate_m=palm_lower,
                    proxy_data_sha256=self.meta['data_sha256'],
                    scope='Observed-target/unchanged-CAD feature on nominal straight/Slerp approach only; not full path/hidden surface/force closure certification or a collision margin change.')
