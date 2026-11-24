"""Original ARX selection equations for a fresh-input geometry fallback."""
import numpy as np


def select_original(rows, camera, center, start, ik, path, closed_checker,
                    contact_height=.015, contact_radius=.065):
    axes = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)
    for index, row in enumerate(rows):
        rg = camera[:3, :3] @ row[4:13].reshape(3, 3)
        surface = camera[:3, :3] @ row[13:16] + camera[:3, 3]
        contact = surface + row[3] * rg[:, 0]
        if rg[2, 0] > -.35 or np.linalg.norm(contact - center) > contact_radius:
            continue
        raise_m = max(0., float(center[2] + contact_height - contact[2]))
        if raise_m > .030:
            continue
        contact[2] += raise_m
        for flip in [np.eye(3), np.diag([1, -1, -1])]:
            grasp = np.eye(4); grasp[:3, :3] = rg @ flip @ axes
            grasp[:3, 3] = contact - .040 * rg[:, 0]
            pre = grasp.copy(); pre[:3, 3] -= .08 * rg[:, 0]
            lift = grasp.copy(); lift[2, 3] += .12
            qg = ik(grasp, start)
            if qg is None or not closed_checker.validate(qg[None, None]).all():
                continue
            qp = ik(pre, qg); ql = ik(lift, qg)
            if qp is None or ql is None:
                continue
            route = path(start, qp)
            if route is None or path(qp, qg) is None:
                continue
            return {'index': index, 'score': float(row[0]), 'width_m': float(row[1]),
                    'depth_m': float(row[3]), 'table_clearance_raise_m': raise_m,
                    'network_row': row.tolist(), 'pregrasp_world': pre.tolist(),
                    'grasp_world': grasp.tolist(), 'lift_world': lift.tolist(),
                    'target_contact_world': contact.tolist(), 'q_pre': qp.tolist(),
                    'q_grasp': qg.tolist(), 'q_lift': ql.tolist()}, route
    return None, None
