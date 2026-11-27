"""Stable candidate identities and exact-width grouping, without GPU imports."""
import math


def width_groups(jobs, capacity):
    if capacity not in (4, 8):
        raise ValueError('Parallel trajectory capacity must be 4 or 8')
    groups = {}
    seen = set()
    for job in jobs:
        index, width = job['index'], float(job['width'])
        if index in seen or not math.isfinite(width) or not 0 < width <= .088:
            raise ValueError('Duplicate candidate or invalid opening')
        seen.add(index)
        # No rounding: even nearby openings retain their own collision model.
        groups.setdefault(width, []).append(job)
    return [values[i:i + capacity] for values in groups.values()
            for i in range(0, len(values), capacity)]


def padded_indices(count, capacity):
    if not 0 < count <= capacity or capacity not in (4, 8):
        raise ValueError('Invalid real trajectory rows')
    return list(range(count)) + [0] * (capacity - count)
