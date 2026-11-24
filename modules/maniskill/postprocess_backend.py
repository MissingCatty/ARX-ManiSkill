"""Resident backend entry for native ZeroGrasp postprocessing benchmark phases."""
import argparse
from pathlib import Path

import resident_backend as service


class NativePostNeural(service.NeuralWorker):
    def request(self, directory, **kwargs):
        result = super().request(directory, **kwargs)
        import numpy as np
        from postprocess_pipeline import processed_rows
        rows, files = processed_rows(Path(directory))
        np.save(Path(directory) / 'postprocessed_grasps.npy', rows, allow_pickle=False)
        service.save(Path(directory) / 'postprocessing_manifest.json', {
            'raw_candidates_archived': result['count'], 'native_postprocessed_candidates': len(rows),
            'source_files': list(map(str, files)),
            'selection_input': 'Native *.grasp.npy, width/depth corrections, native collision filtering and native NMS',
            'extra_geometry_filters': [], 'custom_top_k_cap': None,
            'candidate_protocol': 'zerograsp_native_post_no_custom_geometry_v1'})
        result['native_postprocessed_count'] = len(rows)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=['zerograsp', 'curobo'], required=True)
    parser.add_argument('--socket', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--warmup-trial', type=Path, required=True)
    args = parser.parse_args()
    if args.kind == 'zerograsp':
        service.NeuralWorker = NativePostNeural
    else:
        import resident_planner
        from postprocess_planner import PostprocessPlanner
        resident_planner.GraspCLPlanner = PostprocessPlanner
    service.serve(args)


if __name__ == '__main__':
    main()
