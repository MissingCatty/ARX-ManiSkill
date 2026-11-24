"""Original static ARX IK plus native convex-decomposition intersection checks."""
import argparse
from pathlib import Path
from scripts.curobo_arx.data_factory.common import read, write, sha
from .advisor36_static_teacher import Teacher as Original
from .advisor37_surfaces import SurfaceChecker


class Teacher(Original):
    def handle(self, request):
        if request['action'] != 'static_surface_check':
            return super().handle(request)
        path = request['geometry']
        key = (path, sha(path))
        if getattr(self, 'surface_key', None) != key:
            self.surface_checker = SurfaceChecker(read(path))
            self.surface_key = key
        result = self.surface_checker.check(request['snapshot'])
        write(Path(request['output']) / 'static_surface_receipt.json', result)
        return result


if __name__ == '__main__':
    from scripts.curobo_arx.seed_augmentation import worker
    p = argparse.ArgumentParser()
    p.add_argument('--directory', required=True)
    p.add_argument('--registration', required=True)
    a = p.parse_args()
    worker.Teacher = Teacher
    worker.serve(a.directory, read(a.registration))
