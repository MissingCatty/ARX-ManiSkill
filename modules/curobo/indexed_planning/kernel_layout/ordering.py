"""Restrict the transformation to the two original task-index expressions."""
import ast
from copy import deepcopy


def reorder_source(source):
    module = ast.parse(source)
    if len(module.body) != 1 or not isinstance(module.body[0], ast.FunctionDef):
        raise ValueError('Expected exactly the original collision kernel definition')
    original = deepcopy(module);function = module.body[0]
    changes = {'sph_flat_idx': ('tid / max_n_obs', 'tid % (batch_size * horizon * num_spheres)'),
               'obs_local_idx': ('tid - sph_flat_idx * max_n_obs', 'tid / (batch_size * horizon * num_spheres)')}
    changed = []
    for statement in function.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
            name = statement.targets[0].id
            if name in changes:
                before, after = changes[name]
                if ast.dump(statement.value) != ast.dump(ast.parse(before, mode='eval').body):
                    raise ValueError('Original task-index expression changed: '+name)
                statement.value = ast.parse(after, mode='eval').body;changed.append(name)
    if sorted(changed) != sorted(changes): raise ValueError('Original task indices are missing or duplicated')
    restored = deepcopy(module)
    for statement in restored.body[0].body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
            name = statement.targets[0].id
            if name in changes: statement.value = ast.parse(changes[name][0], mode='eval').body
    if ast.dump(restored) != ast.dump(original): raise ValueError('Transformation changed collision mathematics')
    function.name = 'obstacle_major_collision';function.decorator_list = []
    return ast.unparse(ast.fix_missing_locations(module))+'\n'


def task_pair(tid, spheres, obstacles, *, obstacle_major):
    if spheres <= 0 or obstacles <= 0 or not 0 <= tid < spheres*obstacles:
        raise ValueError('Task index outside positive collision grid')
    return (tid % spheres, tid//spheres) if obstacle_major else (tid//obstacles, tid % obstacles)
