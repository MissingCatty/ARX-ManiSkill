"""Fast JSON reads with equivalent archived-path handling for actual paths."""
import json

from .layout import canonical_path, relocated, relocation_aliases


def relocate_paths(value, aliases):
    relative = tuple((source, target) for source, target in aliases if not source.startswith('/'))

    def visit(item):
        if isinstance(item, str):
            # Current metadata keys, labels, protocols and scores are not paths.
            if item.startswith('/'):
                return relocated(item, aliases)
            if relative:
                return relocated(item, aliases)
            return item
        if isinstance(item, list):
            return [visit(child) for child in item]
        if isinstance(item, dict):
            return {visit(key): visit(child) for key, child in item.items()}
        return item

    return visit(value)


def read_json(path):
    value = json.loads(canonical_path(path).read_text())
    return relocate_paths(value, relocation_aliases())
