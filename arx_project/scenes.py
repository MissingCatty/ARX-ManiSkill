"""The single scene collection selected by the project configuration."""
from pathlib import Path

from .layout import ROOT, read_json


def current_dataset():
    return Path(read_json(ROOT / 'configs/experiments/scene_dataset.json')['dataset'])


def current_schedule():
    return read_json(current_dataset() / 'manifest.json')['schedule']
