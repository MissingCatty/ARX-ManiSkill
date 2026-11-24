"""Run the fixed-orientation probe with a fixed-size native MPC goal buffer."""
import argparse
from pathlib import Path
import shutil
from unittest.mock import patch

from curobo.model_predictive_control import ModelPredictiveControl
from curobo.types import GoalToolPose
import arx_fixed_orientation_goalset_fixture as fixture
import arx_fixed_orientation_goalset_probe as probe


def pad_goals(goals, count):
    if goals.num_goalset == count:
        return goals
    if goals.num_goalset != 1:
        raise ValueError(f'Expected {count} native MPC goals, got {goals.num_goalset}')
    return GoalToolPose(tool_frames=goals.tool_frames,
                        position=goals.position.repeat(1, 1, 1, count, 1).contiguous(),
                        quaternion=goals.quaternion.repeat(1, 1, 1, count, 1).contiguous())


def run(output, calibration):
    setup_goal_buffer = ModelPredictiveControl._prepare_goal_buffer
    update_goal = ModelPredictiveControl.update_goal_tool_poses

    def prepare_buffer(solver, solve_state, goal_tool_poses, *args, **kwargs):
        return setup_goal_buffer(solver, solve_state, pad_goals(goal_tool_poses, solve_state.num_goalset),
                                 *args, **kwargs)

    def update_padded_goal(solver, goal_tool_poses, *args, **kwargs):
        count = solver.goal_registry_manager.goal_buffer.link_goal_poses.num_goalset
        return update_goal(solver, pad_goals(goal_tool_poses, count), *args, **kwargs)

    # The vendored SDK configures max_goalset during setup but the initial FK
    # supplies one pose. Allocate the full shape before capturing CUDA graphs.
    with patch.multiple(ModelPredictiveControl, _prepare_goal_buffer=prepare_buffer,
                        update_goal_tool_poses=update_padded_goal):
        (fixture if calibration else probe).run(output)
    for path in (Path(__file__), Path(probe.__file__)):
        shutil.copy2(path, output / path.name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--calibration', action='store_true')
    args = parser.parse_args()
    run(args.output.resolve(), args.calibration)
