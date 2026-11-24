"""Screen ARX feedback goals over the gripper closure stroke before admission."""
import numpy as np


class ClosureEnvelopeChecker:
    def __init__(self, checker, update):
        self.checker, self.update = checker, update
        self.checked_states = 0
        self.rejected_states = 0

    def validate(self, positions):
        result = None
        try:
            for fraction in np.linspace(0., 1., 9):
                values = {name: lo + fraction * (hi - lo)
                          for name, (lo, hi) in self.update.limits.items()}
                self.update.apply([self.checker.kinematics], values)
                valid = self.checker.validate(positions)
                result = valid.clone() if result is None else result & valid
                self.checked_states += int(valid.numel())
                self.rejected_states += int((~valid).sum())
                if not result.any():
                    break
            return result
        finally:
            self.update.apply([self.checker.kinematics], self.update.nominal)
