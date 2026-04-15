"""Fix 4: Test that default residual_scale is 0.05 (not 0.2).

residual_scale=0.2 gives ±0.2 rad/step, which is 3x-40x the base trajectory
step size (0.005-0.064 rad/step). This allows the policy to completely destroy
the scripted trajectory. Reducing to 0.05 limits corrections to ±0.05 rad/step.
"""

import importlib
import sys
import pytest


class TestResidualScaleDefault:
    """Both training scripts must default to residual_scale=0.05."""

    def test_train_baselines_default(self):
        """train_baselines.py --residual-scale default must be 0.05."""
        # Import the module and inspect its argparse
        import pta.scripts.train_baselines as mod

        # Find the parser by looking for create_parser or build it from main
        # The module uses argparse.ArgumentParser directly
        import argparse

        # Re-parse the module source to find the default
        import inspect

        source = inspect.getsource(mod)
        # Check that 0.05 appears as default for residual-scale
        assert "default=0.05" in source or "default = 0.05" in source, (
            "train_baselines.py residual-scale default is not 0.05. "
            f"Found source containing 'residual' near: "
            f"{[l.strip() for l in source.split(chr(10)) if 'residual' in l.lower()]}"
        )

    def test_train_m7_default(self):
        """train_m7.py --residual-scale default must be 0.05."""
        import pta.scripts.train_m7 as mod
        import inspect

        source = inspect.getsource(mod)
        assert "default=0.05" in source or "default = 0.05" in source, (
            "train_m7.py residual-scale default is not 0.05. "
            f"Found source containing 'residual' near: "
            f"{[l.strip() for l in source.split(chr(10)) if 'residual' in l.lower()]}"
        )

    def test_no_0_2_default(self):
        """Neither script should have default=0.2 for residual-scale."""
        import pta.scripts.train_baselines as mod1
        import pta.scripts.train_m7 as mod2
        import inspect

        for name, mod in [("train_baselines", mod1), ("train_m7", mod2)]:
            source = inspect.getsource(mod)
            lines = [
                l.strip()
                for l in source.split("\n")
                if "residual" in l.lower() and "default" in l.lower()
            ]
            for line in lines:
                assert "0.2" not in line or "0.05" in line, (
                    f"{name}.py still has default=0.2 for residual-scale: {line}"
                )
