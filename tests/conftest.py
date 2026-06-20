"""Test configuration — mock heavy dependencies that aren't installed.

torch is required by pgr_reward.py and step_pgr_trainer.py at import time
but is too heavy to install for unit tests. We mock it with a minimal stub
that satisfies scipy's issubclass checks and transformers' find_spec checks.

trl and accelerate are also not installed; we mock them since only the
StepLevelGRPOTrainer class needs them (not the utility functions we test).
"""

import sys
import os
import importlib.machinery
from unittest.mock import MagicMock
from types import ModuleType

# Add project root to sys.path so tests can import pgr_reward, step_pgr_trainer
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def _make_mock_module(name):
    """Create a MagicMock that also has a valid __spec__ for importlib checks."""
    mod = MagicMock()
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
    mod.__version__ = "0.0.0"
    mod.__file__ = f"/fake/{name}/__init__.py"
    return mod


def _create_torch_mock():
    """Create a torch mock module with a real Tensor class and proper __spec__."""
    torch_mod = ModuleType("torch")
    torch_mod.__spec__ = importlib.machinery.ModuleSpec("torch", None)
    torch_mod.__version__ = "2.4.0"
    torch_mod.__file__ = "/fake/torch/__init__.py"
    torch_mod.Tensor = type("Tensor", (), {})
    torch_mod.device = MagicMock
    torch_mod.dtype = MagicMock
    torch_mod.float32 = "float32"
    torch_mod.bfloat16 = "bfloat16"
    torch_mod.zeros = MagicMock(return_value=MagicMock())
    torch_mod.tensor = MagicMock(return_value=MagicMock())
    torch_mod.exp = MagicMock(return_value=MagicMock())
    torch_mod.full = MagicMock(return_value=MagicMock())
    torch_mod.inference_mode = MagicMock(return_value=lambda f: f)
    torch_mod.no_grad = MagicMock(return_value=lambda f: f)
    torch_mod.cuda = MagicMock()
    torch_mod.cuda.is_available = MagicMock(return_value=False)
    return torch_mod


# Install torch mock before any test module imports
if "torch" not in sys.modules:
    sys.modules["torch"] = _create_torch_mock()

# Mock sentence_transformers (downloads models at import)
if "sentence_transformers" not in sys.modules:
    sys.modules["sentence_transformers"] = _make_mock_module("sentence_transformers")

# Mock TRL and accelerate (not installed, only needed for StepLevelGRPOTrainer class)
_mock_packages = [
    "trl", "trl.data_utils", "trl.models", "trl.trainer", "trl.trainer.utils",
    "accelerate", "accelerate.utils", "accelerate.utils.other",
]
for mod_name in _mock_packages:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = _make_mock_module(mod_name)
