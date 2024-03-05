from typing import List, Any

from moving_targets.masters.backends import Backend, NumpyBackend
from test.masters.backends.test_backend import TestBackend


class TestNumpyBackend(TestBackend):
    @classmethod
    def _backend(cls) -> Backend:
        return NumpyBackend()

    @classmethod
    def _unsupported(cls) -> List[str]:
        return ['objectives', 'variables']

    @classmethod
    def _get_name(cls, variable: Any) -> str:
        return ''
