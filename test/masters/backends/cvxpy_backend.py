from typing import List, Any

from moving_targets.masters.backends import Backend, CvxpyBackend
from test.masters.backends.abstract_backend import TestBackend


class TestCvxpyBackend(TestBackend):
    @classmethod
    def _backend(cls) -> Backend:
        return CvxpyBackend()

    @classmethod
    def _unsupported(cls) -> List[str]:
        return ['sqrt', 'var']

    @classmethod
    def _get_name(cls, variable: Any) -> str:
        return str(variable)
