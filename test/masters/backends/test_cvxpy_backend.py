from typing import List, Any

from moving_targets.masters.backends import Backend, CvxpyBackend
from test.masters.backends.test_backend import TestBackend


class TestCvxpyBackend(TestBackend):
    @classmethod
    def _backend(cls) -> Backend:
        return CvxpyBackend()

    @classmethod
    def _unsupported(cls) -> List[str]:
        return ['is_greater', 'is_less', 'min', 'max', 'sqrt', 'var', 'cov', 'norm_0', 'norm_inf']

    @classmethod
    def _get_name(cls, variable: Any) -> str:
        return str(variable)
