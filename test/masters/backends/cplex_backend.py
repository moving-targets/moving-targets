from typing import List, Any

from moving_targets.masters.backends import Backend, CplexBackend
from test.masters.backends.abstract_backend import TestBackend


class TestCplexBackend(TestBackend):
    @classmethod
    def _backend(cls) -> Backend:
        return CplexBackend()

    @classmethod
    def _unsupported(cls) -> List[str]:
        return ['log', 'min', 'max', 'sqrt', 'divide', 'norm_0', 'norm_inf']

    @classmethod
    def _get_name(cls, variable: Any) -> str:
        return str(variable)
