from typing import List, Any

from moving_targets.masters.backends import Backend, GurobiBackend
from test.masters.backends.abstract_backend import TestBackend


class TestGurobiBackend(TestBackend):
    @classmethod
    def _backend(cls) -> Backend:
        return GurobiBackend()

    @classmethod
    def _unsupported(cls) -> List[str]:
        return []

    @classmethod
    def _get_name(cls, variable: Any) -> str:
        return str(variable)[12:-13].rstrip(']').replace('[', '_').replace(',', '_')
