from typing import List

from moving_targets.masters.backends import Backend, GurobiBackend
from test.masters.abstract import TestBackend


class TestGurobiBackend(TestBackend):
    def _backend(self) -> Backend:
        return GurobiBackend()

    def _unsupported(self) -> List[str]:
        return []
