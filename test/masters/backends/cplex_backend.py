from typing import List

from moving_targets.masters.backends import CplexBackend, Backend
from test.masters.backends.abstract_backend import TestBackend


class TestCplexBackend(TestBackend):
    def _backend(self) -> Backend:
        return CplexBackend()

    def _unsupported(self) -> List[str]:
        # cplex does not support logarithms
        return ['reversed_bce', 'reversed_cce', 'symmetric_bce', 'symmetric_cce']
