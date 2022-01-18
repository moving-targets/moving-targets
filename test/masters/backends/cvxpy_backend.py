from typing import List

from moving_targets.masters.backends import Backend, CvxpyBackend
from test.masters.backends.abstract_backend import TestBackend


class TestCvxpyBackend(TestBackend):
    def _backend(self) -> Backend:
        return CvxpyBackend()

    def _unsupported(self) -> List[str]:
        return []
