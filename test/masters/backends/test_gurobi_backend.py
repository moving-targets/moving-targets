import os
from typing import List, Any

import pytest

from moving_targets.masters.backends import Backend, GurobiBackend
from test.masters.backends.test_backend import TestBackend


@pytest.mark.skipif(os.getenv('GITHUB_ACTIONS') == 'true', reason='No solver in Github Actions')
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
