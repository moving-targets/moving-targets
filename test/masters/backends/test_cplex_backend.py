import os
from typing import List, Any

import pytest

from moving_targets.masters.backends import Backend, CplexBackend
from test.masters.backends.test_backend import TestBackend


@pytest.mark.skipif(os.getenv('GITHUB_ACTIONS') == 'true', reason='No solver in Github Actions')
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
