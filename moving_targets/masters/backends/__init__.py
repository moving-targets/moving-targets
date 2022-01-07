"""Moving Targets Backends for Masters."""

from moving_targets.masters.backends.backend import Backend
from moving_targets.masters.backends.cplex_backend import CplexBackend
from moving_targets.masters.backends.cvxpy_backend import CvxpyBackend
from moving_targets.masters.backends.gurobi_backend import GurobiBackend
from moving_targets.masters.backends.numpy_backend import NumpyBackend

_BACKEND_ALIASES: dict = {
    'cvxpy': CvxpyBackend,
    'cplex': CplexBackend,
    'gurobi': GurobiBackend,
}


def get_backend(alias: str, **backend_kwargs) -> Backend:
    backend_type = _BACKEND_ALIASES.get(alias)
    assert backend_type is not None, f"{alias} is not a valid alias for a backend"
    return backend_type(**backend_kwargs)
