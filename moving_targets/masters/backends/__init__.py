"""Moving Targets Backends for Masters."""

from moving_targets.masters.backends.backend import Backend
from moving_targets.masters.backends.cplex_backend import CplexBackend
from moving_targets.masters.backends.cvxpy_backend import CvxpyBackend
from moving_targets.masters.backends.gurobi_backend import GurobiBackend
from moving_targets.masters.backends.numpy_backend import NumpyBackend

aliases: dict = {
    'cvxpy': CvxpyBackend,
    'cplex': CplexBackend,
    'gurobi': GurobiBackend
}
