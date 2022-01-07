from typing import Any, Union, List, Optional, Dict

import numpy as np

from moving_targets.masters.backends import Backend
from moving_targets.util.errors import MissingDependencyError
from moving_targets.util.typing import Number


class CvxpyBackend(Backend):
    """`Backend` implementation for the Cvxpy Solver."""

    _VTYPES: Dict[str, Any] = {
        'binary': dict(boolean=True),
        'integer': dict(integer=True),
        'continuous': dict()
    }
    """Accepted vtypes with mapping."""

    def __init__(self, solver: Optional[str] = None, **solver_args):
        """
        :param solver:
            The name of the solver (e.g., SCS, ...).

        :param solver_args:
            Parameters of the solver to be passed to the `model.solve()` function.
        """
        super(CvxpyBackend, self).__init__()

        try:
            import cvxpy
            self._cp = cvxpy
            """The lazily imported cvxpy instance."""
        except ModuleNotFoundError:
            raise MissingDependencyError(package='cvxpy')

        self.solver: Optional[str] = solver
        """The name of the solver."""

        self.solver_args: Dict[str, Any] = solver_args
        """Parameters of the solver to be passed to the `model.solve()` function."""

    def _build_model(self) -> Any:
        return []

    def _solve_model(self) -> Optional:
        try:
            self.model.solve(solver=self.solver, **self.solver_args)
            return None if self.model.status in ['infeasible', 'unbounded'] else self.model
        except self._cp.error.SolverError:
            raise AssertionError(self._ERROR_MESSAGE + 'the given operation')

    def minimize(self, cost) -> Any:
        objective = self._cp.Minimize(cost)
        self.model = self._cp.Problem(objective, self.model)
        return self

    def add_constraints(self, constraints: Union[List, np.ndarray], name: Optional[str] = None) -> Any:
        self.model += constraints
        return self

    def add_variables(self,
                      *keys: int,
                      vtype: str,
                      lb: Optional[Number] = None,
                      ub: Optional[Number] = None,
                      name: Optional[str] = None) -> np.ndarray:
        def _recursive_addition(_keys: List[int], _name: str):
            _key = _keys.pop(0)
            if len(_keys) == 0:
                return [self._cp.Variable((1,), name=f'{_name}_{i}', **kw) for i in range(_key)]
            else:
                return [_recursive_addition(_keys=_keys.copy(), _name=f'{_name}_{i}') for i in range(_key)]

        assert vtype in self._VTYPES.keys(), self._ERROR_MESSAGE + f"vtype '{vtype}'"
        kw = self._VTYPES[vtype]
        var = np.array(_recursive_addition(_keys=list(keys), _name=name))
        self.add_constraints([] if lb is None else [v >= lb for v in var.flatten()])
        self.add_constraints([] if ub is None else [v <= ub for v in var.flatten()])
        return var

    def get_objective(self) -> Number:
        return self.solution.value

    def get_values(self, expressions: np.ndarray) -> np.ndarray:
        return np.reshape([v.value.squeeze() for v in expressions.flatten()], expressions.shape)

    def sum(self, vector: np.ndarray) -> Any:
        return vector.sum()

    def sqr(self, vector: np.ndarray) -> np.ndarray:
        return vector ** 2

    def abs(self, vector: np.ndarray) -> np.ndarray:
        return np.reshape([self._cp.abs(v) for v in vector.flatten()], vector.shape)

    def log(self, vector: np.ndarray) -> np.ndarray:
        return np.reshape([self._cp.log(v) for v in vector.flatten()], vector.shape)
