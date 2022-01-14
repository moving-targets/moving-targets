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

        # noinspection PyTypeChecker
        self._objective = self._cp.Minimize(0.0)
        """The model objective (by default, this is set to no objective by passing None)."""

    def _build_model(self) -> Any:
        return []

    def _solve_model(self) -> Optional:
        model = self._cp.Problem(self._objective, self.model)
        try:
            model.solve(solver=self.solver, **self.solver_args)
            return None if model.status in ['infeasible', 'unbounded'] else model
        except self._cp.error.SolverError:
            raise AssertionError(self._ERROR_MESSAGE + 'the given operation')

    def minimize(self, cost) -> Any:
        self._objective = self._cp.Minimize(cost)
        return self

    def add_constraints(self, constraints: Union[List, np.ndarray], name: Optional[str] = None) -> Any:
        self.model += constraints
        return self

    def add_variables(self, *keys: int, vtype: str, lb: Number, ub: Number, name: Optional[str] = None) -> np.ndarray:
        def _get_name(_name: Optional[str], _index: int) -> Optional[str]:
            return None if _name is None else f'{_name}_{_index}'

        def _recursive_addition(_keys: List[int], _name: Optional[str]):
            _key = _keys.pop(0)
            if len(_keys) == 0:
                return [self._cp.Variable((1,), name=_get_name(_name, i), **kw) for i in range(_key)]
            else:
                return [_recursive_addition(_keys=_keys.copy(), _name=_get_name(_name, i)) for i in range(_key)]

        assert vtype in self._VTYPES.keys(), self._ERROR_MESSAGE + f"vtype '{vtype}'"
        kw = self._VTYPES[vtype]
        var = np.array(_recursive_addition(_keys=list(keys), _name=name))
        self.add_constraints([v >= lb for v in var.flatten()])
        self.add_constraints([v <= ub for v in var.flatten()])
        return var

    def get_objective(self) -> Number:
        return self.solution.value

    def get_values(self, expressions: np.ndarray) -> np.ndarray:
        return np.reshape([v.value.squeeze() for v in expressions.flatten()], expressions.shape)

    def sum(self, vector: np.ndarray, aux: Optional[str] = None) -> Any:
        return self.aux(expressions=vector.sum(), aux_vtype=aux)

    def sqr(self, vector: np.ndarray, aux: Optional[str] = None) -> np.ndarray:
        return self.aux(expressions=vector ** 2, aux_vtype=aux)

    def abs(self, vector: np.ndarray, aux: Optional[str] = None) -> np.ndarray:
        expressions = np.reshape([self._cp.abs(v) for v in vector.flatten()], vector.shape)
        return self.aux(expressions=expressions, aux_vtype=aux)

    def log(self, vector: np.ndarray, aux: Optional[str] = None) -> np.ndarray:
        expressions = np.reshape([self._cp.log(v) for v in vector.flatten()], vector.shape)
        return self.aux(expressions=expressions, aux_vtype=aux)
