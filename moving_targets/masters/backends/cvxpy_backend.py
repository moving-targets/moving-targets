from typing import Any, Union, List, Optional, Dict

import numpy as np

from moving_targets.masters.backends import Backend
from moving_targets.util.errors import MissingDependencyError, BackendError


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
            raise BackendError(unsupported='the given operation')

    def minimize(self, cost) -> Any:
        self._objective = self._cp.Minimize(cost)
        return self

    def add_constraints(self, constraints: Union[List, np.ndarray], name: Optional[str] = None) -> Any:
        self.model += constraints
        return self

    def add_variable(self, vtype: str, lb: float, ub: float, name: Optional[str] = None) -> Any:
        if vtype not in self._VTYPES.keys():
            raise BackendError(unsupported=f"vtype '{vtype}'")
        var = self._cp.Variable(1, name=name, **self._VTYPES[vtype])
        self.add_constraints([var[0] >= lb, var[0] <= ub])
        return var

    def add_variables(self, *keys: int, vtype: str, lb: float, ub: float, name: Optional[str] = None) -> np.ndarray:
        def _recursive_addition(_keys: List[int], _name: Optional[str]):
            key, name_fn = _keys.pop(0), lambda i: None if _name is None else f'{_name}_{i}'
            if len(_keys) == 0:
                return [self._cp.Variable((1,), name=name_fn(i), **kw) for i in range(key)]
            return [_recursive_addition(_keys=_keys.copy(), _name=name_fn(i)) for i in range(key)]

        if vtype not in self._VTYPES.keys():
            raise BackendError(unsupported=f"vtype '{vtype}'")
        kw = self._VTYPES[vtype]
        var = np.array(_recursive_addition(_keys=list(keys), _name=name))
        self.add_constraints([v[0] >= lb for v in var.flatten()] + [v[0] <= ub for v in var.flatten()])
        return var

    def get_objective(self) -> float:
        return self.solution.value

    def get_values(self, expressions: np.ndarray) -> np.ndarray:
        return np.reshape([v.value.squeeze() for v in expressions.flatten()], expressions.shape)

    def sum(self, a: np.ndarray, aux: Optional[str] = None) -> Any:
        return self.aux(expressions=a.sum(), aux_vtype=aux)

    def square(self, a: np.ndarray, aux: Optional[str] = None) -> np.ndarray:
        return self.aux(expressions=a ** 2, aux_vtype=aux)

    def abs(self, a: np.ndarray, aux: Optional[str] = None) -> np.ndarray:
        expressions = np.reshape([self._cp.abs(v) for v in a.flatten()], a.shape)
        return self.aux(expressions=expressions, aux_vtype=aux)

    def log(self, a: np.ndarray, aux: Optional[str] = None) -> np.ndarray:
        expressions = np.reshape([self._cp.log(v) for v in a.flatten()], a.shape)
        return self.aux(expressions=expressions, aux_vtype=aux)
