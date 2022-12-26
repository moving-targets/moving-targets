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

        self._objective: Optional = None
        """The model objective (by default, this is set to no objective by passing None)."""

    def _build_model(self) -> Any:
        return []

    def _solve_model(self) -> Optional:
        # noinspection PyTypeChecker
        model = self._cp.Problem(self._cp.Minimize(0.0) if self._objective is None else self._objective, self.model)
        try:
            model.solve(solver=self.solver, **self.solver_args)
            return None if model.status in ['infeasible', 'unbounded'] else model
        except self._cp.error.SolverError:
            raise BackendError(unsupported='the given operation')

    def minimize(self, cost) -> Any:
        self._objective = self._cp.Minimize(cost)
        return self

    def maximize(self, cost) -> Any:
        self._objective = self._cp.Maximize(cost)
        return self

    def get_objective(self) -> float:
        return self.solution.value

    def get_values(self, expressions: np.ndarray) -> np.ndarray:
        return np.reshape([v.value.squeeze() for v in expressions.flatten()], expressions.shape)

    def add_variable(self, vtype: str, lb: float, ub: float, name: Optional[str] = None) -> Any:
        if vtype not in self._VTYPES.keys():
            raise BackendError(unsupported=f"vtype '{vtype}'")
        var = self._cp.Variable(shape=(), name=name, **self._VTYPES[vtype])
        # noinspection PyTypeChecker
        self.add_constraints([var >= lb, var <= ub])
        return var

    def add_variables(self, *keys: int, vtype: str, lb: float, ub: float, name: Optional[str] = None) -> np.ndarray:
        def _recursive_addition(_keys: List[int], _name: Optional[str]):
            key, name_fn = _keys.pop(0), lambda i: None if _name is None else f'{_name}_{i}'
            if len(_keys) == 0:
                return [self._cp.Variable(shape=(), name=name_fn(i), **kw) for i in range(key)]
            return [_recursive_addition(_keys=_keys.copy(), _name=name_fn(i)) for i in range(key)]

        if len(keys) == 0:
            # if no keys are passed, builds a single variable then reshape it into a zero-dimensional numpy array
            var = self.add_variable(vtype=vtype, lb=lb, ub=ub, name=name)
            return np.reshape(var, ())
        elif vtype not in self._VTYPES.keys():
            raise BackendError(unsupported=f"vtype '{vtype}'")
        kw = self._VTYPES[vtype]
        var = np.array(_recursive_addition(_keys=list(keys), _name=name))
        self.add_constraints([v >= lb for v in var.flatten()] + [v <= ub for v in var.flatten()])
        return var

    def add_constraint(self, constraint, name: Optional[str] = None) -> Any:
        if name is not None:
            self._LOGGER.warning(f"name='{name}' has no effect since cvxpy does not support constraint names")
        self.model += [constraint]
        return self

    def add_constraints(self, constraints: Union[List, np.ndarray], name: Optional[str] = None) -> Any:
        if name is not None:
            self._LOGGER.warning(f"name='{name}' has no effect since cvxpy does not support constraint names")
        self.model += constraints
        return self

    def abs(self, a) -> np.ndarray:
        a = np.atleast_1d(a)
        return np.reshape([self._cp.abs(v) for v in a.flatten()], a.shape)

    def log(self, a) -> np.ndarray:
        a = np.atleast_1d(a)
        return np.reshape([self._cp.log(v) for v in a.flatten()], a.shape)

    def var(self,
            a: np.ndarray,
            axis: Optional[int] = None,
            definition: bool = False,
            asarray: bool = False,
            aux: bool = False) -> Any:
        raise BackendError(unsupported='variance due to numerical instability')

    def cov(self, a: np.ndarray, b: np.ndarray, definition: bool = True, asarray: bool = False, aux: bool = False):
        raise BackendError(unsupported='covariance due to numerical instability')

    def subtract(self, a, b):
        if isinstance(b, np.ndarray) and b.size > 1:
            # pairwise differences (a.size should be equal to b.size)
            return super(CvxpyBackend, self).subtract(a, b)
        else:
            # all elements of a minus a single element (b)
            a, b = np.atleast_1d(a), np.atleast_1d(b).flatten()[0]
            expressions = [ai - b for ai in a.flatten()]
            return np.reshape(expressions, a.shape)

    def multiply(self, a, b):
        if isinstance(b, np.ndarray) and b.size > 1:
            # pairwise differences (a.size should be equal to b.size)
            expressions = [self._cp.multiply(ai, bi) for ai, bi in zip(a.flatten(), b.flatten())]
            return np.reshape(expressions, a.shape)
        else:
            # all elements of a multiplied by single element (b)
            a, b = np.atleast_1d(a), np.atleast_1d(b).flatten()[0]
            expressions = [self._cp.multiply(ai, b) for ai in a.flatten()]
            return np.reshape(expressions, a.shape)
