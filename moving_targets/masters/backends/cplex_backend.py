from typing import Any, Union, List, Optional

import numpy as np

from moving_targets.masters.backends import Backend
from moving_targets.util.errors import MissingDependencyError, BackendError


class CplexBackend(Backend):
    """`Backend` implementation for the Cplex Solver."""

    def __init__(self, time_limit: Optional[float] = None):
        """
        :param time_limit:
            The solver time limit.
        """
        super(CplexBackend, self).__init__()

        try:
            import docplex.mp.model
            self._cp = docplex.mp.model
            """The lazily imported docplex instance."""
        except ModuleNotFoundError:
            raise MissingDependencyError(package='docplex')

        self.time_limit: Optional[float] = time_limit
        """The solver time limit."""

    def _build_model(self) -> Any:
        model = self._cp.Model(name='model')
        if self.time_limit is not None:
            model.set_time_limit(self.time_limit)
        return model

    def _solve_model(self) -> Optional:
        solution = self.model.solve()
        return solution

    def clear(self) -> Any:
        self.model.end()
        super(CplexBackend, self).clear()

    def minimize(self, cost) -> Any:
        self.model.minimize(cost)
        return self

    def add_constraints(self, constraints: Union[List, np.ndarray], name: Optional[str] = None) -> Any:
        self.model.add_constraints(constraints, names=name)
        return self

    def add_variable(self, vtype: str, lb: float, ub: float, name: Optional[str] = None) -> Any:
        # handle variable type and bounds
        if vtype == 'binary':
            assert lb == 0 and ub == 1, f"Binary variable type accepts [0, 1] bounds only, but [{lb}, {ub}] was passed."
            return self.model.binary_var(name=name)
        elif vtype == 'integer':
            return self.model.integer_var(lb=lb, ub=ub, name=name)
        elif vtype == 'continuous':
            return self.model.continuous_var(lb=lb, ub=ub, name=name)
        else:
            raise BackendError(unsupported=f"vtype '{vtype}'")

    def add_variables(self, *keys: int, vtype: str, lb: float, ub: float, name: Optional[str] = None) -> np.ndarray:
        # handle variable types
        if vtype not in ['binary', 'integer', 'continuous']:
            raise BackendError(unsupported=f"vtype '{vtype}'")
        # handle dimensionality
        if len(keys) == 1:
            fn = getattr(self.model, f'{vtype}_var_dict')
            kw = dict(keys=keys[0])
        elif len(keys) == 2:
            fn = getattr(self.model, f'{vtype}_var_matrix')
            kw = dict(keys1=keys[0], keys2=keys[1])
        elif len(keys) == 3:
            fn = getattr(self.model, f'{vtype}_var_cube')
            kw = dict(keys1=keys[0], keys2=keys[1], keys3=keys[2])
        else:
            raise BackendError(unsupported='variables having more than three dimensions')
        # handle bounds wrt variable type
        if vtype == 'binary':
            assert lb == 0 and ub == 1, f"Binary variable type accepts [0, 1] bounds only, but [{lb}, {ub}] was passed."
        else:
            kw.update({'lb': lb, 'ub': ub})
        # handle variables
        variables_dict = fn(**kw, name=name)
        variables_list = list(variables_dict.values())
        return np.array(variables_list).reshape(keys)

    def get_objective(self) -> float:
        return self.solution.objective_value

    def get_values(self, expressions: np.ndarray) -> np.ndarray:
        return np.reshape([v.solution_value for v in expressions.flatten()], expressions.shape)

    def sum(self, a: np.ndarray, aux: Optional[str] = None) -> Any:
        from docplex.mp.quad import QuadExpr
        expression = self.model.sum(a)
        if isinstance(expression, QuadExpr):
            self._aux_warning(exp=None, aux=aux, msg='cannot impose equality constraints on quadratic expressions')
            return expression
        else:
            return self.aux(expressions=expression, aux_vtype=aux)

    def square(self, a: np.ndarray, aux: Optional[str] = None) -> np.ndarray:
        self._aux_warning(exp=None, aux=aux, msg='cannot impose equality constraints on quadratic expressions')
        return a ** 2

    def abs(self, a: np.ndarray, aux: Optional[str] = None) -> np.ndarray:
        expressions = np.reshape([self.model.abs(v) for v in a.flatten()], a.shape)
        return self.aux(expressions=expressions, aux_vtype=aux)
