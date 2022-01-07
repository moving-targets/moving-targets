from typing import Any, Union, List, Optional

import numpy as np

from moving_targets.masters.backends import Backend
from moving_targets.util.errors import MissingDependencyError
from moving_targets.util.typing import Number


class CplexBackend(Backend):
    """`Backend` implementation for the Cplex Solver."""

    def __init__(self, time_limit: Optional[Number] = None):
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

        self.time_limit: Optional[Number] = time_limit
        """The solver time limit."""

    def _build_model(self) -> Any:
        model = self._cp.Model(name='model')
        if self.time_limit is not None:
            model.set_time_limit(self.time_limit)
        return model

    def _solve_model(self) -> Optional:
        solution = self.model.solve()
        return solution

    def minimize(self, cost) -> Any:
        self.model.minimize(cost)
        return self

    def add_constraints(self, constraints: Union[List, np.ndarray], name: Optional[str] = None) -> Any:
        self.model.add_constraints(constraints, names=name)
        return self

    def add_variables(self,
                      *keys: int,
                      vtype: str,
                      lb: Optional[Number] = None,
                      ub: Optional[Number] = None,
                      name: Optional[str] = None) -> np.ndarray:
        assert vtype in ['binary', 'integer', 'continuous'], self._ERROR_MESSAGE + f"vtype '{vtype}'"
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
            raise AssertionError(self._ERROR_MESSAGE + 'variables having more than three dimensions')
        kw.update({} if vtype == 'binary' else {'lb': lb, 'ub': ub})
        return np.array(list(fn(**kw, name=name).values())).reshape(keys)

    def get_objective(self) -> Number:
        return self.solution.objective_value

    def get_values(self, expressions: np.ndarray) -> np.ndarray:
        return np.reshape([v.solution_value for v in expressions.flatten()], expressions.shape)

    def sum(self, vector: np.ndarray) -> Any:
        return self.model.sum(vector)

    def sqr(self, vector: np.ndarray) -> np.ndarray:
        return vector ** 2

    def abs(self, vector: np.ndarray) -> np.ndarray:
        return np.reshape([self.model.abs(v) for v in vector.flatten()], vector.shape)
