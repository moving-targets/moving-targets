from typing import Any, Union, List, Optional

import numpy as np

from moving_targets.masters.backends import Backend
from moving_targets.util.errors import MissingDependencyError, BackendError


class CplexBackend(Backend):
    """`Backend` implementation for the Cplex Solver."""

    def __init__(self, time_limit: Optional[float] = None, verbose: bool = False):
        """
        :param time_limit:
            The solver time limit.

        :param verbose:
            Whether or not to print information at the end of the optimization process.
        """
        super(CplexBackend, self).__init__(sum_fn=lambda v: self.model.sum(v))

        try:
            import docplex.mp.model
            self._cp = docplex.mp.model
            """The lazily imported docplex instance."""
        except ModuleNotFoundError:
            raise MissingDependencyError(package='docplex')

        self.time_limit: Optional[float] = time_limit
        """The solver time limit."""

        self.verbose: bool = verbose
        """Whether or not to print information at the end of the optimization process."""

    def _build_model(self) -> Any:
        model = self._cp.Model(name='model')
        if self.time_limit is not None:
            model.set_time_limit(self.time_limit)
        return model

    def _solve_model(self) -> Optional:
        solution = self.model.solve()
        if self.verbose:
            print(solution.solve_details)
        return solution

    def clear(self) -> Any:
        self.model.end()
        super(CplexBackend, self).clear()

    def maximize(self, cost) -> Any:
        self.model.maximize(cost)
        return self

    def minimize(self, cost) -> Any:
        self.model.minimize(cost)
        return self

    def get_objective(self) -> float:
        return self.solution.objective_value

    def get_values(self, expressions: np.ndarray) -> np.ndarray:
        return np.reshape([v.solution_value for v in expressions.flatten()], expressions.shape)

    def add_constant(self, val, vtype: str = 'continuous', name: Optional[str] = None) -> Any:
        if vtype == 'binary':
            var = self.add_binary_variable(name=name)
            self.add_constraint(var == val)
        else:
            var = self.add_variable(vtype=vtype, lb=val, ub=val, name=name)
        return var

    def add_constants(self, val, vtype: str = 'continuous', name: Optional[str] = None) -> np.ndarray:
        # a default strategy to create an array of constants is to compute their names leveraging the utility function,
        # creating a mono-dimensional list of variables with correct names and fixed lower/upper bounds to the given
        # values then, eventually, reshaping then into the correct shape
        names = np.array(self._nested_names(*val.shape, name=name)).flatten()
        if vtype == 'binary':
            var = [self.add_binary_variable(name=n) for v, n in zip(val.flatten(), names)]
            self.add_constraints([vr == vl for vr, vl in zip(var, val.flatten())])
        else:
            var = [self.add_variable(vtype=vtype, lb=v, ub=v, name=n) for v, n in zip(val.flatten(), names)]
        return np.reshape(var, val.shape)

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
        if len(keys) == 0:
            # if no keys are passed, builds a single variable then reshape it into a zero-dimensional numpy array
            var = self.add_variable(vtype=vtype, lb=lb, ub=ub, name=name)
        elif vtype not in ['binary', 'integer', 'continuous']:
            # check correctness of type variables
            raise BackendError(unsupported=f"vtype '{vtype}'")
        else:
            # handle non-zero dimensionality
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
            # and check the correctness of bounds wrt variable type
            if vtype == 'binary':
                assert lb == 0 and ub == 1, f"Binary variables accept [0, 1] bounds only, but [{lb}, {ub}] was passed."
            else:
                kw.update({'lb': lb, 'ub': ub})
            # handle variables
            var = fn(**kw, name=name)
            var = list(var.values())
        return np.array(var).reshape(keys)

    def add_constraints(self, constraints: Union[List, np.ndarray], name: Optional[str] = None) -> Any:
        self.model.add_constraints(constraints, names=name)
        return self

    def add_constraint(self, constraint, name: Optional[str] = None) -> Any:
        self.model.add_constraint(constraint, ctname=name)
        return self

    def add_indicator_constraints(self,
                                  indicators: np.ndarray,
                                  expressions: Union[List, np.ndarray],
                                  value: int = 1,
                                  name: Optional[str] = None) -> Any:
        binary_vars, cst = np.array(indicators).flatten(), np.array(expressions).flatten()
        self.model.add_indicators(binary_vars=binary_vars, cts=cst, true_values=value, names=name)
        return self

    def add_indicator_constraint(self, indicator, expression, value: int = 1, name: Optional[str] = None) -> Any:
        self.model.add_indicator(binary_var=indicator, linear_ct=expression, active_value=value, name=name)

    def abs(self, a) -> np.ndarray:
        a = np.atleast_1d(a)
        return np.reshape([self.model.abs(v) for v in a.flatten()], a.shape)

    def subtract(self, a, b):
        if not isinstance(a, np.ndarray) and not isinstance(b, np.ndarray):
            return super(CplexBackend, self).subtract(a, b)
        elif isinstance(b, np.ndarray) and b.size > 1:
            # pairwise differences (a.size should be equal to b.size)
            return super(CplexBackend, self).subtract(a, b)
        else:
            # all elements of a minus a single element (b)
            a, b = np.atleast_1d(a), np.atleast_1d(b).flatten()[0]
            expressions = [ai - b for ai in a.flatten()]
            return np.reshape(expressions, a.shape)

    def divide(self, a, b):
        try:
            return super(CplexBackend, self).divide(a, b)
        except self._cp.DOcplexException:
            # docplex.mp.utils.DOcplexException: operation not supported, only numbers can be denominators
            # in case the divisor is not an array of constants, we cannot handle the case by adding the auxiliary
            # variables as in the gurobi backend because cplex cannot handle quadratic constraints
            raise BackendError(unsupported='divisions having variables in the denominator')
