from typing import Any, Union, List, Optional, Dict

import numpy as np

from moving_targets.masters.backends import Backend
from moving_targets.util.errors import MissingDependencyError, BackendError


class GurobiBackend(Backend):
    """`Backend` implementation for the Gurobi Solver."""

    def __init__(self, verbose: bool = False, **solver_args):
        """
        :param verbose:
            Whether or not to print information during the optimization process.

        :param solver_args:
            Parameters of the solver to be set via the `model.SetParam()` function.
        """
        super(GurobiBackend, self).__init__()

        try:
            import gurobipy
            self._gp = gurobipy
            """The lazily imported gurobipy instance."""
        except ModuleNotFoundError:
            raise MissingDependencyError(package='gurobipy')

        self.verbose: bool = verbose
        """Whether or not to print information during the optimization process."""

        self.solver_args: Dict[str, Any] = solver_args
        """Parameters of the solver to be set via the `model.SetParam()` function."""

        self._env: Optional = None
        """The gurobi environment instance."""

    def _build_model(self) -> Any:
        self._env = self._gp.Env(empty=True)
        self._env.setParam('OutputFlag', self.verbose)
        self._env.start()
        model = self._gp.Model(env=self._env, name='model')
        for param, value in self.solver_args.items():
            model.setParam(param, value)
        return model

    def _solve_model(self) -> Optional:
        self.model.update()
        self.model.optimize()
        if self.model.SolCount == 0:
            # if no solution is found, remove the stopping criteria and optimize until the first feasible solution
            for param in self.solver_args.keys():
                self.model.setParam(param, 'default')
            self.model.setParam('SolutionLimit', 1)
            self.model.optimize()
        return self.model

    def clear(self) -> Any:
        self._env.dispose()
        self.model.dispose()
        super(GurobiBackend, self).clear()

    def minimize(self, cost) -> Any:
        self.model.setObjective(cost, self._gp.GRB.MINIMIZE)
        return self

    def maximize(self, cost) -> Any:
        self.model.setObjective(cost, self._gp.GRB.MAXIMIZE)
        return self

    def get_objective(self) -> float:
        return self.solution.objVal

    def get_values(self, expressions: np.ndarray) -> np.ndarray:
        values = [v.x if isinstance(v, self._gp.Var) else v.getValue() for v in expressions.flatten()]
        return np.reshape(values, expressions.shape)

    def add_variable(self, vtype: str, lb: float, ub: float, name: Optional[str] = None) -> Any:
        if not hasattr(self._gp.GRB, vtype.upper()):
            raise BackendError(unsupported=f"vtype '{vtype}'")
        # addVar does not accept name=None as parameter
        kwargs = dict() if name is None else dict(name=name)
        var = self.model.addVar(vtype=getattr(self._gp.GRB, vtype.upper()), lb=lb, ub=ub, **kwargs)
        return var

    def add_variables(self, *keys: int, vtype: str, lb: float, ub: float, name: Optional[str] = None) -> np.ndarray:
        if len(keys) == 0:
            # if no keys are passed, builds a single variable then reshape it into a zero-dimensional numpy array
            var = self.add_variable(vtype=vtype, lb=lb, ub=ub, name=name)
        elif not hasattr(self._gp.GRB, vtype.upper()):
            raise BackendError(unsupported=f"vtype '{vtype}'")
        else:
            vtype = getattr(self._gp.GRB, vtype.upper())
            var = self.model.addVars(*keys, vtype=vtype, lb=lb, ub=ub, name=name)
            var = list(var.values())
        return np.array(var).reshape(keys)

    def add_constraints(self, constraints: Union[List, np.ndarray], name: Optional[str] = None) -> Any:
        self.model.addConstrs((c for c in constraints), name=name)
        return self

    def add_constraint(self, constraint, name: Optional[str] = None) -> Any:
        # addConstr does not accept name=None as parameter
        kwargs = dict() if name is None else dict(name=name)
        self.model.addConstr(constraint, **kwargs)
        return self

    def add_indicator_constraint(self, indicator, expression, value: int = 1, name: Optional[str] = None) -> Any:
        # addGenConstrIndicator does not accept name=None as parameter
        kwargs = dict() if name is None else dict(name=name)
        self.model.addGenConstrIndicator(indicator, value, expression, **kwargs)

    def sqrt(self, a) -> np.ndarray:
        a = np.atleast_1d(a)
        # creating auxiliary variables is necessary since 'addGenConstr' does not accept expressions
        aux_vector = self.aux(expressions=a, aux_vtype='continuous')
        sqrt_vector = self.add_continuous_variables(*a.shape, lb=0.0, ub=float('inf'))
        for aux_var, sqrt_var in zip(aux_vector.flatten(), sqrt_vector.flatten()):
            self.model.addGenConstrPow(sqrt_var, aux_var, 2)
        return sqrt_vector

    def abs(self, a) -> np.ndarray:
        a = np.atleast_1d(a)
        # creating auxiliary variables is necessary since 'addGenConstr' does not accept expressions
        aux_vector = self.aux(expressions=a, aux_vtype='continuous')
        abs_vector = self.add_continuous_variables(*a.shape, lb=0.0, ub=float('inf'))
        for aux_var, abs_var in zip(aux_vector.flatten(), abs_vector.flatten()):
            self.model.addGenConstrAbs(abs_var, aux_var)
        return abs_vector

    def log(self, a) -> np.ndarray:
        a = np.atleast_1d(a)
        # creating auxiliary variables is necessary since 'addGenConstr' does not accept expressions
        aux_vector = self.aux(expressions=a, aux_vtype='continuous')
        log_vector = self.add_continuous_variables(*a.shape, lb=-float('inf'), ub=float('inf'))
        for aux_var, log_var in zip(aux_vector.flatten(), log_vector.flatten()):
            self.model.addGenConstrExp(log_var, aux_var)
        return log_vector

    def min(self, a: np.ndarray, axis: Optional[int] = None, asarray: bool = False, aux: bool = False) -> Any:
        def _min(_array):
            min_val = self.add_continuous_variable()
            self.model.addGenConstrMin(min_val, list(_array), constant=float('inf'))
            return min_val

        # creating auxiliary variables is necessary since 'addGenConstr' does not accept expressions
        aux_vector = self.aux(expressions=a, aux_vtype='continuous')
        return self._handle_axes(aux_vector, operation=_min, axis=axis, asarray=asarray)

    def max(self, a: np.ndarray, axis: Optional[int] = None, asarray: bool = False, aux: bool = False) -> Any:
        def _max(_array):
            max_val = self.add_continuous_variable()
            self.model.addGenConstrMax(max_val, list(_array), constant=-float('inf'))
            return max_val

        # creating auxiliary variables is necessary since 'addGenConstr' does not accept expressions
        aux_vector = self.aux(expressions=a, aux_vtype='continuous')
        return self._handle_axes(aux_vector, operation=_max, axis=axis, asarray=asarray)

    # set aux True by default since Gurobi can handle quadratic constraints and this speeds up the computation a lot
    def var(self,
            a: np.ndarray,
            axis: Optional[int] = None,
            definition: bool = False,
            asarray: bool = False,
            aux: bool = True) -> Any:
        return super(GurobiBackend, self).var(a, axis=axis, definition=definition, asarray=asarray, aux=aux)

    # set aux True by default since Gurobi can handle quadratic constraints and this speeds up the computation a lot
    def cov(self, a: np.ndarray, b: np.ndarray, definition: bool = False, asarray: bool = False, aux: bool = True):
        return super(GurobiBackend, self).cov(a, b, definition=definition, asarray=asarray, aux=aux)

    def divide(self, a, b):
        try:
            return super(GurobiBackend, self).divide(a, b)
        except self._gp.GurobiError:
            # gurobipy.GurobiError: Divisor must be a constant
            # in case the divisor is not an array of constants, we handle the case by adding N auxiliary variables z_i
            # so that a_i / b_i = z_i --> a_i = z_i * b_i
            a, b = np.atleast_1d(a), np.atleast_2d(b)
            z = self.add_continuous_variables(*a.shape)
            self.add_constraints([ai == zi * bi for ai, bi, zi in zip(a.flatten(), b.flatten(), z.flatten())])
            return z

    def norm_0(self, a: np.ndarray, axis: Optional[int] = None, asarray: bool = False, aux: bool = False) -> Any:
        def _norm(_array):
            norm_val = self.add_continuous_variable(lb=-float('inf'), ub=float('inf'))
            self.model.addGenConstrNorm(norm_val, list(_array), which=0)
            return norm_val

        # creating auxiliary variables is necessary since 'addGenConstr' does not accept expressions
        aux_vector = self.aux(expressions=a, aux_vtype='continuous')
        return self._handle_axes(aux_vector, operation=_norm, axis=axis, asarray=asarray)

    def norm_1(self, a: np.ndarray, axis: Optional[int] = None, asarray: bool = False, aux: bool = False) -> Any:
        def _norm(_array):
            norm_val = self.add_continuous_variable(lb=-float('inf'), ub=float('inf'))
            self.model.addGenConstrNorm(norm_val, list(_array), which=1)
            return norm_val

        # creating auxiliary variables is necessary since 'addGenConstr' does not accept expressions
        aux_vector = self.aux(expressions=a, aux_vtype='continuous')
        return self._handle_axes(aux_vector, operation=_norm, axis=axis, asarray=asarray)

    def norm_2(self,
               a: np.ndarray,
               squared: bool = True,
               axis: Optional[int] = None,
               asarray: bool = False,
               aux: bool = False) -> Any:
        if squared:
            def _norm(_array):
                norm_val = self.add_continuous_variable(lb=-float('inf'), ub=float('inf'))
                self.model.addGenConstrNorm(norm_val, list(_array), which=2)
                return self.square(norm_val)
        else:
            def _norm(_array):
                norm_val = self.add_continuous_variable(lb=-float('inf'), ub=float('inf'))
                self.model.addGenConstrNorm(norm_val, list(_array), which=2)
                return norm_val

        # creating auxiliary variables is necessary since 'addGenConstr' does not accept expressions
        aux_vector = self.aux(expressions=a, aux_vtype='continuous')
        return self._handle_axes(aux_vector, operation=_norm, axis=axis, asarray=asarray)

    def norm_inf(self, a: np.ndarray, axis: Optional[int] = None, asarray: bool = False, aux: bool = False) -> Any:
        def _norm(_array):
            from gurobipy import GRB
            norm_val = self.add_continuous_variable(lb=-float('inf'), ub=float('inf'))
            self.model.addGenConstrNorm(norm_val, list(_array), which=GRB.INFINITY)
            return norm_val

        # creating auxiliary variables is necessary since 'addGenConstr' does not accept expressions
        aux_vector = self.aux(expressions=a, aux_vtype='continuous')
        return self._handle_axes(aux_vector, operation=_norm, axis=axis, asarray=asarray)
