from typing import Any, Union, List, Optional, Dict

import numpy as np

from moving_targets.masters.backends import Backend
from moving_targets.util.errors import MissingDependencyError, BackendError


class GurobiBackend(Backend):
    """`Backend` implementation for the Gurobi Solver."""

    def __init__(self,
                 time_limit: Optional[float] = None,
                 solution_limit: Optional[float] = None,
                 verbose: bool = False,
                 **solver_args):
        """
        :param time_limit:
            The solver time limit.

        :param solution_limit:
            The solver solution limit.

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

        if time_limit is not None and 'TimeLimit' not in self.solver_args:
            self.solver_args['TimeLimit'] = time_limit
        if solution_limit is not None and 'SolutionLimit' not in self.solver_args:
            self.solver_args['SolutionLimit'] = solution_limit

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
        return None if self.model.SolCount == 0 else self.model

    def clear(self) -> Any:
        self._env.dispose()
        self.model.dispose()
        super(GurobiBackend, self).clear()

    def minimize(self, cost) -> Any:
        self.model.setObjective(cost, self._gp.GRB.MINIMIZE)
        return self

    def add_constraints(self, constraints: Union[List, np.ndarray], name: Optional[str] = None) -> Any:
        self.model.addConstrs((c for c in constraints), name=name)
        return self

    def add_variable(self, vtype: str, lb: float, ub: float, name: Optional[str] = None) -> Any:
        if not hasattr(self._gp.GRB, vtype.upper()):
            raise BackendError(unsupported=f"vtype '{vtype}'")
        var = self.model.addVar(vtype=getattr(self._gp.GRB, vtype.upper()), lb=lb, ub=ub, name=name, obj=0, column=None)
        self.model.update()
        return var

    def add_variables(self, *keys: int, vtype: str, lb: float, ub: float, name: Optional[str] = None) -> np.ndarray:
        if not hasattr(self._gp.GRB, vtype.upper()):
            raise BackendError(unsupported=f"vtype '{vtype}'")
        vtype = getattr(self._gp.GRB, vtype.upper())
        var = self.model.addVars(*keys, vtype=vtype, lb=lb, ub=ub, name=name).values()
        self.model.update()
        return np.array(var).reshape(keys)

    def get_objective(self) -> float:
        return self.solution.objVal

    def get_values(self, expressions: np.ndarray) -> np.ndarray:
        values = [v.x if isinstance(v, self._gp.Var) else v.getValue() for v in expressions.flatten()]
        return np.reshape(values, expressions.shape)

    def sum(self, a: np.ndarray, aux: Optional[str] = None) -> Any:
        return self.aux(expressions=np.sum(a), aux_vtype=aux)

    def square(self, a: np.ndarray, aux: Optional[str] = None) -> np.ndarray:
        return self.aux(expressions=a ** 2, aux_vtype=aux)

    def abs(self, a: np.ndarray, aux: Optional[str] = 'continuous') -> np.ndarray:
        self._aux_warning(exp='continuous', aux=aux, msg='needs aux variables to compute absolute values')
        # creating auxiliary variables is necessary since 'addGenConstrAbs' does not accept expressions
        aux_vector = self.aux(expressions=a.flatten(), aux_vtype='continuous')
        abs_vector = self.add_continuous_variables(len(aux_vector), lb=0.0, ub=float('inf'))
        for aux_var, abs_var in zip(aux_vector, abs_vector):
            self.model.addGenConstrAbs(abs_var, aux_var)
        return np.reshape(abs_vector, a.shape)

    def log(self, a: np.ndarray, aux: Optional[str] = 'continuous') -> np.ndarray:
        self._aux_warning(exp='continuous', aux=aux, msg='needs aux variables to compute logarithms')
        # creating auxiliary variables is necessary since 'addGenConstrExp' does not accept expressions
        aux_vector = self.aux(expressions=a.flatten(), aux_vtype='continuous')
        log_vector = self.add_continuous_variables(len(aux_vector), lb=-float('inf'), ub=float('inf'))
        for aux_var, log_var in zip(aux_vector, log_vector):
            self.model.addGenConstrExp(log_var, aux_var)
        return np.reshape(log_vector, a.shape)
