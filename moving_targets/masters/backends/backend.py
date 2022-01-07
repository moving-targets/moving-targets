"""Basic Backend Interface."""
from typing import Any, Union, List, Optional

import numpy as np

from moving_targets.util.errors import not_implemented_message
from moving_targets.util.typing import Number


class Backend:
    """Basic Interface for a Moving Targets Master Backend."""

    _ERROR_MESSAGE: str = 'This backend cannot deal with '
    """Error message for unsupported operations."""

    def __init__(self):
        super(Backend, self).__init__()

        self.model: Optional = None
        """The inner model instance."""

        self.solution: Optional = None
        """The inner solution instance."""

    def _build_model(self) -> Any:
        """Initializes a model instance.

        :return:
            The model instance.
        """
        raise NotImplementedError(not_implemented_message(name='_build_model'))

    def _solve_model(self) -> Optional:
        """Solves the optimization problem.

        :return:
            The solution instance or None if no solution is found.
        """
        raise NotImplementedError(not_implemented_message(name='_solve_model'))

    def build(self) -> Any:
        """Initializes a model instance.

        :return:
            The backend itself.
        """
        assert self.model is None, "A model instance is already present, please solve that before building a new one"
        self.model = self._build_model()
        self.solution = None
        return self

    def solve(self) -> Any:
        """Solves the optimization problem.

        :return:
            The backend itself.
        """
        self.solution = self._solve_model()
        self.model = None
        return self

    def minimize(self, cost) -> Any:
        """Sets a cost function to minimize before solving.

        :param cost:
            The cost function.

        :return:
            The backend itself.
        """
        raise NotImplementedError(not_implemented_message(name='minimize'))

    def maximize(self, cost) -> Any:
        """Sets a cost function to maximize before solving.

        :param cost:
            The cost function.

        :return:
            The backend itself.
        """
        return self.minimize(-cost)

    def add_constraints(self, constraints: Union[List, np.ndarray], name: Optional[str] = None) -> Any:
        """Adds a list of constraints to the model.

        :param constraints:
            The list of constraints.

        :param name:
            The constraints name.

        :return:
            The backend itself.
        """
        raise NotImplementedError(not_implemented_message(name='add_constraints'))

    def add_constraint(self, constraint, name: Optional[str] = None) -> Any:
        """Adds a constraint to the model.

        :param constraint:
            The constraint.

        :param name:
            The constraint name.

        :return:
            The backend itself.
        """
        return self.add_constraints(constraints=[constraint], name=name)

    def add_binary_variables(self, *keys: int, name: Optional[str] = None) -> np.ndarray:
        """Creates an array of binary model variables.

        :param keys:
            The number of keys per dimension of the array of variables.

        :param name:
            The variables name.

        :return:
            The array of binary model variables.
        """
        return self.add_variables(*keys, vtype='binary', name=name)

    def add_binary_variable(self, name: Optional[str] = None) -> Any:
        """Creates a binary model variable.

        :param name:
            The variable name.

        :return:
            The binary model variable.
        """
        return self.add_variable(vtype='binary', name=name)

    def add_integer_variables(self,
                              *keys: int,
                              lb: Optional[Number] = None,
                              ub: Optional[Number] = None,
                              name: Optional[str] = None) -> np.ndarray:
        """Creates an array of integer model variables.

        :param keys:
            The number of keys per dimension of the array of variables.

        :param lb:
            The variables lower bound.

        :param ub:
            The variables upper bound.

        :param name:
            The variables name.

        :return:
            The array of integer model variables.
        """
        return self.add_variables(*keys, vtype='integer', name=name, lb=lb, ub=ub)

    def add_integer_variable(self,
                             lb: Optional[Number] = 0,
                             ub: Optional[Number] = None,
                             name: Optional[str] = None) -> np.ndarray:
        """Creates an integer model variable.

        :param lb:
            The variable lower bound.

        :param ub:
            The variable upper bound.

        :param name:
            The variable name.

        :return:
            The integer model variable.
        """
        return self.add_variable(vtype='integer', name=name, lb=lb, ub=ub)

    def add_continuous_variables(self,
                                 *keys: int,
                                 lb: Optional[Number] = None,
                                 ub: Optional[Number] = None,
                                 name: Optional[str] = None) -> np.ndarray:
        """Creates an array of continuous model variables.

        :param keys:
            The number of keys per dimension of the array of variables.

        :param lb:
            The variables lower bound.

        :param ub:
            The variables upper bound.

        :param name:
            The variables name.

        :return:
            The array of continuous model variables.
        """
        return self.add_variables(*keys, vtype='continuous', name=name, lb=lb, ub=ub)

    def add_continuous_variable(self,
                                lb: Optional[Number] = None,
                                ub: Optional[Number] = None,
                                name: Optional[str] = None) -> np.ndarray:
        """Creates a continuous model variable.

        :param lb:
            The variable lower bound.

        :param ub:
            The variable upper bound.

        :param name:
            The variable name.

        :return:
            The continuous model variable.
        """
        return self.add_variable(vtype='continuous', name=name, lb=lb, ub=ub)

    def add_variables(self,
                      *keys: int,
                      vtype: str,
                      lb: Optional[Number] = None,
                      ub: Optional[Number] = None,
                      name: Optional[str] = None) -> np.ndarray:
        """Creates an array of model variables.

        :param keys:
            The number of keys per dimension of the array of variables.

        :param vtype:
            The variables type, usually 'binary', 'integer', or 'continuous'.

        :param lb:
            The variables lower bound.

        :param ub:
            The variables upper bound.

        :param name:
            The variables name.

        :return:
            The array of model variables.
        """
        raise NotImplementedError(not_implemented_message(name='add_variables'))

    def add_variable(self,
                     vtype: str,
                     lb: Optional[Number] = None,
                     ub: Optional[Number] = None,
                     name: Optional[str] = None) -> Any:
        """Creates a model variable.

        :param vtype:
            The variables type, usually 'binary' or 'continuous'.

        :param lb:
            The variable lower bound.

        :param ub:
            The variable upper bound.

        :param name:
            The variable name.

        :return:
            The model variable.
        """
        return self.add_variables(1, vtype=vtype, name=name, lb=lb, ub=ub)[0]

    def get_objective(self) -> Number:
        raise NotImplementedError(not_implemented_message(name='get_objective'))

    def get_values(self, expressions: np.ndarray) -> np.ndarray:
        raise NotImplementedError(not_implemented_message(name='get_values'))

    def get_value(self, expression: Any) -> Number:
        return self.get_values(np.array([expression]))[0]

    def sum(self, vector: np.ndarray) -> Any:
        """Performs the sum over a vector of variables.

        :param vector:
            A vector of model variables.

        :return:
            The sum of the variables.
        """
        raise NotImplementedError(not_implemented_message(name='sum'))

    def sqr(self, vector: np.ndarray) -> np.ndarray:
        """Computes the squared values over a vector of variables.

        :param vector:
            A vector of model variables.

        :return:
            The vector of squared values.
        """
        raise AssertionError(self._ERROR_MESSAGE + 'squared values')

    def abs(self, vector: np.ndarray) -> np.ndarray:
        """Computes the absolute values over a vector of variables.

        :param vector:
            A vector of model variables.

        :return:
            The vector of absolute values.
        """
        raise AssertionError(self._ERROR_MESSAGE + 'absolute values')

    def log(self, vector: np.ndarray) -> np.ndarray:
        """Computes the logarithms over a vector of variables.

        :param vector:
            A vector of model variables.

        :return:
            The vector of logarithms.
        """
        raise AssertionError(self._ERROR_MESSAGE + 'logarithms')
