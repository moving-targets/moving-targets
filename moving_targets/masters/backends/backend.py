"""Basic Backend Interface."""
import logging
from typing import Any, Union, List, Optional

import numpy as np

from moving_targets.util.errors import not_implemented_message


class Backend:
    """Basic Interface for a Moving Targets Master Backend."""

    _ERROR_MESSAGE: str = 'This backend cannot deal with '
    """Error message for unsupported operations."""

    _LOGGER: logging.Logger = logging.getLogger('Backend')

    @classmethod
    def _aux_warning(cls, exp: Optional[str], aux: Optional[str], msg: str):
        """Logs a warning on the Backend Logger when the 'aux' parameter is not used correctly.

        :param exp:
            The expected value of the aux parameter.

        :param aux:
            The actual value of the aux parameter.

        :param msg:
            The warning explanation.
        """
        if not (exp is None and aux is None) and exp != aux:
            cls._LOGGER.warning(f"'aux={aux}' has no effect on '{cls.__name__}' since this solver {msg}.")

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
        if self.model is not None:
            self._LOGGER.warning("A model instance was already present and has been overwritten. If that was " +
                                 "intentional consider calling 'backend.clear()' before creating a new one.")
        self.model = self._build_model()
        self.solution = None
        return self

    def solve(self) -> Any:
        """Solves the optimization problem.

        :return:
            The backend itself.
        """
        self.solution = self._solve_model()
        return self

    def clear(self) -> Any:
        """Clears the backend resources.

        :return:
            The backend itself.
        """
        self.model = None
        self.solution = None
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
        return self.add_variables(*keys, vtype='binary', name=name, lb=0, ub=1)

    def add_binary_variable(self, name: Optional[str] = None) -> Any:
        """Creates a binary model variable.

        :param name:
            The variable name.

        :return:
            The binary model variable.
        """
        return self.add_variable(vtype='binary', name=name, lb=0, ub=1)

    def add_integer_variables(self,
                              *keys: int,
                              lb: float = 0,
                              ub: float = float('inf'),
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
                             lb: float = 0,
                             ub: float = float('inf'),
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
                                 lb: float = -float('inf'),
                                 ub: float = float('inf'),
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
                                lb: float = -float('inf'),
                                ub: float = float('inf'),
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

    def add_variable(self, vtype: str, lb: float, ub: float, name: Optional[str] = None) -> Any:
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

    def add_variables(self, *keys: int, vtype: str, lb: float, ub: float, name: Optional[str] = None) -> np.ndarray:
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

    def get_objective(self) -> float:
        """Gets the objective value of the solved model.

        :return:
            The objective value.
        """
        raise NotImplementedError(not_implemented_message(name='get_objective'))

    def get_values(self, expressions: np.ndarray) -> np.ndarray:
        """Gets the values of a vector of expressions as found in the model solution.

        :param expressions:
            The vector of expressions.

        :return:
            The vector of solution values.
        """
        raise NotImplementedError(not_implemented_message(name='get_values'))

    def get_value(self, expression: Any) -> float:
        """Gets the value of an expressions as found in the model solution.

        :param expression:
            The expressions.

        :return:
            The solution value.
        """
        return self.get_values(np.array([expression]))[0]

    def aux(self,
            expressions: Union[Any, np.ndarray],
            aux_vtype: Optional[str] = 'continuous',
            aux_lb: float = -float('inf'),
            aux_ub: float = float('inf'),
            aux_name: Optional[str] = None) -> np.ndarray:
        """If the 'aux_vtype' parameter is none, it simply return the input expressions, otherwise it builds and return
        an auxiliary variable (or a vector of variables) having the given vtype and the respective other properties,
        which is equal to the given expression (or vector of expressions). Using auxiliary variables may come in handy
        when dealing with huge datasets since they can considerably speedup the model formulation; still, imposing
        equality constraints on certain expressions may lead to solving errors due to broken model assumptions.

        :param expressions:
            The vector of expressions.

        :param aux_vtype:
            Either None or the auxiliary variables type, usually 'binary', 'integer', or 'continuous'.

        :param aux_lb:
            The auxiliary variables lower bound.

        :param aux_ub:
            The auxiliary variables upper bound.

        :param aux_name:
            The auxiliary variables name.

        :return:
            The array of auxiliary variables.
        """
        if aux_vtype is None:
            return expressions
        else:
            expressions = expressions if isinstance(expressions, np.ndarray) else np.array([expressions])
            variables = self.add_variables(*expressions.shape, vtype=aux_vtype, lb=aux_lb, ub=aux_ub, name=aux_name)
            self.add_constraints([v == e for v, e in zip(variables.flatten(), expressions.flatten())])
            return variables[0] if variables.size == 1 else variables

    def sum(self, vector: np.ndarray, aux: Optional[str] = None) -> Any:
        """Computes the sum of a vector of variables.

        :param vector:
            A vector of model variables.

        :param aux:
            If None is passed, it returns the result in the form of an expression, otherwise it builds an auxiliary
            variable bounded to the expression value. Using auxiliary variables may come in handy when dealing with
            huge datasets since they can considerably speedup the model formulation; still, imposing equality
            constraints on certain expressions may lead to solving errors due to broken model assumptions.

        :return:
            The variables sum.
        """
        raise NotImplementedError(not_implemented_message(name='sum'))

    def mean(self, vector: np.ndarray, aux: Optional[str] = None) -> Any:
        """Computes the mean of a vector of variables.

        :param vector:
            A vector of model variables.

        :param aux:
            If None is passed, it returns the result in the form of an expression, otherwise it builds an auxiliary
            variable bounded to the expression value. Using auxiliary variables may come in handy when dealing with
            huge datasets since they can considerably speedup the model formulation; still, imposing equality
            constraints on certain expressions may lead to solving errors due to broken model assumptions.

        :return:
            The variables mean.
        """
        sum_expression = self.sum(vector, aux=aux)
        return self.aux(sum_expression / len(vector), aux_vtype=aux)

    def sqr(self, vector: np.ndarray, aux: Optional[str] = None) -> np.ndarray:
        """Computes the squared values over a vector of variables.

        :param vector:
            A vector of model variables.

        :param aux:
            If None is passed, it returns the result in the form of an expression, otherwise it builds an auxiliary
            variable bounded to the expression value. Using auxiliary variables may come in handy when dealing with
            huge datasets since they can considerably speedup the model formulation; still, imposing equality
            constraints on certain expressions may lead to solving errors due to broken model assumptions.

        :return:
            The vector of squared values.
        """
        raise AssertionError(self._ERROR_MESSAGE + 'squared values')

    def abs(self, vector: np.ndarray, aux: Optional[str] = None) -> np.ndarray:
        """Computes the absolute values over a vector of variables.

        :param vector:
            A vector of model variables.

        :param aux:
            If None is passed, it returns the result in the form of an expression, otherwise it builds an auxiliary
            variable bounded to the expression value. Using auxiliary variables may come in handy when dealing with
            huge datasets since they can considerably speedup the model formulation; still, imposing equality
            constraints on certain expressions may lead to solving errors due to broken model assumptions.

        :return:
            The vector of absolute values.
        """
        raise AssertionError(self._ERROR_MESSAGE + 'absolute values')

    def log(self, vector: np.ndarray, aux: Optional[str] = None) -> np.ndarray:
        """Computes the logarithms over a vector of variables.

        :param vector:
            A vector of model variables.

        :param aux:
            If None is passed, it returns the result in the form of an expression, otherwise it builds an auxiliary
            variable bounded to the expression value. Using auxiliary variables may come in handy when dealing with
            huge datasets since they can considerably speedup the model formulation; still, imposing equality
            constraints on certain expressions may lead to solving errors due to broken model assumptions.

        :return:
            The vector of logarithms.
        """
        raise AssertionError(self._ERROR_MESSAGE + 'logarithms')

    def add(self, vector1: np.ndarray, vector2: np.ndarray, aux: Optional[str] = None):
        """Performs the pairwise sum between two vectors.

        :param vector1:
            The first vector.

        :param vector2:
            The second vector.

        :param aux:
            If None is passed, it returns the result in the form of an expression, otherwise it builds an auxiliary
            variable bounded to the expression value. Using auxiliary variables may come in handy when dealing with
            huge datasets since they can considerably speedup the model formulation; still, imposing equality
            constraints on certain expressions may lead to solving errors due to broken model assumptions.

        :return:
            The array of pairwise sums between vector1 and vector2.
        """
        return self.aux(expressions=vector1 + vector2, aux_vtype=aux)

    def multiply(self, vector1: np.ndarray, vector2: np.ndarray, aux: Optional[str] = None):
        """Performs the pairwise product between two vectors.

        :param vector1:
            The first vector.

        :param vector2:
            The second vector.

        :param aux:
            If None is passed, it returns the result in the form of an expression, otherwise it builds an auxiliary
            variable bounded to the expression value. Using auxiliary variables may come in handy when dealing with
            huge datasets since they can considerably speedup the model formulation; still, imposing equality
            constraints on certain expressions may lead to solving errors due to broken model assumptions.

        :return:
            The array of pairwise products between vector1 and vector2.
        """
        return self.aux(expressions=vector1 * vector2, aux_vtype=aux)

    def divide(self, vector1: np.ndarray, vector2: np.ndarray, aux: Optional[str] = None):
        """Performs the pairwise division between two vectors.

        :param vector1:
            The first vector.

        :param vector2:
            The second vector.

        :param aux:
            If None is passed, it returns the result in the form of an expression, otherwise it builds an auxiliary
            variable bounded to the expression value. Using auxiliary variables may come in handy when dealing with
            huge datasets since they can considerably speedup the model formulation; still, imposing equality
            constraints on certain expressions may lead to solving errors due to broken model assumptions.

        :return:
            The array of pairwise divisions between vector1 and vector2.
        """
        return self.aux(expressions=vector1 / vector2, aux_vtype=aux)

    def dot(self, vector1: np.ndarray, vector2: np.ndarray, aux: Optional[str] = None) -> Any:
        """Performs the vector product between two vectors.

        :param vector1:
            The first vector.

        :param vector2:
            The second vector.

        :param aux:
            If None is passed, it returns the result in the form of an expression, otherwise it builds an auxiliary
            variable bounded to the expression value. Using auxiliary variables may come in handy when dealing with
            huge datasets since they can considerably speedup the model formulation; still, imposing equality
            constraints on certain expressions may lead to solving errors due to broken model assumptions.

        :return:
            The dot product between vector1 and vector2.
        """
        product_expression = self.multiply(vector1=vector1, vector2=vector2, aux=aux)
        return self.aux(expressions=self.sum(product_expression), aux_vtype=aux)
