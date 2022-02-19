"""Basic Backend Interface."""
import logging
from typing import Any, Union, List, Optional, Callable

import numpy as np

from moving_targets.util.errors import not_implemented_message, BackendError


class Backend:
    """Basic Interface for a Moving Targets Master Backend."""

    _LOGGER: logging.Logger = logging.getLogger('Backend')

    @staticmethod
    def _transposed_axes(dim: int, index: int, place: int) -> List[int]:
        """Computes the order of the transposed axes.

        :param dim:
            The total number of dimensions.

        :param index:
            The index of the axis to move.

        :param place:
            The position where to move the given index. E.g., if dim = 4, index = 2, and place = 0, the third dimension
            will be brought to front thus it will be returned the list [2, 0, 1, 3]; otherwise, if dim = 4, index = 0,
            and place = 2, the operation will be inverted, thus returning [1, 2, 0, 3].

        :return:
            The list of indices of the transposed axes.
        """
        axes = list(range(dim))
        axes.insert(place, axes.pop(index))
        return axes

    @classmethod
    def _nested_names(cls, *keys: int, name: Optional[str]) -> List:
        """Generates an keys-dimensional list of nested lists containing the correct variables names when an array of
        variables is wanted instead of a single one by appending the variable position in the keys-dimensional tensor.

        :param keys:
            The number of keys per dimension of the array of variables.

        :param name:
            The common name for the variables.

        :return:
            The keys-dimensional structure of nested lists.
        """
        if len(keys) == 1:
            return [None if name is None else f'{name}_{i}' for i in range(keys[0])]
        else:
            return [cls._nested_names(*keys[1:], name=None if name is None else f'{name}_{i}') for i in range(keys[0])]

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
        if aux != 'auto' and exp != aux:
            cls._LOGGER.warning(f"'aux={aux}' has no effect since the solver needs {exp} auxiliary variables to {msg}.")

    def __init__(self, sum_fn: Callable = lambda v: np.sum(v)):
        """
        :param sum_fn:
            The backend function to perform a sum over a one-dimensional vector of model variables.
        """
        super(Backend, self).__init__()

        self._sum_fn: Callable = sum_fn
        """The backend function to perform a sum over an array of model variables."""

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
        raise NotImplementedError(not_implemented_message(name='maximize'))

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

    def add_constant(self, val: Any, vtype: str = 'continuous', name: Optional[str] = None) -> Any:
        """Creates a model variable with a constant value.

        :param val:
            The numerical value of the constant.

        :param vtype:
            The constant type, by default this is 'continuous'.

        :param name:
            The constant name.

        :return:
            A model variable with value fixed to the given one via a constraint.
        """
        return self.add_variable(vtype=vtype, lb=val, ub=val, name=name)

    def add_constants(self, val: np.ndarray, vtype: str = 'continuous', name: Optional[str] = None) -> np.ndarray:
        """Creates an array of model variables with constant values.

        :param val:
            The array of numerical values of the constants.

        :param vtype:
            The constants type, by default this is 'continuous'.

        :param name:
            The constants name.

        :return:
            An array of model variables with value fixed to the given ones via a set of constraints.
        """
        # a default strategy to create an array of constants is to compute their names leveraging the utility function,
        # creating a mono-dimensional list of variables with correct names and fixed lower/upper bounds to the given
        # values then, eventually, reshaping then into the correct shape
        names = np.array(self._nested_names(*val.shape, name=name)).flatten()
        var = [self.add_variable(vtype=vtype, lb=v, ub=v, name=n) for v, n in zip(val.flatten(), names)]
        return np.reshape(var, val.shape)

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
        raise NotImplementedError(not_implemented_message(name='add_variable'))

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
        # a default strategy to create an array of variables is to compute their names leveraging the utility function
        # then creating a mono-dimensional list of variables and reshaping then into the correct shape
        names = np.array(self._nested_names(*keys, name=name)).flatten()
        return np.reshape([self.add_variable(vtype=vtype, lb=lb, ub=ub, name=n) for n in names], keys)

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
            expressions: Any,
            aux_vtype: Optional[str] = 'continuous',
            aux_lb: float = -float('inf'),
            aux_ub: float = float('inf'),
            aux_name: Optional[str] = None) -> Any:
        """If the 'aux_vtype' parameter is None, it simply return the input expressions, otherwise it builds and return
        an auxiliary variable (or a vector of variables) having the given vtype and the respective other properties,
        which is equal to the given expression (or vector of expressions). Using auxiliary variables may come in handy
        when dealing with huge datasets since they can considerably speedup the model formulation; still, imposing
        equality constraints on certain expressions may lead to solving errors due to broken model assumptions.

        :param expressions:
            Either a single expression or a vector of expressions.

        :param aux_vtype:
            Either None (or 'auto') or the auxiliary variables type, usually 'binary', 'integer', or 'continuous'.

        :param aux_lb:
            The auxiliary variables lower bound.

        :param aux_ub:
            The auxiliary variables upper bound.

        :param aux_name:
            The auxiliary variables name.

        :return:
            Either a single auxiliary variable or the array of auxiliary variables.
        """
        if aux_vtype is None or aux_vtype == 'auto':
            return expressions
        else:
            if isinstance(expressions, np.ndarray):
                expressions, is_numpy = expressions, True
            else:
                expressions, is_numpy = np.array([expressions]), False
            variables = self.add_variables(*expressions.shape, vtype=aux_vtype, lb=aux_lb, ub=aux_ub, name=aux_name)
            self.add_constraints([v == e for v, e in zip(variables.flatten(), expressions.flatten())])
            return variables if is_numpy else variables[0]

    def sum(self, a: np.ndarray, axis: Optional[int] = None, asarray: bool = False, aux: Optional[str] = 'auto') -> Any:
        """Computes the sum of an array of variables.

        :param a:
            An array of model variables.

        :param axis:
            The dimension on which to aggregate or None to aggregate the whole data.

        :param asarray:
            In case the aggregation should return a single expression, whether to return it as a numpy zero-dimensional
            array or as the expression itself.

        :param aux:
            The vtype of the auxiliary variables which may be added the represent the results values and, optionally,
            the partial results obtained in the computation. If 'auto' is passed, it automatically decides how to deal
            with auxiliary variables in order to maximize the computational gain without introducing formulation issues;
            if None is passed, it returns the result in the form of an expression, otherwise it builds an auxiliary
            variable bounded to the expression value. Using auxiliary variables may come in handy when dealing with
            huge datasets since they can considerably speedup the model formulation; still, imposing equality
            constraints on certain expressions may lead to solving errors due to broken model assumptions.

        :return:
            The variables sum.
        """
        # if no axis is specified, simply leverage the internal sum function to get a single expression, otherwise
        # compute the sum over the requested axis and then reshape the output array accordingly
        if axis is None:
            expressions = self._sum_fn(a)
            expressions = np.reshape(expressions, ()) if asarray else expressions
        else:
            # check that the axis is in bound and handle negative axis values
            assert axis in range(-a.ndim, a.ndim), f"Axis {axis} is out of bound for array with {a.ndim} dimensions"
            axis = axis % a.ndim
            # transpose the array in order to bring the axis-th dimension to the back, then reshape it into a matrix so
            # that we can aggregate only on the last dimension, which is the one representing the chosen axis
            axes = self._transposed_axes(dim=a.ndim, index=axis, place=a.ndim - 1)
            expressions = a.transpose(axes).reshape((-1, a.shape[axis]))
            expressions = [np.sum(row) for row in expressions]
            # at this point, the "expressions" list will have size a.size / a.shape[axis], thus we need to reshape it
            # accordingly to the input shape by popping out the axis-th dimension, which is now one due to the sum
            # (also, take care that the output is at least one-dimensional, otherwise return a single expression)
            new_shape = list(a.shape)
            new_shape.pop(axis)
            expressions = np.reshape(expressions, new_shape) if len(new_shape) > 0 or asarray else expressions[0]
        return self.aux(expressions, aux_vtype=aux)

    def square(self, a: np.ndarray, aux: Optional[str] = 'auto') -> np.ndarray:
        """Computes the squared values over an array of variables.

        :param a:
            An array of model variables.

        :param aux:
            The vtype of the auxiliary variables which may be added the represent the results values and, optionally,
            the partial results obtained in the computation. If 'auto' is passed, it automatically decides how to deal
            with auxiliary variables in order to maximize the computational gain without introducing formulation issues;
            if None is passed, it returns the result in the form of an expression, otherwise it builds an auxiliary
            variable bounded to the expression value. Using auxiliary variables may come in handy when dealing with
            huge datasets since they can considerably speedup the model formulation; still, imposing equality
            constraints on certain expressions may lead to solving errors due to broken model assumptions.

        :return:
            The vector of squared values.

        :raise `BackendError`:
            If the backend cannot handle squared values.
        """
        raise BackendError(unsupported='squared values')

    def sqrt(self, a: np.ndarray, aux: Optional[str] = 'auto') -> np.ndarray:
        """Computes the squared roots over an array of variables.

        :param a:
            An array of model variables.

        :param aux:
            The vtype of the auxiliary variables which may be added the represent the results values and, optionally,
            the partial results obtained in the computation. If 'auto' is passed, it automatically decides how to deal
            with auxiliary variables in order to maximize the computational gain without introducing formulation issues;
            if None is passed, it returns the result in the form of an expression, otherwise it builds an auxiliary
            variable bounded to the expression value. Using auxiliary variables may come in handy when dealing with
            huge datasets since they can considerably speedup the model formulation; still, imposing equality
            constraints on certain expressions may lead to solving errors due to broken model assumptions.

        :return:
            The vector of squared roots.
        """
        raise BackendError(unsupported='squared roots')

    def abs(self, a: np.ndarray, aux: Optional[str] = 'auto') -> np.ndarray:
        """Computes the absolute values over a vector of variables.

        :param a:
            An array of model variables.

        :param aux:
            The vtype of the auxiliary variables which may be added the represent the results values and, optionally,
            the partial results obtained in the computation. If 'auto' is passed, it automatically decides how to deal
            with auxiliary variables in order to maximize the computational gain without introducing formulation issues;
            if None is passed, it returns the result in the form of an expression, otherwise it builds an auxiliary
            variable bounded to the expression value. Using auxiliary variables may come in handy when dealing with
            huge datasets since they can considerably speedup the model formulation; still, imposing equality
            constraints on certain expressions may lead to solving errors due to broken model assumptions.

        :return:
            The vector of absolute values.

        :raise `BackendError`:
            If the backend cannot handle absolute values.
        """
        raise BackendError(unsupported='absolute values')

    def log(self, a: np.ndarray, aux: Optional[str] = 'auto') -> np.ndarray:
        """Computes the logarithms over an array of variables.

        :param a:
            An array of model variables.

        :param aux:
            The vtype of the auxiliary variables which may be added the represent the results values and, optionally,
            the partial results obtained in the computation. If 'auto' is passed, it automatically decides how to deal
            with auxiliary variables in order to maximize the computational gain without introducing formulation issues;
            if None is passed, it returns the result in the form of an expression, otherwise it builds an auxiliary
            variable bounded to the expression value. Using auxiliary variables may come in handy when dealing with
            huge datasets since they can considerably speedup the model formulation; still, imposing equality
            constraints on certain expressions may lead to solving errors due to broken model assumptions.

        :return:
            The vector of logarithms.

        :raise `BackendError`:
            If the backend cannot handle logarithms.
        """
        raise BackendError(unsupported='logarithms')

    def mean(self,
             a: np.ndarray,
             axis: Optional[int] = None,
             asarray: bool = False,
             aux: Optional[str] = 'auto') -> Any:
        """Computes the mean of an array of variables.

        :param a:
            An array of model variables.

        :param axis:
            The dimension on which to aggregate or None to aggregate the whole data.

        :param asarray:
            In case the aggregation should return a single expression, whether to return it as a numpy zero-dimensional
            array or as the expression itself.

        :param aux:
            The vtype of the auxiliary variables which may be added the represent the results values and, optionally,
            the partial results obtained in the computation. If 'auto' is passed, it automatically decides how to deal
            with auxiliary variables in order to maximize the computational gain without introducing formulation issues;
            if None is passed, it returns the result in the form of an expression, otherwise it builds an auxiliary
            variable bounded to the expression value. Using auxiliary variables may come in handy when dealing with
            huge datasets since they can considerably speedup the model formulation; still, imposing equality
            constraints on certain expressions may lead to solving errors due to broken model assumptions.

        :return:
            The variables mean.
        """
        sum_expression = self.sum(a, axis=axis, asarray=asarray, aux=aux)
        num_aggregated = a.size if axis is None else a.shape[axis]
        return self.aux(sum_expression / num_aggregated, aux_vtype=aux)

    def var(self, a: np.ndarray, axis: Optional[int] = None, asarray: bool = False, aux: Optional[str] = 'auto') -> Any:
        """Computes the variance of an array of variables.

        :param a:
            An array of model variables.

        :param axis:
            The dimension on which to aggregate or None to aggregate the whole data.

        :param asarray:
            In case the aggregation should return a single expression, whether to return it as a numpy zero-dimensional
            array or as the expression itself.

        :param aux:
            The vtype of the auxiliary variables which may be added the represent the results values and, optionally,
            the partial results obtained in the computation. If 'auto' is passed, it automatically decides how to deal
            with auxiliary variables in order to maximize the computational gain without introducing formulation issues;
            if None is passed, it returns the result in the form of an expression, otherwise it builds an auxiliary
            variable bounded to the expression value. Using auxiliary variables may come in handy when dealing with
            huge datasets since they can considerably speedup the model formulation; still, imposing equality
            constraints on certain expressions may lead to solving errors due to broken model assumptions.

        :return:
            The variables variance.
        """
        # if no axis is specified, simply compute the mean (which will be a single expression), then compute the array
        # of differences and shape it correctly, otherwise compute the means over the requested axis (which will be in
        # the form of an array), then compute the differences and eventually reshape them accordingly
        aux_mean, aux_squared = ('continuous', None) if aux == 'auto' else (aux, aux)
        mean_expressions = self.mean(a, axis=axis, asarray=True, aux=aux_mean)
        if axis is None:
            # we compute the array of differences element by element because of some problems that certain backends
            # arise due to the impossibility to subtract a single variable (or expression) to an array of variables
            diff_expressions = np.reshape([self.subtract(ai, mean_expressions) for ai in a.flatten()], a.shape)
        else:
            # check that the axis is in bound and handle negative axis values
            assert axis in range(-a.ndim, a.ndim), f"Axis {axis} is out of bound for array with {a.ndim} dimensions"
            axis = axis % a.ndim
            # transpose the array in order to bring the axis-th dimension to the front so to compute the differences of
            # each row in the given axis with respect to the mean of that axis, then transpose the axis from the front
            # back to its place since we want to match exactly the original shape
            a = a.transpose(self._transposed_axes(dim=a.ndim, index=axis, place=0))
            diff_expressions = np.array([self.subtract(ai, mean_expressions) for ai in a])
            diff_expressions = diff_expressions.transpose(self._transposed_axes(dim=a.ndim, index=0, place=axis))
        # we can finally build auxiliary variables for the differences (if needed), then square these differences and,
        # eventually, compute the mean of the squared differences over the given axis
        diff_expressions = self.aux(expressions=diff_expressions, aux_vtype=aux)
        squared_expressions = self.square(diff_expressions, aux=aux_squared)
        return self.mean(squared_expressions, axis=axis, asarray=asarray, aux=aux)

    def add(self, a: np.ndarray, b: np.ndarray, aux: Optional[str] = 'auto'):
        """Performs the pairwise sum between two arrays.

        :param a:
            The first array.

        :param b:
            The second array.

        :param aux:
            The vtype of the auxiliary variables which may be added the represent the results values and, optionally,
            the partial results obtained in the computation. If 'auto' is passed, it automatically decides how to deal
            with auxiliary variables in order to maximize the computational gain without introducing formulation issues;
            if None is passed, it returns the result in the form of an expression, otherwise it builds an auxiliary
            variable bounded to the expression value. Using auxiliary variables may come in handy when dealing with
            huge datasets since they can considerably speedup the model formulation; still, imposing equality
            constraints on certain expressions may lead to solving errors due to broken model assumptions.

        :return:
            The array of pairwise sums between a and b.
        """
        return self.aux(expressions=a + b, aux_vtype=aux)

    def subtract(self, a: np.ndarray, b: np.ndarray, aux: Optional[str] = 'auto'):
        """Performs the pairwise subtraction between two arrays.

        :param a:
            The first array.

        :param b:
            The second array.

        :param aux:
            The vtype of the auxiliary variables which may be added the represent the results values and, optionally,
            the partial results obtained in the computation. If 'auto' is passed, it automatically decides how to deal
            with auxiliary variables in order to maximize the computational gain without introducing formulation issues;
            if None is passed, it returns the result in the form of an expression, otherwise it builds an auxiliary
            variable bounded to the expression value. Using auxiliary variables may come in handy when dealing with
            huge datasets since they can considerably speedup the model formulation; still, imposing equality
            constraints on certain expressions may lead to solving errors due to broken model assumptions.

        :return:
            The array of pairwise differences between a and b.
        """
        return self.aux(expressions=a - b, aux_vtype=aux)

    def multiply(self, a: np.ndarray, b: np.ndarray, aux: Optional[str] = 'auto'):
        """Performs the pairwise product between two arrays.

        :param a:
            The first array.

        :param b:
            The second array.

        :param aux:
            The vtype of the auxiliary variables which may be added the represent the results values and, optionally,
            the partial results obtained in the computation. If 'auto' is passed, it automatically decides how to deal
            with auxiliary variables in order to maximize the computational gain without introducing formulation issues;
            if None is passed, it returns the result in the form of an expression, otherwise it builds an auxiliary
            variable bounded to the expression value. Using auxiliary variables may come in handy when dealing with
            huge datasets since they can considerably speedup the model formulation; still, imposing equality
            constraints on certain expressions may lead to solving errors due to broken model assumptions.

        :return:
            The array of pairwise products between a and b.
        """
        return self.aux(expressions=a * b, aux_vtype=aux)

    def divide(self, a: np.ndarray, b: np.ndarray, aux: Optional[str] = 'auto'):
        """Performs the pairwise division between two arrays.

        :param a:
            The first array.

        :param b:
            The second array.

        :param aux:
            The vtype of the auxiliary variables which may be added the represent the results values and, optionally,
            the partial results obtained in the computation. If 'auto' is passed, it automatically decides how to deal
            with auxiliary variables in order to maximize the computational gain without introducing formulation issues;
            if None is passed, it returns the result in the form of an expression, otherwise it builds an auxiliary
            variable bounded to the expression value. Using auxiliary variables may come in handy when dealing with
            huge datasets since they can considerably speedup the model formulation; still, imposing equality
            constraints on certain expressions may lead to solving errors due to broken model assumptions.

        :return:
            The array of pairwise divisions between a and b.
        """
        return self.aux(expressions=a / b, aux_vtype=aux)

    def dot(self, a: np.ndarray, b: np.ndarray, aux: Optional[str] = 'auto') -> Union[Any, np.ndarray]:
        """Performs the vector product between two arrays.

        :param a:
            The first array (at most 2d).

        :param b:
            The second array (at most 2d).

        :param aux:
            The vtype of the auxiliary variables which may be added the represent the results values and, optionally,
            the partial results obtained in the computation. If 'auto' is passed, it automatically decides how to deal
            with auxiliary variables in order to maximize the computational gain without introducing formulation issues;
            if None is passed, it returns the result in the form of an expression, otherwise it builds an auxiliary
            variable bounded to the expression value. Using auxiliary variables may come in handy when dealing with
            huge datasets since they can considerably speedup the model formulation; still, imposing equality
            constraints on certain expressions may lead to solving errors due to broken model assumptions.

        :return:
            The dot product between a and b.
        """
        a = a if a.ndim == 2 else a.reshape((1, len(a)))
        b = b if b.ndim == 2 else b.reshape((len(b), 1))
        # we use a list of lists instead of an array to let numpy handle the data type on its own
        r = []
        for i, row in enumerate(a):
            r.append([])
            for j, column in enumerate(b.T):
                pairwise_products = self.multiply(a=row, b=column, aux=aux)
                r[i].append(self.sum(pairwise_products, aux=aux))
        r = np.array(r)
        return r[0, 0] if r.size == 1 else r.squeeze()
