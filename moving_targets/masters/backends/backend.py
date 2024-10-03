"""Basic Backend Interface."""
import logging
from typing import Any, Union, List, Optional, Callable, Set

import numpy as np

from moving_targets.util.errors import not_implemented_message, BackendError


class Backend:
    """Basic Interface for a Moving Targets Master Backend."""

    _LOGGER: logging.Logger = logging.getLogger('Backend')
    """The inner backend logger instance."""

    _INDICATORS: Set[str] = {'>', '>=', '<', '<=', '==', '!='}

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
    def _handle_axes(cls, a: np.ndarray, operation: Callable, axis: Optional[int], asarray: bool):
        """Handle a vector operation that must be carried out among a certain axis.

        :param a:
            An array of model variables.

        :param operation:
            The operation routine, i.e., a function of type f(array) -> value.

        :param axis:
            The dimension on which to aggregate or None to aggregate the whole data.

        :param asarray:
            In case the aggregation returns a single expression, whether to return it as a zero-dimensional numpy array
            or as the expression itself.

        :return:
            Either a single variable or a vector of variables with the results of the axis-wise operation.
        """
        # if no axis is specified, simply leverage the given operation to get a single expression
        # otherwise perform the operation over the requested axis and then reshape the output array accordingly
        if axis is None:
            results = operation(np.reshape(a, -1))
            results = np.reshape(results, ()) if asarray else results
        else:
            # check that the axis is in bound and handle negative axis values
            assert axis in range(-a.ndim, a.ndim), f"Axis {axis} is out of bound for array with {a.ndim} dimensions"
            axis = axis % a.ndim
            # transpose the array in order to bring the axis-th dimension to the back, then reshape it into a matrix so
            # that we can aggregate only on the last dimension, which is the one representing the chosen axis
            axes = cls._transposed_axes(dim=a.ndim, index=axis, place=a.ndim - 1)
            results = a.transpose(axes).reshape((-1, a.shape[axis]))
            results = [operation(row) for row in results]
            # at this point, the "expressions" list will have size a.size / a.shape[axis], thus we need to reshape it
            # accordingly to the input shape by popping out the axis-th dimension, which is now one due to the sum
            # (also, take care that the output is at least one-dimensional, otherwise return a single expression)
            new_shape = list(a.shape)
            new_shape.pop(axis)
            results = np.reshape(results, new_shape) if len(new_shape) > 0 or asarray else results[0]
        return results

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

    def __init__(self, sum_fn: Callable = lambda v: np.sum(v)):
        """
        :param sum_fn:
            The backend function to perform a sum over a one-dimensional array of model variables.
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

    def get_objective(self) -> float:
        """Gets the objective value of the solved model.

        :return:
            The objective value.
        """
        raise NotImplementedError(not_implemented_message(name='get_objective'))

    def get_values(self, expressions) -> np.ndarray:
        """Gets the values of an array of expressions as found in the model solution.

        :param expressions:
            The array of expressions.

        :return:
            The array of solution values.
        """
        raise NotImplementedError(not_implemented_message(name='get_values'))

    def get_value(self, expression) -> float:
        """Gets the value of an expressions as found in the model solution.

        :param expression:
            The expressions.

        :return:
            The solution value.
        """
        return self.get_values(np.array([expression]))[0]

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

    def add_constant(self, val, vtype: str = 'continuous', name: Optional[str] = None) -> Any:
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

    def add_constants(self, val, vtype: str = 'continuous', name: Optional[str] = None) -> np.ndarray:
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

    def aux(self,
            expressions,
            aux_vtype: Optional[str] = 'continuous',
            aux_lb: float = -float('inf'),
            aux_ub: float = float('inf'),
            aux_name: Optional[str] = None) -> Any:
        """If the 'aux_vtype' parameter is None, it simply return the input expressions, otherwise it builds and return
        an auxiliary variable (or an array of variables) having the given vtype and the respective other properties,
        which is equal to the given expression (or an array of expressions). Using auxiliary variables may come in
        handy when dealing with huge datasets since they can considerably speedup the model formulation; still,
        imposing equality constraints on certain expressions may lead to solving errors due to broken model assumptions.

        :param expressions:
            Either a single expression or a array of expressions.

        :param aux_vtype:
            Either None or the auxiliary variables type, usually 'binary', 'integer', or 'continuous'.

        :param aux_lb:
            The auxiliary variables lower bound.

        :param aux_ub:
            The auxiliary variables upper bound.

        :param aux_name:
            The auxiliary variables name.

        :return:
            Either a single auxiliary variable or the array of auxiliary variables.
        """
        if aux_vtype is None:
            return expressions
        else:
            if isinstance(expressions, np.ndarray):
                expressions, is_numpy = expressions, True
            else:
                expressions, is_numpy = np.array([expressions]), False
            variables = self.add_variables(*expressions.shape, vtype=aux_vtype, lb=aux_lb, ub=aux_ub, name=aux_name)
            self.add_constraints([v == e for v, e in zip(variables.flatten(), expressions.flatten())])
            return variables if is_numpy else variables[0]

    def add_constraints(self, constraints: Union[List, np.ndarray], name: Optional[str] = None) -> Any:
        """Adds a list of constraints to the model.

        :param constraints:
            The list of constraints.

        :param name:
            The constraints name.

        :return:
            The backend itself.
        """
        constraints = np.array(constraints)
        names = np.array(self._nested_names(*constraints.shape, name=name))
        for c, n in zip(constraints.flatten(), names.flatten()):
            self.add_constraint(constraint=c, name=n)
        return self

    def add_constraint(self, constraint, name: Optional[str] = None) -> Any:
        """Adds a constraint to the model.

        :param constraint:
            The constraint.

        :param name:
            The constraint name.

        :return:
            The backend itself.
        """
        raise NotImplementedError(not_implemented_message(name='add_constraint'))

    def add_indicator_constraints(self,
                                  indicators: Union[List, np.ndarray],
                                  expressions: Union[List, np.ndarray],
                                  value: int = 1,
                                  name: Optional[str] = None) -> Any:
        """Impose a set of indicator constraints over the given expressions using the given binary indicators. The
        indicators indicators[i] are such that if indicators[i] == value (with value either 0 or 1), then the i-th
        expression holds, but in case indicators[i] == not value, then there is no information about that expression.

        :param indicators:
            An array of binary variables on which to impose the indicator constraints.

        :param expressions:
            An array or list of expressions for which it is necessary to check whether they hold or not.

        :param value:
            The value assumed by a binary variable when the respective expression holds, either 0 or 1.

        :param name:
            The constraints name.

        :return:
            The backend itself.

        :raise `BackendError`:
            If the backend cannot handle indicator variables.
        """
        indicators, expressions = np.array(indicators), np.array(expressions)
        names = np.array(self._nested_names(*expressions.shape, name=name))
        for i, e, n in zip(indicators.flatten(), expressions.flatten(), names.flatten()):
            self.add_indicator_constraint(indicator=i, expression=e, value=value, name=n)
        return self

    def add_indicator_constraint(self, indicator, expression, value: int = 1, name: Optional[str] = None) -> Any:
        """Impose an indicator constraint over the given expression using the given binary indicator. The indicator
        is such that if indicator == value (with value either 0 or 1), then the expression holds, but in case
        indicator == not value, then there is no information about the expression.

        :param indicator:
            The binary variable on which to impose the indicator constraints.

        :param expression:
            The expression for which it is necessary to check whether they hold or not.

        :param value:
            The value assumed by a binary variable when the respective expression holds, either 0 or 1.

        :param name:
            The constraint name.

        :return:
            The backend itself.

        :raise `BackendError`:
            If the backend cannot handle indicator variables.
        """
        raise BackendError(unsupported='indicator constraints')

    def is_greater(self, a, b) -> np.ndarray:
        """Builds auxiliary binary indicator variables which take value one if the expressions in the first array are
         greater than the expressions in the second array. Please note that this is enforced via indicator constraints
         so that if z[i] == 1 -> a[i] >= b[i] and if z[i] == 0 -> a[i] <= b[i], thus in case a[i] is strictly equal to
         b[i] the indicator variable z[i] can assume both values.

        :param a:
            The first array/value representing the left-hand sides.

        :param b:
            Either a single reference value or a second array of expressions representing the right-hand sides.

        :return:
            The array of binary indicator variables.

        :raise `BackendError`:
            If the backend cannot handle indicator variables.
        """
        # if b is an array of expressions, the i-th right hand side will be b[i] (with b flattened), otherwise it is b
        if isinstance(b, np.ndarray):
            b = b.flatten()
            rhs = lambda i: b[i]
        else:
            rhs = lambda i: b
        a = np.atleast_1d(a)
        z = self.add_binary_variables(a.size)
        self.add_indicator_constraints(z, expressions=[lhs >= rhs(i) for i, lhs in enumerate(a.flatten())], value=1)
        self.add_indicator_constraints(z, expressions=[lhs <= rhs(i) for i, lhs in enumerate(a.flatten())], value=0)
        return z.reshape(a.shape)

    def is_less(self, a, b) -> np.ndarray:
        """Builds auxiliary binary indicator variables which take value one if the expressions in the first array are
         lower than the expressions in the second array. Please note that this is enforced via indicator constraints
         so that if z[i] == 1 -> a[i] <= b[i] and if z[i] == 0 -> a[i] >= b[i], thus in case a[i] is strictly equal to
         b[i] the indicator variable z[i] can assume both values.

        :param a:
            The first array/value representing the left-hand sides.

        :param b:
            Either a single reference value or a second array of expressions representing the right-hand sides.

        :return:
            The array of binary indicator variables.

        :raise `BackendError`:
            If the backend cannot handle indicator variables.
        """
        # if b is an array of expressions, the i-th right hand side will be b[i] (with b flattened), otherwise it is b
        if isinstance(b, np.ndarray):
            b = b.flatten()
            rhs = lambda i: b[i]
        else:
            rhs = lambda i: b
        a = np.atleast_1d(a)
        z = self.add_binary_variables(a.size)
        self.add_indicator_constraints(z, expressions=[lhs <= rhs(i) for i, lhs in enumerate(a.flatten())], value=1)
        self.add_indicator_constraints(z, expressions=[lhs >= rhs(i) for i, lhs in enumerate(a.flatten())], value=0)
        return z.reshape(a.shape)

    def sum(self, a: np.ndarray, axis: Optional[int] = None, asarray: bool = False, aux: bool = False) -> Any:
        """Computes the sum of an array of variables.

        :param a:
            An array of model variables.

        :param axis:
            The dimension on which to aggregate or None to aggregate the whole data.

        :param asarray:
            In case the aggregation returns a single expression, whether to return it as a zero-dimensional numpy array
            or as the expression itself.

        :param aux:
            Whether to create continuous auxiliary variables for aggregated results or not.

        :return:
            The variables sum.
        """
        expressions = self._handle_axes(a, operation=self._sum_fn, axis=axis, asarray=asarray)
        return self.aux(expressions, aux_vtype='continuous' if aux else None)

    def sqrt(self, a) -> np.ndarray:
        """Computes the squared roots over an array of variables.

        :param a:
            Either a single model variable or an array of such.

        :return:
            The array of squared roots.
        """
        raise BackendError(unsupported='squared roots')

    def abs(self, a) -> np.ndarray:
        """Computes the absolute values over an array of variables.

        :param a:
            Either a single model variable or an array of such.

        :return:
            The array of absolute values.

        :raise `BackendError`:
            If the backend cannot handle absolute values.
        """
        raise BackendError(unsupported='absolute values')

    def log(self, a) -> np.ndarray:
        """Computes the logarithms over an array of variables.

        :param a:
            Either a single model variable or an array of such.

        :return:
            The array of logarithms.

        :raise `BackendError`:
            If the backend cannot handle logarithms.
        """
        raise BackendError(unsupported='logarithms')

    # noinspection PyMethodMayBeStatic
    def square(self, a) -> np.ndarray:
        """Computes the squared values over an array of variables.

        :param a:
            Either a single model variable or an array of such.

        :return:
            The array of squared values.

        :raise `BackendError`:
            If the backend cannot handle squared values.
        """
        return a ** 2

    def min(self, a: np.ndarray, axis: Optional[int] = None, asarray: bool = False, aux: bool = False) -> Any:
        """Returns the minimum over an array of variables.

        :param a:
            Either a single model variable or an array of such.

        :param axis:
            The dimension on which to aggregate or None to aggregate the whole data.

        :param asarray:
            In case the aggregation returns a single expression, whether to return it as a zero-dimensional numpy array
            or as the expression itself.

        :param aux:
            Whether to create continuous auxiliary variables for aggregated results or not.

        :return:
            The array minimum.

        :raise `BackendError`:
            If the backend cannot handle min values.
        """
        raise BackendError(unsupported='min values')

    def max(self, a: np.ndarray, axis: Optional[int] = None, asarray: bool = False, aux: bool = False) -> Any:
        """Returns the maximum over an array of variables.

        :param a:
            Either a single model variable or an array of such.

        :param axis:
            The dimension on which to aggregate or None to aggregate the whole data.

        :param asarray:
            In case the aggregation returns a single expression, whether to return it as a zero-dimensional numpy array
            or as the expression itself.

        :param aux:
            Whether to create continuous auxiliary variables for aggregated results or not.

        :return:
            The array minimum.

        :raise `BackendError`:
            If the backend cannot handle max values.
        """
        raise BackendError(unsupported='max values')

    def mean(self,
             a: np.ndarray,
             axis: Optional[int] = None,
             asarray: bool = False,
             aux: bool = False) -> Any:
        """Computes the mean of an array of variables.

        :param a:
            An array of model variables.

        :param axis:
            The dimension on which to aggregate or None to aggregate the whole data.

        :param asarray:
            In case the aggregation returns a single expression, whether to return it as a zero-dimensional numpy array
            or as the expression itself.

        :param aux:
            Whether to create continuous auxiliary variables for aggregated results or not.

        :return:
            The variables mean.
        """
        sum_expression = self.sum(a, axis=axis, asarray=asarray, aux=aux)
        num_aggregated = a.size if axis is None else a.shape[axis]
        return self.aux(sum_expression / num_aggregated, aux_vtype=None)

    def var(self,
            a: np.ndarray,
            axis: Optional[int] = None,
            definition: bool = False,
            asarray: bool = False,
            aux: bool = False) -> Any:
        """Computes the variance of an array of variables.

        :param a:
            An array of model variables.

        :param axis:
            The dimension on which to aggregate or None to aggregate the whole data.

        :param definition:
            If True, computes the covariance as in the definition, i.e., var(a) = E[(a - E[a]) ^ 2].
            Otherwise, computes it as the difference of expected values, i.e., var(a, b) = E[a ^ 2] - E[a] ^ 2.

        :param asarray:
            In case the aggregation returns a single expression, whether to return it as a zero-dimensional numpy array
            or as the expression itself.

        :param aux:
            Whether to create continuous auxiliary variables for aggregated results or not.

        :return:
            The variables variance.
        """
        # if no axis is specified, simply compute the mean (which will be a single expression), then compute the array
        # of differences and shape it correctly, otherwise compute the means over the requested axis (which will be in
        # the form of an array), then compute the differences and eventually reshape them accordingly
        if axis is None:
            return self.cov(a, a, definition=definition, asarray=asarray, aux=aux)
        else:
            # check that the axis is in bound and handle negative axis values
            assert axis in range(-a.ndim, a.ndim), f"Axis {axis} is out of bound for array with {a.ndim} dimensions"
            axis = axis % a.ndim
            # compute partial means
            a_mean = self.mean(a, axis=axis, aux=aux)
            # transpose the array in order to bring the axis-th dimension to the front so to compute the differences of
            # each row in the given axis with respect to the mean of that axis, then transpose the axis from the front
            # back to its place since we want to match exactly the original shape
            if definition:
                a = a.transpose(self._transposed_axes(dim=a.ndim, index=axis, place=0))
                a_residuals = self.subtract(a, a_mean)
                a_residuals = a_residuals.transpose(self._transposed_axes(dim=a.ndim, index=0, place=axis))
                # we can finally compute the mean of the squared differences over the given axis
                squared_residuals = self.square(a_residuals)
                return self.mean(squared_residuals, axis=axis, asarray=asarray, aux=aux)
            else:
                a_squared = self.square(a)
                mean_squared = self.mean(a_squared, axis=axis, aux=aux)
                squared_mean = self.square(a_mean)
                var_expression = self.subtract(mean_squared, squared_mean)
                return np.array(var_expression) if asarray else var_expression

    def cov(self, a: np.ndarray, b: np.ndarray, definition: bool = False, asarray: bool = False, aux: bool = False):
        """Returns the covariance between the vectors a and b.

        :param a:
            The first (one-dimensional) array.

        :param b:
            The second (one-dimensional) array.

        :param definition:
            If True, computes the covariance as in the definition, i.e., cov(a, b) = E[(a - E[a]) * (b - E[b])].
            Otherwise, computes it as the difference of expected values, i.e., cov(a, b) = E[a * b] - E[a] * E[b].

        :param asarray:
            In case the aggregation returns a single expression, whether to return it as a zero-dimensional numpy array
            or as the expression itself.

        :param aux:
            Whether to create continuous auxiliary variables for aggregated results or not.

        :return:
            The covariance between a and b.
        """
        a_mean = self.mean(a, aux=aux)
        b_mean = self.mean(b, aux=aux)
        if definition:
            a_residuals = self.subtract(a, a_mean)
            b_residuals = self.subtract(b, b_mean)
            mul_residuals = self.multiply(a_residuals, b_residuals)
            return self.mean(mul_residuals, asarray=asarray, aux=aux)
        else:
            ab = self.multiply(a, b)
            ab_mean = self.mean(ab, aux=aux)
            mul_means = self.multiply(a_mean, b_mean)
            cov_expression = self.subtract(ab_mean, mul_means)
            return np.array(cov_expression) if asarray else cov_expression

    # noinspection PyMethodMayBeStatic
    def add(self, a, b):
        """Performs the pairwise sum between two arrays.

        :param a:
            Either a single model variable or an array of such.

        :param b:
            Either a single model variable or an array of such.

        :return:
            The array of pairwise sums between a and b.
        """
        return a + b

    # noinspection PyMethodMayBeStatic
    def subtract(self, a, b):
        """Performs the pairwise subtraction between two arrays.

        :param a:
            Either a single model variable or an array of such.

        :param b:
            Either a single model variable or an array of such.

        :return:
            The array of pairwise differences between a and b.
        """
        return a - b

    # noinspection PyMethodMayBeStatic
    def multiply(self, a, b):
        """Performs the pairwise product between two arrays.

        :param a:
            Either a single model variable or an array of such.

        :param b:
            Either a single model variable or an array of such.

        :return:
            The array of pairwise products between a and b.
        """
        return a * b

    def divide(self, a, b):
        """Performs the pairwise division between two arrays.

        :param a:
            Either a single model variable or an array of such.

        :param b:
            Either a single model variable or an array of such.

        :return:
            The array of pairwise divisions between a and b.
        """
        return a / b

    def dot(self, a, b, asarray: bool = False, aux: bool = False) -> Any:
        """Performs the dot product between two arrays.

        :param a:
            The first array (at most 2d).

        :param b:
            The second array (at most 2d).

        :param asarray:
            In case the aggregation returns a single expression, whether to return it as a zero-dimensional numpy array
            or as the expression itself.

        :param aux:
            Whether to create continuous auxiliary variables for aggregated results or not.

        :return:
            The dot product between a and b.
        """
        dot_expression = self.aux(expressions=a @ b, aux_vtype='continuous' if aux else None)
        return np.array(dot_expression) if asarray else dot_expression

    def norm_0(self, a: np.ndarray, axis: Optional[int] = None, asarray: bool = False, aux: bool = False) -> Any:
        """Compute the norm 0 of an array.

        :param a:
            The first array (at most 1d if axis is None).

        :param axis:
            The dimension on which to aggregate or None to aggregate the whole data.

        :param asarray:
            In case the aggregation returns a single expression, whether to return it as a zero-dimensional numpy array
            or as the expression itself.

        :param aux:
            Whether to create continuous auxiliary variables for aggregated results or not.

        :return:
            The norm 0 of the input array.

        :raise `BackendError`:
            If the backend cannot handle equalities.
        """
        raise BackendError(unsupported='equalities')

    def norm_1(self, a: np.ndarray, axis: Optional[int] = None, asarray: bool = False, aux: bool = False) -> Any:
        """Compute the norm 1 of an array.

        :param a:
            The first array (at most 1d if axis is None).

        :param axis:
            The dimension on which to aggregate or None to aggregate the whole data.

        :param asarray:
            In case the aggregation returns a single expression, whether to return it as a zero-dimensional numpy array
            or as the expression itself.

        :param aux:
            Whether to create continuous auxiliary variables for aggregated results or not.

        :return:
            The norm 1 of the input array.

        :raise `BackendError`:
            If the backend cannot handle absolute variables.
        """
        norm = self._handle_axes(a, operation=lambda v: self.sum(self.abs(v)), axis=axis, asarray=asarray)
        return self.aux(expressions=norm, aux_vtype='continuous' if aux else None)

    def norm_2(self,
               a: np.ndarray,
               squared: bool = True,
               axis: Optional[int] = None,
               asarray: bool = False,
               aux: bool = False) -> Any:
        """Compute the norm 2 of an array.

        :param a:
            The first array (at most 1d if axis is None).

        :param squared:
            Whether to return the squared norm 2, or the default norm 2 (which is under the square root).

        :param axis:
            The dimension on which to aggregate or None to aggregate the whole data.

        :param asarray:
            In case the aggregation returns a single expression, whether to return it as a zero-dimensional numpy array
            or as the expression itself.

        :param aux:
            Whether to create continuous auxiliary variables for aggregated results or not.

        :return:
            The norm 2 of the input array.

        :raise `BackendError`:
            If the backend cannot handle squared values or square roots.
        """
        if squared:
            operation = lambda v: self.dot(v, v)
        else:
            operation = lambda v: self.sqrt(self.dot(v, v))
        norm = self._handle_axes(a, operation=operation, axis=axis, asarray=asarray)
        return self.aux(expressions=norm, aux_vtype='continuous' if aux else None)

    def norm_inf(self, a: np.ndarray, axis: Optional[int] = None, asarray: bool = False, aux: bool = False) -> Any:
        """Compute the norm infinity of an array.

        :param a:
            The first array (at most 1d if axis is None).

        :param axis:
            The dimension on which to aggregate or None to aggregate the whole data.

        :param asarray:
            In case the aggregation returns a single expression, whether to return it as a zero-dimensional numpy array
            or as the expression itself.

        :param aux:
            Whether to create continuous auxiliary variables for aggregated results or not.

        :return:
            The norm infinity of the input array.

        :raise `BackendError`:
            If the backend cannot handle absolute and maximum values.
        """
        norm = self._handle_axes(a, operation=lambda v: self.max(self.abs(v)), axis=axis, asarray=asarray)
        return self.aux(expressions=norm, aux_vtype='continuous' if aux else None)
