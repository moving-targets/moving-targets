from typing import List, Dict, Optional, Any, Union

import numpy as np

from moving_targets.masters.backends import Backend
from moving_targets.util.errors import BackendError, not_implemented_message
from test.abstract import AbstractTest


class TestBackend(AbstractTest):
    _SIZES: Dict[str, List[int]] = {
        '1D': [AbstractTest.NUM_SAMPLES],
        '1D inv': [AbstractTest.NUM_CLASSES],
        '2D': [AbstractTest.NUM_SAMPLES, AbstractTest.NUM_CLASSES],
        '2D inv': [AbstractTest.NUM_CLASSES, AbstractTest.NUM_SAMPLES],
        '3D': [AbstractTest.NUM_SAMPLES, AbstractTest.NUM_CLASSES, AbstractTest.NUM_FEATURES],
    }
    """A dictionary which associates to each vector dimension a list representing the vector shape."""

    @classmethod
    def _backend(cls) -> Backend:
        """The `Backend` instance to be tested.

        :return
            The `Backend` instance
        """
        raise NotImplementedError(not_implemented_message(name='_backend'))

    @classmethod
    def _unsupported(cls) -> List[str]:
        """The list of unsupported operations.

        :return:
            A list of strings representing the unsupported operations.
        """
        raise NotImplementedError(not_implemented_message(name='_unsupported'))

    @classmethod
    def _get_name(cls, variable: Any) -> str:
        """Gets a formatted variable.

        :param variable:
            The variable.

        :return:
            The formatted variable name.
        """
        raise NotImplementedError(not_implemented_message(name='_get_variable_name'))

    def _check_exception(self, operation: Optional[str], exception: Exception):
        if self.__class__.__name__ == 'TestBackend':
            # if we are in the abstract test, check that the error is "NotImplementedError"
            self.assertIsInstance(exception, NotImplementedError)
        elif operation is None or operation not in self._unsupported():
            # if the operation is None or not in the unsupported,, it means that it must be supported, thus we fail
            self.fail()
        else:
            # otherwise, check that the error is a "BackendError"
            self.assertIsInstance(exception, BackendError)

    def _test_numeric_operation(self, *dims: str, operation: str):
        """Implements the main strategy to test the backend numeric operations.

        :param dims:
            The dimension(s) of the vector(s) to be passed to the backend/numpy operation.

        :param operation:
            The operation name, which must be the same between the backend and numpy (which is used as reference).
        """
        try:
            np.random.seed(self.SEED)
            backend = self._backend()
            mt_operation = getattr(backend, operation)
            ref_operation = getattr(np, operation)
            if operation in self._unsupported():
                backend.build()
                with self.assertRaises(BackendError):
                    values = [np.random.random(size=self._SIZES[d]) for d in dims]
                    values = [backend.add_constants(v) for v in values]
                    mt_operation(*values)
                backend.clear()
            else:
                for i in range(self.NUM_TESTS):
                    # build random vector(s) of reference values and compute the operation result
                    ref_values = [np.random.random(size=self._SIZES[d]) for d in dims]
                    ref_result = ref_operation(*ref_values)
                    # build constant model variables vector(s) from values then obtain the operation result
                    backend.build()
                    mt_values = [backend.add_constants(v) for v in ref_values]
                    mt_result = mt_operation(*mt_values)
                    backend.solve()
                    mt_result = backend.get_value(mt_result)
                    backend.clear()
                    # compare the two results
                    msg = f'Error at iteration {i} in operation {operation}, {mt_result} != {ref_result}'
                    self.assertTrue(np.allclose(mt_result, ref_result, atol=10 ** -self.PLACES), msg=msg)
        except Exception as exception:
            self._check_exception(operation=operation, exception=exception)

    # TEST MODELLING OPERATIONS

    def test_model(self):
        try:
            # create empty backend
            backend = self._backend()
            self.assertIsNone(backend.model)
            self.assertIsNone(backend.solution)
            # build model
            backend.build()
            self.assertIsNotNone(backend.model)
            self.assertIsNone(backend.solution)
            # solve model
            backend.solve()
            self.assertIsNotNone(backend.model)
            self.assertIsNotNone(backend.solution)
            # clear model
            backend.clear()
            self.assertIsNone(backend.model)
            self.assertIsNone(backend.solution)
        except Exception as exception:
            self._check_exception(operation=None, exception=exception)

    def test_objectives(self):
        try:
            backend = self._backend()
            for operation in ['minimize', 'maximize']:
                backend.build()
                operation = getattr(backend, operation)
                operation(backend.add_constant(1.0))
                backend.solve()
                self.assertAlmostEqual(backend.get_objective(), 1.0, places=self.PLACES)
                backend.clear()
        except Exception as exception:
            self._check_exception(operation='objectives', exception=exception)

    def test_variables(self):
        try:
            backend = self._backend()
            backend.build()
            for dim in ['0D', '1D', '2D', '3D']:
                if dim == '0D':
                    var = backend.add_variable(vtype='continuous', lb=0, ub=1, name='var')
                    backend.solve()
                    self.assertEqual(self._get_name(var), 'var')
                else:
                    def _expected_names(*_keys: int, _name: str = 'var') -> Union[str, List[str]]:
                        if len(_keys) == 1:
                            return [f'{_name}_{i}' for i in range(_keys[0])]
                        else:
                            res = []
                            for i in range(_keys[0]):
                                res += _expected_names(*_keys[1:], _name=f'{_name}_{i}')
                            return res

                    keys = self._SIZES[dim]
                    var = backend.add_variables(*keys, vtype='continuous', lb=0, ub=1, name='var').flatten()
                    backend.solve()
                    self.assertListEqual([self._get_name(v) for v in var], _expected_names(*keys, _name='var'))
            backend.clear()
        except Exception as exception:
            self._check_exception(operation='variables', exception=exception)

    # UNARY NUMERIC OPERATIONS

    def test_sum(self):
        self._test_numeric_operation('1D', operation='sum')
        self._test_numeric_operation('2D', operation='sum')
        self._test_numeric_operation('3D', operation='sum')

    def test_square(self):
        self._test_numeric_operation('1D', operation='square')
        self._test_numeric_operation('2D', operation='square')
        self._test_numeric_operation('3D', operation='square')

    def test_sqrt(self):
        self._test_numeric_operation('1D', operation='sqrt')
        self._test_numeric_operation('2D', operation='sqrt')
        self._test_numeric_operation('3D', operation='sqrt')

    def test_abs(self):
        self._test_numeric_operation('1D', operation='abs')
        self._test_numeric_operation('2D', operation='abs')
        self._test_numeric_operation('3D', operation='abs')

    def test_log(self):
        self._test_numeric_operation('1D', operation='log')
        self._test_numeric_operation('2D', operation='log')
        self._test_numeric_operation('3D', operation='log')

    def test_mean(self):
        self._test_numeric_operation('1D', operation='mean')
        self._test_numeric_operation('2D', operation='mean')
        self._test_numeric_operation('3D', operation='mean')

    def test_var(self):
        self._test_numeric_operation('1D', operation='var')
        self._test_numeric_operation('2D', operation='var')
        self._test_numeric_operation('3D', operation='var')

    # BINARY NUMERIC OPERATIONS

    def test_add(self):
        self._test_numeric_operation('1D', '1D', operation='add')
        self._test_numeric_operation('2D', '2D', operation='add')
        self._test_numeric_operation('3D', '3D', operation='add')

    def test_subtract(self):
        self._test_numeric_operation('1D', '1D', operation='subtract')
        self._test_numeric_operation('2D', '2D', operation='subtract')
        self._test_numeric_operation('3D', '3D', operation='subtract')

    def test_multiply(self):
        self._test_numeric_operation('1D', '1D', operation='multiply')
        self._test_numeric_operation('2D', '2D', operation='multiply')
        self._test_numeric_operation('3D', '3D', operation='multiply')

    def test_divide(self):
        self._test_numeric_operation('1D', '1D', operation='divide')
        self._test_numeric_operation('2D', '2D', operation='divide')
        self._test_numeric_operation('3D', '3D', operation='divide')

    def test_dot(self):
        self._test_numeric_operation('1D', '1D', operation='dot')
        self._test_numeric_operation('1D', '2D', operation='dot')
        self._test_numeric_operation('2D', '1D inv', operation='dot')
        self._test_numeric_operation('2D', '2D inv', operation='dot')
