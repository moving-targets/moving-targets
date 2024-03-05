from typing import List, Dict, Optional, Any, Union

import numpy as np

from moving_targets.masters.backends import Backend
from moving_targets.util.errors import BackendError, not_implemented_message
from test.test_abstract import TestAbstract


class TestBackend(TestAbstract):
    _SIZES: Dict[str, List[int]] = {
        '1D': [TestAbstract.NUM_SAMPLES],
        '1D inv': [TestAbstract.NUM_CLASSES],
        '2D': [TestAbstract.NUM_SAMPLES, TestAbstract.NUM_CLASSES],
        '2D inv': [TestAbstract.NUM_CLASSES, TestAbstract.NUM_SAMPLES],
        '3D': [TestAbstract.NUM_SAMPLES, TestAbstract.NUM_CLASSES, TestAbstract.NUM_FEATURES],
    }
    """A dictionary which associates to each vector dimension a list representing the vector shape."""

    @classmethod
    def _backend(cls) -> Backend:
        """The `Backend` instance to be tested."""
        raise NotImplementedError(not_implemented_message(name='_backend'))

    @classmethod
    def _unsupported(cls) -> List[str]:
        """The list of unsupported operations."""
        raise NotImplementedError(not_implemented_message(name='_unsupported'))

    @classmethod
    def _get_name(cls, variable: Any) -> str:
        """Gets a formatted variable name."""
        raise NotImplementedError(not_implemented_message(name='_get_variable_name'))

    def _check_exception(self, operation: Optional[str], exception: Exception):
        """Checks if the correct exception is raised."""
        if self.__class__.__name__ == 'TestBackend':
            # if we are in the abstract test, check that the error is "NotImplementedError"
            self.assertIsInstance(exception, NotImplementedError)
        elif operation is None or operation not in self._unsupported():
            # if the operation is None or not in the unsupported,, it means that it must be supported, thus we fail
            self.fail()
        else:
            # otherwise, check that the error is a "BackendError"
            self.assertIsInstance(exception, BackendError)

    def _test_indicator_operations(self, dim: str, operation: str, **op_args):
        """Checks the correctness, using an array of the given dimensions, of the given the backend indicator operation
        that is instantiated according to the additional parameters"""
        try:
            rng = np.random.default_rng(self.SEED)
            backend = self._backend()
            sizes = (self._SIZES['1D'], 1) if dim == '0D' else (self._SIZES[dim], self._SIZES[dim])
            mt_operation = getattr(backend, operation)
            ref_operation = getattr(np, operation.lstrip('is_'))
            if operation in self._unsupported():
                backend.build()
                with self.assertRaises(BackendError):
                    a, b = [rng.random(size=size) for size in sizes]
                    a = backend.add_constants(a)
                    b = backend.add_constant(b[0]) if dim == '0D' else backend.add_constants(b)
                    mt_operation(a, b, **op_args)
                backend.clear()
            else:
                for i in range(self.NUM_TESTS):
                    # build random vectors of reference values and compute the operation result
                    ref_a, ref_b = [rng.random(size=size) for size in sizes]
                    ref_result = ref_operation(ref_a, ref_b, **op_args)
                    # build constant model variables vector(s) from values then obtain the operation result
                    backend.build()
                    mt_a = backend.add_constants(ref_a)
                    mt_b = backend.add_constant(ref_b[0]) if dim == '0D' else backend.add_constants(ref_b)
                    mt_result = mt_operation(mt_a, mt_b, **op_args)
                    backend.solve()
                    mt_result = backend.get_value(mt_result)
                    backend.clear()
                    # compare the two results
                    msg = f'Error at it. {i} with dim {dim}, {mt_result} != {ref_result}'
                    self.assertTrue(np.allclose(mt_result, ref_result, atol=10 ** -self.PLACES), msg=msg)
        except Exception as exception:
            self._check_exception(operation='indicator', exception=exception)

    def _test_numeric_operation(self, *dims: str, operation: str, **op_args):
        """Checks the correctness, using an array of the given dimensions, of the given the backend numeric operation
        that is instantiated according to the additional parameters (which must shared between the backend and numpy)"""
        try:
            rng = np.random.default_rng(self.SEED)
            backend = self._backend()
            mt_operation = getattr(backend, operation)
            ref_operation = getattr(np, operation)
            if operation in self._unsupported():
                backend.build()
                with self.assertRaises(BackendError):
                    values = [rng.random(size=self._SIZES[d]) for d in dims]
                    values = [backend.add_constants(v) for v in values]
                    mt_operation(*values, **op_args)
                backend.clear()
            else:
                for i in range(self.NUM_TESTS):
                    # build random vector(s) of reference values and compute the operation result
                    ref_values = [rng.random(size=self._SIZES[d]) for d in dims]
                    if operation == 'cov':
                        ref_result = np.cov(*ref_values, bias=True)[0, 1]
                    elif operation == 'var':
                        args = {k: v for k, v in op_args.items() if k != 'definition'}
                        ref_result = np.var(*ref_values, **args)
                    else:
                        ref_result = ref_operation(*ref_values, **op_args)
                    # build constant model variables vector(s) from values then obtain the operation result
                    backend.build()
                    mt_values = [backend.add_constants(v) for v in ref_values]
                    mt_result = mt_operation(*mt_values, **op_args)
                    backend.solve()
                    mt_result = backend.get_value(mt_result)
                    backend.clear()
                    # compare the two results
                    msg = f'Error at it. {i}, in op. {operation}, with dims {list(dims)}, {mt_result} != {ref_result}'
                    self.assertTrue(np.allclose(mt_result, ref_result, atol=10 ** -self.PLACES), msg=msg)
        except Exception as exception:
            self._check_exception(operation=operation, exception=exception)

    def _test_norm_operation(self, dim: str, norm: Union[int, str], **op_args):
        """Checks the correctness, using an array of the given dimensions, of the given the backend numeric operation
        that is instantiated according to the additional parameters (which must shared between the backend and numpy)"""
        try:
            rng = np.random.default_rng(self.SEED)
            backend = self._backend()
            mt_operation = getattr(backend, f'norm_{norm}')
            if f'norm_{norm}' in self._unsupported():
                backend.build()
                with self.assertRaises(BackendError):
                    values = rng.random(size=self._SIZES[dim])
                    values = backend.add_constants(values)
                    mt_operation(values, **op_args)
                backend.clear()
            else:
                for i in range(self.NUM_TESTS):
                    # build random vector(s) of reference values and compute the operation result
                    ref_values = rng.random(size=self._SIZES[dim])
                    if norm == 0:
                        ref_result = np.count_nonzero(ref_values, **op_args)
                    elif norm == 1:
                        ref_result = np.linalg.norm(ref_values, ord=1, **op_args)
                    elif norm == 2:
                        args = {k: v for k, v in op_args.items() if k != 'squared'}
                        ref_result = np.linalg.norm(ref_values, ord=2, **args) ** 2
                    elif norm == 'inf':
                        ref_result = np.linalg.norm(ref_values, ord=np.inf, **op_args)
                    else:
                        raise AssertionError(f"Unexpected norm {norm}")
                    # build constant model variables vector(s) from values then obtain the operation result
                    backend.build()
                    mt_values = backend.add_constants(ref_values)
                    mt_result = mt_operation(mt_values, **op_args)
                    backend.solve()
                    mt_result = backend.get_value(mt_result)
                    backend.clear()
                    # compare the two results
                    msg = f'Error at it. {i}, in norm_{norm}, with dim {dim}, {mt_result} != {ref_result}'
                    self.assertTrue(np.allclose(mt_result, ref_result, atol=10 ** -self.PLACES), msg=msg)
        except Exception as exception:
            self._check_exception(operation=f'norm_{norm}', exception=exception)

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
            ref_value = self.NUM_SAMPLES + 1
            # test minimize function and variables bounds (expected objective is -ref_value)
            backend = backend.build()
            variable = backend.add_continuous_variable(lb=-1, ub=1, name='variable')
            variables = backend.add_continuous_variables(self.NUM_SAMPLES, lb=-1, ub=1, name='variables')
            mt_value = backend.minimize(variable + backend.sum(variables)).solve().get_objective()
            self.assertAlmostEqual(mt_value, -ref_value, places=self.PLACES)
            backend.clear()
            # test maximize function and variables bounds (expected objective is +ref_value)
            backend = backend.build()
            variable = backend.add_continuous_variable(lb=-1, ub=1, name='variable')
            variables = backend.add_continuous_variables(self.NUM_SAMPLES, lb=-1, ub=1, name='variables')
            mt_value = backend.maximize(variable + backend.sum(variables)).solve().get_objective()
            self.assertAlmostEqual(mt_value, ref_value, places=self.PLACES)
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
                    backend.add_constraint(var >= 0, name=None)
                    backend.add_constraint(var >= 0, name='test')
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
                    backend.add_constraints([v >= 0 for v in var.flatten()], name=None)
                    backend.add_constraints([v >= 0 for v in var.flatten()], name='test')
                    backend.solve()
                    self.assertListEqual([self._get_name(v) for v in var], _expected_names(*keys, _name='var'))
            backend.clear()
        except Exception as exception:
            self._check_exception(operation='variables', exception=exception)

    # INDICATOR OPERATIONS

    def test_is_greater(self):
        self._test_indicator_operations('0D', operation='is_greater')
        self._test_indicator_operations('1D', operation='is_greater')
        self._test_indicator_operations('2D', operation='is_greater')
        self._test_indicator_operations('3D', operation='is_greater')

    def test_is_less(self):
        self._test_indicator_operations('0D', operation='is_less')
        self._test_indicator_operations('1D', operation='is_less')
        self._test_indicator_operations('2D', operation='is_less')
        self._test_indicator_operations('3D', operation='is_less')

    # UNARY NUMERIC OPERATIONS

    def test_sum(self):
        self._test_numeric_operation('1D', operation='sum')
        self._test_numeric_operation('2D', operation='sum')
        self._test_numeric_operation('3D', operation='sum')

    def test_sum_axis(self):
        self._test_numeric_operation('1D', operation='sum', axis=-1)
        self._test_numeric_operation('1D', operation='sum', axis=0)
        self._test_numeric_operation('2D', operation='sum', axis=-1)
        self._test_numeric_operation('2D', operation='sum', axis=0)
        self._test_numeric_operation('2D', operation='sum', axis=1)
        self._test_numeric_operation('3D', operation='sum', axis=-1)
        self._test_numeric_operation('3D', operation='sum', axis=0)
        self._test_numeric_operation('3D', operation='sum', axis=1)
        self._test_numeric_operation('3D', operation='sum', axis=2)

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

    def test_square(self):
        self._test_numeric_operation('1D', operation='square')
        self._test_numeric_operation('2D', operation='square')
        self._test_numeric_operation('3D', operation='square')

    def test_min(self):
        self._test_numeric_operation('1D', operation='min')
        self._test_numeric_operation('2D', operation='min')
        self._test_numeric_operation('3D', operation='min')

    def test_min_axis(self):
        self._test_numeric_operation('1D', operation='min', axis=-1)
        self._test_numeric_operation('1D', operation='min', axis=0)
        self._test_numeric_operation('2D', operation='min', axis=-1)
        self._test_numeric_operation('2D', operation='min', axis=0)
        self._test_numeric_operation('2D', operation='min', axis=1)
        self._test_numeric_operation('3D', operation='min', axis=-1)
        self._test_numeric_operation('3D', operation='min', axis=0)
        self._test_numeric_operation('3D', operation='min', axis=1)
        self._test_numeric_operation('3D', operation='min', axis=2)

    def test_max(self):
        self._test_numeric_operation('1D', operation='max')
        self._test_numeric_operation('2D', operation='max')
        self._test_numeric_operation('3D', operation='max')

    def test_max_axis(self):
        self._test_numeric_operation('1D', operation='max', axis=-1)
        self._test_numeric_operation('1D', operation='max', axis=0)
        self._test_numeric_operation('2D', operation='max', axis=-1)
        self._test_numeric_operation('2D', operation='max', axis=0)
        self._test_numeric_operation('2D', operation='max', axis=1)
        self._test_numeric_operation('3D', operation='max', axis=-1)
        self._test_numeric_operation('3D', operation='max', axis=0)
        self._test_numeric_operation('3D', operation='max', axis=1)
        self._test_numeric_operation('3D', operation='max', axis=2)

    def test_mean(self):
        self._test_numeric_operation('1D', operation='mean')
        self._test_numeric_operation('2D', operation='mean')
        self._test_numeric_operation('3D', operation='mean')

    def test_mean_axis(self):
        self._test_numeric_operation('1D', operation='mean', axis=-1)
        self._test_numeric_operation('1D', operation='mean', axis=0)
        self._test_numeric_operation('2D', operation='mean', axis=-1)
        self._test_numeric_operation('2D', operation='mean', axis=0)
        self._test_numeric_operation('2D', operation='mean', axis=1)
        self._test_numeric_operation('3D', operation='mean', axis=-1)
        self._test_numeric_operation('3D', operation='mean', axis=0)
        self._test_numeric_operation('3D', operation='mean', axis=1)
        self._test_numeric_operation('3D', operation='mean', axis=2)

    def test_var(self):
        self._test_numeric_operation('1D', operation='var', definition=True)
        self._test_numeric_operation('1D', operation='var', definition=False)
        self._test_numeric_operation('2D', operation='var', definition=True)
        self._test_numeric_operation('2D', operation='var', definition=False)
        self._test_numeric_operation('3D', operation='var', definition=True)
        self._test_numeric_operation('3D', operation='var', definition=False)

    def test_var_axis(self):
        self._test_numeric_operation('1D', operation='var', axis=-1, definition=True)
        self._test_numeric_operation('1D', operation='var', axis=-1, definition=False)
        self._test_numeric_operation('1D', operation='var', axis=0, definition=True)
        self._test_numeric_operation('1D', operation='var', axis=0, definition=False)
        self._test_numeric_operation('2D', operation='var', axis=-1, definition=True)
        self._test_numeric_operation('2D', operation='var', axis=-1, definition=False)
        self._test_numeric_operation('2D', operation='var', axis=0, definition=True)
        self._test_numeric_operation('2D', operation='var', axis=0, definition=False)
        self._test_numeric_operation('2D', operation='var', axis=1, definition=True)
        self._test_numeric_operation('2D', operation='var', axis=1, definition=False)
        self._test_numeric_operation('3D', operation='var', axis=-1, definition=True)
        self._test_numeric_operation('3D', operation='var', axis=-1, definition=False)
        self._test_numeric_operation('3D', operation='var', axis=0, definition=True)
        self._test_numeric_operation('3D', operation='var', axis=0, definition=False)
        self._test_numeric_operation('3D', operation='var', axis=1, definition=True)
        self._test_numeric_operation('3D', operation='var', axis=1, definition=False)
        self._test_numeric_operation('3D', operation='var', axis=2, definition=True)
        self._test_numeric_operation('3D', operation='var', axis=2, definition=False)

    # BINARY NUMERIC OPERATIONS

    def test_cov(self):
        self._test_numeric_operation('1D', '1D', operation='cov', definition=True)
        self._test_numeric_operation('1D', '1D', operation='cov', definition=False)

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

    def test_norm_0(self):
        self._test_norm_operation('1D', norm=0)
        self._test_norm_operation('1D', norm=0, axis=-1)
        self._test_norm_operation('1D', norm=0, axis=0)
        self._test_norm_operation('2D', norm=0, axis=-1)
        self._test_norm_operation('2D', norm=0, axis=0)
        self._test_norm_operation('2D', norm=0, axis=1)
        self._test_norm_operation('3D', norm=0, axis=-1)
        self._test_norm_operation('3D', norm=0, axis=0)
        self._test_norm_operation('3D', norm=0, axis=1)
        self._test_norm_operation('3D', norm=0, axis=2)

    def test_norm_1(self):
        self._test_norm_operation('1D', norm=1)
        self._test_norm_operation('1D', norm=1, axis=-1)
        self._test_norm_operation('1D', norm=1, axis=0)
        self._test_norm_operation('2D', norm=1, axis=-1)
        self._test_norm_operation('2D', norm=1, axis=0)
        self._test_norm_operation('2D', norm=1, axis=1)
        self._test_norm_operation('3D', norm=1, axis=-1)
        self._test_norm_operation('3D', norm=1, axis=0)
        self._test_norm_operation('3D', norm=1, axis=1)
        self._test_norm_operation('3D', norm=1, axis=2)

    def test_norm_2(self):
        self._test_norm_operation('1D', norm=2, squared=True)
        self._test_norm_operation('1D', norm=2, squared=True, axis=-1)
        self._test_norm_operation('1D', norm=2, squared=True, axis=0)
        self._test_norm_operation('2D', norm=2, squared=True, axis=-1)
        self._test_norm_operation('2D', norm=2, squared=True, axis=0)
        self._test_norm_operation('2D', norm=2, squared=True, axis=1)
        self._test_norm_operation('3D', norm=2, squared=True, axis=-1)
        self._test_norm_operation('3D', norm=2, squared=True, axis=0)
        self._test_norm_operation('3D', norm=2, squared=True, axis=1)
        self._test_norm_operation('3D', norm=2, squared=True, axis=2)

    def test_norm_inf(self):
        self._test_norm_operation('1D', norm='inf')
        self._test_norm_operation('1D', norm='inf', axis=-1)
        self._test_norm_operation('1D', norm='inf', axis=0)
        self._test_norm_operation('2D', norm='inf', axis=-1)
        self._test_norm_operation('2D', norm='inf', axis=0)
        self._test_norm_operation('2D', norm='inf', axis=1)
        self._test_norm_operation('3D', norm='inf', axis=-1)
        self._test_norm_operation('3D', norm='inf', axis=0)
        self._test_norm_operation('3D', norm='inf', axis=1)
        self._test_norm_operation('3D', norm='inf', axis=2)
