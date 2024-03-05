from typing import Dict

import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

from moving_targets.util.scalers import Scaler
from test.test_abstract import TestAbstract


class TestScalers(TestAbstract):
    """Tests the correctness of the `Scaler` class."""

    FLOAT_DTYPE = np.zeros(1, dtype='float64').dtype

    RNG = np.random.default_rng(seed=TestAbstract.SEED)

    DATA = pd.DataFrame.from_dict({
        '(0, 1)': RNG.random(TestAbstract.NUM_SAMPLES),
        '(-1, 1)': 2 * RNG.random(TestAbstract.NUM_SAMPLES) - 1,
        '{0, 2}': RNG.choice([0, 1, 2], TestAbstract.NUM_SAMPLES),
        '{M, F}': RNG.choice(['M', 'F'], TestAbstract.NUM_SAMPLES)
    })

    # decide name of 'sparse' parameter based on scikit version ('sparse' if <= 1.1, 'sparse_output' if >= 1.2)
    version = sklearn.__version__.split('.')
    kwargs = dict(sparse=False) if int(version[0]) <= 1 and int(version[1]) <= 1 else dict(sparse_output=False)

    SCALED = {
        'none': pd.DataFrame(DATA),
        'norm': pd.DataFrame(
            data=MinMaxScaler().fit_transform(DATA[['(0, 1)', '(-1, 1)', '{0, 2}']]),
            columns=['(0, 1)', '(-1, 1)', '{0, 2}']
        ),
        'std': pd.DataFrame(
            data=StandardScaler().fit_transform(DATA[['(0, 1)', '(-1, 1)', '{0, 2}']]),
            columns=['(0, 1)', '(-1, 1)', '{0, 2}']
        ),
        'onehot': pd.DataFrame(
            data=OneHotEncoder(drop='if_binary', **kwargs).fit_transform(DATA[['{0, 2}', '{M, F}']]),
            columns=['{0, 2}_0', '{0, 2}_1', '{0, 2}_2', '{M, F}_M']
        )
    }

    def _test(self, columns: Dict[str, str], scaler: Scaler, numpy: bool):
        """Checks that the given scaler, when fitted with certain data (which may be either an array or a dataframe),
        behaves as expected in the dictionary of input columns paired with the expected scaling type."""
        # handle input and reference
        one_dimensional = len(columns) == 1 and 'onehot' not in columns.values()
        input_data = self.DATA[[k for k in columns.keys()]].squeeze()
        reference_data = pd.DataFrame()
        for c, m in columns.items():
            if m == 'onehot':
                if c == '{0, 2}':
                    reference_data['{0, 2}_0'] = self.SCALED['onehot']['{0, 2}_0']
                    reference_data['{0, 2}_1'] = self.SCALED['onehot']['{0, 2}_1']
                    reference_data['{0, 2}_2'] = self.SCALED['onehot']['{0, 2}_2']
                elif c == '{M, F}':
                    reference_data['{M, F}_M'] = self.SCALED['onehot']['{M, F}_M']
                else:
                    raise AssertionError(f'Unexpected column {c}')
            else:
                reference_data[c] = self.SCALED[m][c]
        # handle types and perform dtype checks based
        if numpy:
            input_data = np.array(input_data)
            reference_data = np.array(reference_data)
            scaled_data = scaler.fit_transform(input_data)
            inverted_data = scaler.inverse_transform(scaled_data)
            self.assertIsInstance(scaled_data, np.ndarray)
            self.assertIsInstance(inverted_data, np.ndarray)
            self.assertEqual(self.FLOAT_DTYPE, scaled_data.dtype)
            self.assertEqual(input_data.dtype, inverted_data.dtype)
        else:
            scaled_data = scaler.fit_transform(input_data)
            inverted_data = scaler.inverse_transform(scaled_data)
            if one_dimensional:
                self.assertIsInstance(scaled_data, pd.Series)
                self.assertIsInstance(inverted_data, pd.Series)
                self.assertEqual(input_data.dtypes, inverted_data.dtypes)
                self.assertEqual(self.FLOAT_DTYPE, scaled_data.dtypes)
            else:
                self.assertIsInstance(scaled_data, pd.DataFrame)
                self.assertIsInstance(inverted_data, pd.DataFrame)
                self.assertDictEqual(input_data.dtypes.to_dict(), inverted_data.dtypes.to_dict())
                self.assertDictEqual({c: self.FLOAT_DTYPE for c in scaled_data.columns}, scaled_data.dtypes.to_dict())
        # check shapes
        self.assertEqual(scaled_data.shape, input_data.shape if one_dimensional else reference_data.shape)
        self.assertEqual(inverted_data.shape, input_data.shape)
        # build dataframes for compatibility
        input_data = pd.DataFrame(input_data, columns=scaler.input_columns)
        scaled_data = pd.DataFrame(scaled_data, columns=scaler.output_columns)
        inverted_data = pd.DataFrame(inverted_data, columns=scaler.input_columns)
        reference_data = pd.DataFrame(reference_data, columns=scaler.output_columns)
        # check equality between scaled and reference data, and between input and inverted data
        for c in scaler.input_columns:
            msg = str(pd.concat((input_data[c], inverted_data[c]), axis=1))
            # use try-except to deal with data types (i.e., for float-like values the check is 'all close')
            try:
                inp, inv = input_data[c].astype(np.float64), inverted_data[c].astype(np.float64)
                self.assertTrue(np.allclose(inp, inv, atol=10 ** -self.PLACES), msg=msg)
            except ValueError:
                self.assertTrue(np.array_equal(input_data[c], inverted_data[c]), msg=msg)
        for c in scaler.output_columns:
            # here there is no need to deal with data types since data should always be float-like
            msg = str(pd.concat((scaled_data[c], reference_data[c]), axis=1))
            self.assertTrue(np.allclose(scaled_data[c], reference_data[c], atol=10 ** -self.PLACES), msg=msg)

    def test_numpy_1d(self):
        self._test(columns={'(0, 1)': 'none'}, scaler=Scaler('none'), numpy=True)

    def test_pandas_1d(self):
        self._test(columns={'(0, 1)': 'none'}, scaler=Scaler('none'), numpy=False)

    def test_numpy_none(self):
        self._test(columns={'(0, 1)': 'none', '(-1, 1)': 'none', '{0, 2}': 'none'}, scaler=Scaler('none'), numpy=True)

    def test_pandas_none(self):
        self._test(columns={'(0, 1)': 'none', '(-1, 1)': 'none', '{0, 2}': 'none'}, scaler=Scaler('none'), numpy=False)

    def test_numpy_norm(self):
        self._test(columns={'(0, 1)': 'norm', '(-1, 1)': 'norm', '{0, 2}': 'norm'}, scaler=Scaler('norm'), numpy=True)

    def test_pandas_norm(self):
        self._test(columns={'(0, 1)': 'norm', '(-1, 1)': 'norm', '{0, 2}': 'norm'}, scaler=Scaler('norm'), numpy=False)

    def test_numpy_std(self):
        self._test(columns={'(0, 1)': 'std', '(-1, 1)': 'std', '{0, 2}': 'std'}, scaler=Scaler('std'), numpy=True)

    def test_pandas_std(self):
        self._test(columns={'(0, 1)': 'std', '(-1, 1)': 'std', '{0, 2}': 'std'}, scaler=Scaler('std'), numpy=False)

    def test_numpy_onehot(self):
        self._test(columns={'{0, 2}': 'onehot', '{M, F}': 'onehot'}, scaler=Scaler('onehot'), numpy=True)

    def test_pandas_onehot(self):
        self._test(columns={'{0, 2}': 'onehot', '{M, F}': 'onehot'}, scaler=Scaler('onehot'), numpy=False)

    def test_numpy_mixed(self):
        methods = {'(0, 1)': 'norm', '(-1, 1)': 'std', '{0, 2}': 'none', '{M, F}': 'onehot'}
        self._test(columns=methods, scaler=Scaler(*methods.values()), numpy=True)

    def test_pandas_mixed(self):
        methods = {'(0, 1)': 'norm', '(-1, 1)': 'std', '{M, F}': 'onehot'}
        self._test(columns={**methods, '{0, 2}': 'none'}, scaler=Scaler(default_method='none', **methods), numpy=False)
