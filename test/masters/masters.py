from typing import Union, Optional

import numpy as np

from moving_targets import MACS
from moving_targets.learners import Learner
from moving_targets.masters import ClassificationMaster, RegressionMaster
from moving_targets.util import probabilities
from test.abstract import AbstractTest


class TestMasters(AbstractTest):
    class DummyLearner(Learner):
        def __init__(self, regression: bool):
            super(TestMasters.DummyLearner, self).__init__(x_scaler=None, y_scaler=None, stats=False)
            self.regression: bool = regression
            self.p: Optional[np.ndarray] = None

        def _fit(self, x, y, sample_weight=None):
            self.p = y

        def _predict(self, x):
            return self.p if self.regression or self.p.ndim == 2 else probabilities.get_onehot(self.p.astype(int))

    def _test(self, task: Union[int, str], loss: str, vtype: Optional[str], rtype: Optional[str]):
        """Checks that the masters implementations behave correctly with respect to the given parameters."""
        np.random.seed(0)
        x = np.random.random((self.NUM_SAMPLES, self.NUM_FEATURES))
        # handle outputs
        if task == 'regression':
            p = y = np.random.random(self.NUM_SAMPLES)
            master = RegressionMaster(backend='gurobi', loss=loss)
        elif task == 'binary':
            y = np.random.random(self.NUM_SAMPLES).round().astype(int)
            p = y if rtype == 'class' else y.astype(float)
            master = ClassificationMaster(backend='gurobi', loss=loss, rtype=rtype)
            self.assertEqual(master.vtype, vtype)
        elif task == 'multiclass':
            y = np.random.random((self.NUM_SAMPLES, self.NUM_CLASSES)).argmax(axis=1)
            p = y if rtype == 'class' else probabilities.get_onehot(y).astype(float)
            master = ClassificationMaster(backend='gurobi', loss=loss, rtype=rtype)
            self.assertEqual(master.vtype, vtype)
        elif task == 'multilabel':
            y = np.random.random((self.NUM_SAMPLES, self.NUM_CLASSES)).round().astype(int)
            p = y if rtype == 'class' else y.astype(float)
            master = ClassificationMaster(backend='gurobi', loss=loss, rtype=rtype, labelling=True)
            self.assertEqual(master.vtype, vtype)
        else:
            raise AssertionError(f"Unsupported task '{task}'")
        # check correctness of targets
        learner = self.DummyLearner(regression=task == 'regression')
        MACS(learner=learner, master=master).fit(x, y, iterations=1)
        self.assertEqual(learner.p.shape, p.shape)
        self.assertEqual(learner.p.dtype, p.dtype)
        msg = f'Loss: {loss}\n\nRef:\n{learner.p}\n\nAct:\n{p}'
        self.assertTrue(np.allclose(learner.p, p, atol=10 ** -self.PLACES), msg=msg)

    def test_regression(self):
        self._test(task='regression', loss='mae', vtype=None, rtype=None)
        self._test(task='regression', loss='mse', vtype=None, rtype=None)

    def test_binary_class(self):
        self._test(task='binary', loss='hd', vtype='binary', rtype='class')
        self._test(task='binary', loss='ce', vtype='continuous', rtype='class')
        self._test(task='binary', loss='mae', vtype='continuous', rtype='class')
        self._test(task='binary', loss='mse', vtype='continuous', rtype='class')

    def test_binary_probability(self):
        self._test(task='binary', loss='hd', vtype='binary', rtype='probability')
        self._test(task='binary', loss='ce', vtype='continuous', rtype='probability')
        self._test(task='binary', loss='mae', vtype='continuous', rtype='probability')
        self._test(task='binary', loss='mse', vtype='continuous', rtype='probability')

    def test_multiclass_class(self):
        self._test(task='multiclass', loss='hd', vtype='binary', rtype='class')
        self._test(task='multiclass', loss='ce', vtype='continuous', rtype='class')
        self._test(task='multiclass', loss='mae', vtype='continuous', rtype='class')
        self._test(task='multiclass', loss='mse', vtype='continuous', rtype='class')

    def test_multiclass_probability(self):
        self._test(task='multiclass', loss='hd', vtype='binary', rtype='probability')
        self._test(task='multiclass', loss='ce', vtype='continuous', rtype='probability')
        self._test(task='multiclass', loss='mae', vtype='continuous', rtype='probability')
        self._test(task='multiclass', loss='mse', vtype='continuous', rtype='probability')

    def test_multilabel_class(self):
        self._test(task='multilabel', loss='hd', vtype='binary', rtype='class')
        self._test(task='multilabel', loss='ce', vtype='continuous', rtype='class')
        self._test(task='multilabel', loss='mae', vtype='continuous', rtype='class')
        self._test(task='multilabel', loss='mse', vtype='continuous', rtype='class')

    def test_multilabel_probability(self):
        self._test(task='multilabel', loss='hd', vtype='binary', rtype='probability')
        self._test(task='multilabel', loss='ce', vtype='continuous', rtype='probability')
        self._test(task='multilabel', loss='mae', vtype='continuous', rtype='probability')
        self._test(task='multilabel', loss='mse', vtype='continuous', rtype='probability')
