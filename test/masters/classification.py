from typing import Union

import numpy as np

from moving_targets import MACS
from moving_targets.learners import Learner
from moving_targets.masters import ClassificationMaster
from moving_targets.util import probabilities
from test.abstract import AbstractTest


class TestClassificationMaster(AbstractTest):
    class DummyLearner(Learner):
        def __init__(self):
            super(TestClassificationMaster.DummyLearner, self).__init__(stats=False)
            self.p = None

        def fit(self, x, y):
            self.p = y

        def predict(self, x):
            return self.p

    def _test(self, loss: str, vtype: str, rtype: str, task: Union[int, str]):
        np.random.seed(0)
        x = np.random.random((self.NUM_SAMPLES, self.NUM_FEATURES))
        # handle outputs
        if task == 2:
            y = np.random.random(self.NUM_SAMPLES).round().astype(int)
        elif isinstance(task, int):
            y = np.random.random((self.NUM_SAMPLES, task)).argmax(axis=1)
        else:
            y = np.random.random((self.NUM_SAMPLES, self.NUM_CLASSES)).round().astype(int)
        # handle expected predictions
        if rtype == 'class':
            p = y
        elif isinstance(task, int):
            p = probabilities.get_onehot(y).astype(float)
        else:
            p = y.astype(float)
        # create macs
        learner = self.DummyLearner()
        master = ClassificationMaster(backend='gurobi', p_loss=loss, task='auto', vtype='auto', rtype=rtype)
        MACS(learner=learner, master=master).fit(x, y, iterations=1)
        # check master parameters
        self.assertEqual(master.task, task)
        self.assertEqual(master.vtype, vtype)
        self.assertEqual(master.rtype, rtype)
        # check adjusted targets shape and type
        self.assertEqual(learner.p.shape, p.shape)
        self.assertEqual(learner.p.dtype, p.dtype)
        self.assertTrue(np.allclose(learner.p, p))

    def test_binary_discrete_class(self):
        self._test(loss='bce', vtype='binary', rtype='class', task=2)

    def test_binary_discrete_probability(self):
        self._test(loss='bce', vtype='binary', rtype='probability', task=2)

    def test_binary_continuous_class(self):
        self._test(loss='rbce', vtype='continuous', rtype='class', task=2)

    def test_binary_continuous_probability(self):
        self._test(loss='rbce', vtype='continuous', rtype='probability', task=2)

    def test_multiclass_discrete_class(self):
        self._test(loss='bce', vtype='binary', rtype='class', task=self.NUM_CLASSES)

    def test_multiclass_discrete_probability(self):
        self._test(loss='bce', vtype='binary', rtype='probability', task=self.NUM_CLASSES)

    def test_multiclass_continuous_class(self):
        self._test(loss='rbce', vtype='continuous', rtype='class', task=self.NUM_CLASSES)

    def test_multiclass_continuous_probability(self):
        self._test(loss='rbce', vtype='continuous', rtype='probability', task=self.NUM_CLASSES)

    def test_multilabel_discrete_class(self):
        self._test(loss='bce', vtype='binary', rtype='class', task='multi')

    def test_multilabel_discrete_probability(self):
        self._test(loss='bce', vtype='binary', rtype='probability', task='multi')

    def test_multilabel_continuous_class(self):
        self._test(loss='rbce', vtype='continuous', rtype='class', task='multi')

    def test_multilabel_continuous_probability(self):
        self._test(loss='rbce', vtype='continuous', rtype='probability', task='multi')
