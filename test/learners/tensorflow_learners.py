import random
from typing import Any

import numpy as np

from moving_targets.learners import MultiLayerPerceptron
from test.learners.abstract import TestLearners


# TENSORFLOW IS IMPORTED LAZILY TO AVOID CONFLICTS WITH DEPENDENCIES TESTS

class TestTensorflowLearners(TestLearners):
    @staticmethod
    def _random_state():
        import tensorflow as tf
        random.seed(0)
        np.random.seed(0)
        tf.random.set_seed(0)

    def _reference(self, learner: Any, x: Any, y, sample_weight) -> np.ndarray:
        learner.fit(x, y, sample_weight=sample_weight, epochs=2, batch_size=4, verbose=False)
        return learner.predict(x).squeeze()

    def test_regression_mlp(self):
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense

        mt = MultiLayerPerceptron(loss='mse', optimizer='sgd', epochs=2, batch_size=4, verbose=False)
        ref = Sequential([Dense(128, activation='relu'), Dense(1, activation=None)])
        ref.compile(optimizer='sgd', loss='mse')
        self._test(mt_learner=mt, ref_learner=ref, classification=False, random_state=self._random_state)

    def test_classification_mlp(self):
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.models import Sequential

        mt = MultiLayerPerceptron(loss='binary_crossentropy', output_activation='sigmoid', optimizer='sgd',
                                  epochs=2, batch_size=4, verbose=False)
        ref = Sequential([Dense(128, activation='relu'), Dense(1, activation='sigmoid')])
        ref.compile(optimizer='sgd', loss='binary_crossentropy')
        self._test(mt_learner=mt, ref_learner=ref, classification=True, random_state=self._random_state)
