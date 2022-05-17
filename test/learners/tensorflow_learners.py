import random

import numpy as np

from moving_targets.learners import TensorflowMLP
from test.learners.abstract import TestLearners


# TENSORFLOW IS IMPORTED LAZILY TO AVOID CONFLICTS WITH DEPENDENCIES TESTS

class TestTensorflowLearners(TestLearners):
    _EPOCHS: int = 2
    _UNITS: int = 64
    _BATCH_SIZE: int = 4
    _OPTIMIZER: str = 'rmsprop'
    _SHUFFLE: bool = True

    @staticmethod
    def _random_state():
        import tensorflow as tf
        random.seed(0)
        np.random.seed(0)
        tf.random.set_seed(0)

    def _reference(self, learner, x, y, sample_weight) -> np.ndarray:
        learner.fit(x=x,
                    y=y,
                    sample_weight=sample_weight,
                    epochs=self._EPOCHS,
                    batch_size=self._BATCH_SIZE,
                    shuffle=self._SHUFFLE,
                    verbose=False)
        return learner.predict(x).squeeze()

    def test_regression_mlp(self):
        from tensorflow.python.keras.models import Sequential
        from tensorflow.python.keras.layers import Dense

        def ref():
            model = Sequential([Dense(self._UNITS, activation='relu'), Dense(1, activation=None)])
            model.compile(optimizer=self._OPTIMIZER, loss='mse')
            return model

        self._test(
            mt_learner=lambda: TensorflowMLP(
                loss='mse',
                optimizer=self._OPTIMIZER,
                hidden_units=[self._UNITS],
                epochs=self._EPOCHS,
                batch_size=self._BATCH_SIZE,
                shuffle=self._SHUFFLE,
                verbose=False
            ),
            ref_learner=ref,
            classification=False
        )

    def test_classification_mlp(self):
        from tensorflow.python.keras.layers import Dense
        from tensorflow.python.keras.models import Sequential

        def ref():
            model = Sequential([Dense(self._UNITS, activation='relu'), Dense(1, activation='sigmoid')])
            model.compile(optimizer=self._OPTIMIZER, loss='binary_crossentropy')
            return model

        self._test(
            mt_learner=lambda: TensorflowMLP(
                loss='binary_crossentropy',
                output_activation='sigmoid',
                optimizer=self._OPTIMIZER,
                hidden_units=[self._UNITS],
                epochs=self._EPOCHS,
                batch_size=self._BATCH_SIZE,
                shuffle=self._SHUFFLE,
                verbose=False
            ),
            ref_learner=ref,
            classification=True
        )
