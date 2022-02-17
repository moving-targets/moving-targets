from typing import Optional

import numpy as np

from moving_targets.masters import losses
from moving_targets.masters.backends import GurobiBackend
from moving_targets.util import probabilities
from test.abstract import AbstractTest


# TENSORFLOW IS IMPORTED LAZILY TO AVOID CONFLICTS WITH DEPENDENCIES TESTS

class TestLosses(AbstractTest):
    @staticmethod
    def _normalize(array: np.ndarray) -> np.ndarray:
        """Normalizes the values of an array so that each of its rows sum up to one."""
        return array if array.ndim == 1 else np.transpose(array.transpose() / array.sum(axis=1))

    @staticmethod
    def _ref_value(loss: str,
                   alpha: float,
                   values: np.ndarray,
                   targets: np.ndarray,
                   predictions: np.ndarray,
                   weights: Optional[np.ndarray] = None) -> float:
        """Implements the correct loss definition based on the derivative of the given reference loss."""
        import tensorflow as tf
        import tensorflow.keras.losses as ls
        # handle inputs
        v = tf.constant(values.reshape((len(values), -1)), dtype=tf.float32)
        t = tf.constant(targets.reshape((len(targets), -1)), dtype=tf.float32)
        p = tf.Variable(predictions.reshape((len(predictions), -1)), dtype=tf.float32)
        w = None if weights is None else tf.constant(len(weights) * weights / weights.sum(), dtype=tf.float32)
        # handle loss
        if loss == 'mae':
            loss = ls.MeanAbsoluteError(reduction='none')
        elif loss == 'mse':
            loss = ls.MeanSquaredError(reduction='none')
        elif loss == 'bce':
            loss = ls.BinaryCrossentropy(reduction='none')
        elif loss == 'cce':
            loss = ls.CategoricalCrossentropy(reduction='none')
        elif loss in ['bh', 'ch']:
            def _hamming_distance(y_true, y_pred, sample_weight):
                partial_losses = tf.reduce_mean(y_true * (1 - y_pred) + (1 - y_true) * y_pred, axis=1)
                return partial_losses if sample_weight is None else sample_weight * partial_losses

            loss = _hamming_distance
        else:
            raise AssertionError(f"Unknown loss {loss}")
        # handle differentiation
        with tf.GradientTape() as tape:
            loss = loss(t, p, sample_weight=w)
            nabla = tape.gradient(loss, p)
        nabla_term = tf.reduce_sum(nabla * (v - p), axis=1)
        squared_term = ls.MeanSquaredError(reduction='none').__call__(v, p, sample_weight=w)
        return tf.reduce_mean(alpha * nabla_term + squared_term).numpy()

        # nabla_term = []
        # for v, t, p in zip(values, targets, predictions):
        #     nabla = derivative(func=lambda x: loss(np.reshape(t, (1, -1)), np.reshape(x, (1, -1))), x0=p - t, dx=1e-3)
        #     nabla_term.append(np.dot(nabla, v - p))
        # nabla_term = np.array(nabla_term) if sample_weight is None else np.multiply(sample_weight, nabla_term)
        # squared_term = mean_squared_error(values, predictions, sample_weight=sample_weight)
        # losses = alpha * nabla_term + squared_term
        # return losses.mean()

    def _test(self, loss: str, task: str, classes: Optional[int], weights: bool, **loss_args):
        """Checks that the given moving targets loss behaves as the reference one with respect to the given task (which
        is used to account for the postprocessing needed for the predictions), the given number of classes (in case of
        classification tasks), and the given vector of sample weights."""
        try:
            np.random.seed(self.SEED)
            backend = GurobiBackend()
            size = (self.NUM_SAMPLES,) if classes is None or classes == 2 else (self.NUM_SAMPLES, classes)
            kind = 'binary' if task in ['indicator', 'probability'] else 'continuous'
            for i in range(self.NUM_TESTS):
                backend.build()
                # assign alpha, predictions, and sample weights
                alpha = np.random.uniform(1, 0)
                predictions = self._normalize(np.random.uniform(0, 1, size=size))
                sample_weight = np.random.uniform(size=self.NUM_SAMPLES) if weights else None
                if kind == 'binary':
                    # for binary reference values and targets, create a vector of classes and then binarize them
                    values = probabilities.get_onehot(vector=np.random.choice(range(classes), size=self.NUM_SAMPLES))
                    targets = probabilities.get_onehot(vector=np.random.choice(range(classes), size=self.NUM_SAMPLES))
                else:
                    # for continuous reference values and targets, create a vector of outputs and then normalize
                    values = self._normalize(np.random.uniform(0, 1, size=size))
                    targets = self._normalize(np.random.uniform(0, 1, size=size))
                # create constant model variables from values then compute the backend objective
                variables = backend.add_constants(values, vtype=kind, name='var')
                mt_loss = losses.aliases[loss](**loss_args).__call__(
                    backend=backend,
                    alpha=alpha,
                    variables=variables,
                    targets=targets,
                    predictions=predictions,
                    sample_weight=sample_weight
                )
                mt_value = backend.minimize(mt_loss).solve().get_objective()
                backend.clear()
                # optionally post-process the values then compute the reference objective
                if task == 'indicator':
                    labelling = loss_args.get('labelling') or False
                    values = probabilities.get_classes(values, labelling=labelling)
                    targets = probabilities.get_classes(targets, labelling=labelling)
                    predictions = probabilities.get_classes(predictions, labelling=labelling)
                    if not labelling:
                        values = probabilities.get_onehot(values, classes=classes)
                        targets = probabilities.get_onehot(targets, classes=classes)
                        predictions = probabilities.get_onehot(predictions, classes=classes)
                ref_value = self._ref_value(
                    loss=loss,
                    alpha=alpha,
                    values=values,
                    targets=targets,
                    predictions=predictions,
                    weights=sample_weight
                )
                # compare moving targets objective with the reference one
                self.assertAlmostEqual(mt_value, ref_value, places=self.PLACES, msg=f'Error at iteration {i}')
        except NotImplementedError:
            # the abstract testcase will be executed as well, thus if we check whether we are running one of its tests
            self.assertTrue(self.__class__.__name__ == 'TestBackend')

    def test_mae(self):
        self._test(loss='mae', task='regression', classes=None, weights=False)

    def test_mae_weights(self):
        self._test(loss='mae', task='regression', classes=None, weights=True)

    def test_binary_mae(self):
        self._test(loss='mae', task='probability', classes=self.NUM_CLASSES, weights=False, binary=True)

    def test_binary_mae_weights(self):
        self._test(loss='mae', task='probability', classes=self.NUM_CLASSES, weights=True, binary=True)

    def test_mse(self):
        self._test(loss='mse', task='regression', classes=None, weights=False)

    def test_mse_weights(self):
        self._test(loss='mse', task='regression', classes=None, weights=True)

    def test_binary_mse(self):
        self._test(loss='mse', task='probability', classes=self.NUM_CLASSES, weights=False, binary=True)

    def test_binary_mse_weights(self):
        self._test(loss='mse', task='probability', classes=self.NUM_CLASSES, weights=True, binary=True)

    def test_bh(self):
        self._test(loss='bh', task='indicator', classes=2, weights=False)

    def test_bh_weights(self):
        self._test(loss='bh', task='indicator', classes=2, weights=True)

    def test_labelling_bh(self):
        self._test(loss='bh', task='indicator', classes=2, weights=False, labelling=True)

    def test_labelling_bh_weights(self):
        self._test(loss='bh', task='indicator', classes=2, weights=True, labelling=True)

    def test_ch(self):
        self._test(loss='ch', task='indicator', classes=self.NUM_CLASSES, weights=False)

    def test_ch_weights(self):
        self._test(loss='ch', task='indicator', classes=self.NUM_CLASSES, weights=True)

    def test_labelling_ch(self):
        self._test(loss='ch', task='indicator', classes=self.NUM_CLASSES, weights=False, labelling=True)

    def test_labelling_ch_weights(self):
        self._test(loss='ch', task='indicator', classes=self.NUM_CLASSES, weights=True, labelling=True)

    def test_bce(self):
        self._test(loss='bce', task='probability', classes=2, weights=False)

    def test_bce_weights(self):
        self._test(loss='bce', task='probability', classes=2, weights=True)

    def test_binary_bce(self):
        self._test(loss='bce', task='probability', classes=2, weights=False, binary=True)

    def test_binary_bce_weights(self):
        self._test(loss='bce', task='probability', classes=2, weights=True, binary=True)

    def test_cce(self):
        self._test(loss='cce', task='probability', classes=self.NUM_CLASSES, weights=False)

    def test_cce_weights(self):
        self._test(loss='cce', task='probability', classes=self.NUM_CLASSES, weights=True)

    def test_binary_cce(self):
        self._test(loss='cce', task='probability', classes=self.NUM_CLASSES, weights=False, binary=True)

    def test_binary_cce_weights(self):
        self._test(loss='cce', task='probability', classes=self.NUM_CLASSES, weights=True, binary=True)
