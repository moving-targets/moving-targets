from typing import Callable, Optional

import numpy as np
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae, log_loss as ce, precision_score as ps

from moving_targets.masters.backends import GurobiBackend
from moving_targets.masters.losses import Loss, SAE, SSE, MAE, MSE, HammingDistance, CrossEntropy, \
    ReversedCrossEntropy, SymmetricCrossEntropy
from moving_targets.util import probabilities
from test.abstract import AbstractTest


# TENSORFLOW IS IMPORTED LAZILY TO AVOID CONFLICTS WITH DEPENDENCIES TESTS

class TestLosses(AbstractTest):

    @staticmethod
    def _ce_handler(y: np.ndarray, p: np.ndarray, sw: Optional[np.ndarray], bnr: bool, sym: bool) -> np.ndarray:
        """Computes the categorical crossentropy using tensorflow primitives."""
        import tensorflow as tf
        import tensorflow.keras.backend as k

        k.set_epsilon(1e-15)
        y = tf.cast(y, tf.float32)
        p = tf.cast(p, tf.float32)
        sw = tf.cast(tf.constant(1.0) if sw is None else len(sw) * tf.constant(sw) / k.sum(sw), tf.float32)
        fn = k.binary_crossentropy if bnr else k.categorical_crossentropy
        ls = fn(p, y) + (fn(y, p) if sym else 0)
        return k.mean(sw * ls).numpy()

    def _test(self, mt_loss: Loss, ref_loss: Callable, task: str, classes: Optional[int], weights: bool):
        """The core class, which must be implemented by the solver so to compare the obtained loss and the ground loss.

        :param mt_loss:
            The solver loss.

        :param ref_loss:
            The ground truth loss (obtained from scikit learn and custom losses).

        :param task:
            The kind of task, which indicates for the postprocessing needed for the predictions.

            It can be one in:
            - regression (i.e., discrete values);
            - indicator (i.e., discrete values);
            - probabilities (i.e., continuous values which sum up to 1);
            - reversed (i.e., probabilities used to compute reserved crossentropy);
            - symmetric (i.e., probabilities used to compute symmetric crossentropy).

        :param classes:
            Either None in case of regression losses or the number of classes.

        :param weights:
            Whether or not to use sample weights.
        """
        try:
            np.random.seed(self.SEED)
            backend = GurobiBackend()
            size = (self.NUM_SAMPLES,) if classes is None or classes == 2 else (self.NUM_SAMPLES, classes)
            kind = 'binary' if task in ['indicator', 'probability'] else 'continuous'
            for i in range(self.NUM_TESTS):
                backend.build()
                # assign sample weights, numeric variables (ground truths)
                sample_weight = np.random.uniform(size=self.NUM_SAMPLES) if weights else None
                numeric_values = np.random.uniform(0.001, 0.999, size=size).transpose()
                numeric_values = numeric_values / (1 if classes is None or classes == 2 else numeric_values.sum(axis=0))
                numeric_values = numeric_values.transpose()
                if kind == 'binary':
                    # for the binary variables create a vector of classes and then binarize them
                    vector = np.random.choice(range(classes), size=self.NUM_SAMPLES)
                    model_values = probabilities.get_onehot(vector=vector)
                else:
                    # for the continuous variables create a vector of outputs and then normalize if multiclass
                    model_values = np.random.uniform(0, 1, size=size).transpose()
                    model_values = model_values / (1.0 if classes is None or classes == 2 else model_values.sum(axis=0))
                    model_values = model_values.transpose()
                # create constant model variables from values
                model_variables = backend.add_constants(model_values, vtype=kind, name='var')
                # compute backend objective
                mt_value = mt_loss(backend, numeric_values, model_variables, sample_weight)
                mt_value = backend.minimize(mt_value).solve().get_objective()
                backend.clear()
                # compute reference objective
                if task == 'indicator':
                    model_values = probabilities.get_classes(model_values)
                    numeric_values = probabilities.get_classes(numeric_values)
                ref_value = ref_loss(model_values, numeric_values, sample_weight=sample_weight)
                # compare moving targets objective with the reference one
                self.assertAlmostEqual(mt_value, ref_value, places=self.PLACES)
        except NotImplementedError:
            # the abstract testcase will be executed as well, thus if we check whether we are running one of its tests
            self.assertTrue(self.__class__.__name__ == 'TestBackend')

    def test_sae(self):
        self._test(mt_loss=SAE(),
                   ref_loss=lambda y, p, sample_weight: self.NUM_SAMPLES * mae(y, p, sample_weight=sample_weight),
                   task='regression',
                   classes=None,
                   weights=False)

    def test_sae_weights(self):
        self._test(mt_loss=SAE(),
                   ref_loss=lambda y, p, sample_weight: self.NUM_SAMPLES * mae(y, p, sample_weight=sample_weight),
                   task='regression',
                   classes=None,
                   weights=True)

    def test_sse(self):
        self._test(mt_loss=SSE(),
                   ref_loss=lambda y, p, sample_weight: self.NUM_SAMPLES * mse(y, p, sample_weight=sample_weight),
                   task='regression',
                   classes=None,
                   weights=False)

    def test_sse_weights(self):
        self._test(mt_loss=SSE(),
                   ref_loss=lambda y, p, sample_weight: self.NUM_SAMPLES * mse(y, p, sample_weight=sample_weight),
                   task='regression',
                   classes=None,
                   weights=True)

    def test_mae(self):
        self._test(mt_loss=MAE(),
                   ref_loss=mae,
                   task='regression',
                   classes=None,
                   weights=False)

    def test_mae_weights(self):
        self._test(mt_loss=MAE(),
                   ref_loss=mae,
                   task='regression',
                   classes=None,
                   weights=True)

    def test_mse(self):
        self._test(mt_loss=MSE(),
                   ref_loss=mse,
                   task='regression',
                   classes=None,
                   weights=False)

    def test_mse_weights(self):
        self._test(mt_loss=MSE(),
                   ref_loss=mse,
                   task='regression',
                   classes=None,
                   weights=True)

    def test_binary_sae(self):
        self._test(mt_loss=SAE(binary=True, sum_features=False),
                   ref_loss=lambda y, p, sample_weight: self.NUM_SAMPLES * mae(y, p, sample_weight=sample_weight),
                   task='probability',
                   classes=self.NUM_CLASSES,
                   weights=False)

    def test_binary_sae_weights(self):
        self._test(mt_loss=SAE(binary=True, sum_features=False),
                   ref_loss=lambda y, p, sample_weight: self.NUM_SAMPLES * mae(y, p, sample_weight=sample_weight),
                   task='probability',
                   classes=self.NUM_CLASSES,
                   weights=True)

    def test_binary_sse(self):
        self._test(mt_loss=SSE(binary=True, sum_features=False),
                   ref_loss=lambda y, p, sample_weight: self.NUM_SAMPLES * mse(y, p, sample_weight=sample_weight),
                   task='probability',
                   classes=self.NUM_CLASSES,
                   weights=False)

    def test_binary_sse_weights(self):
        self._test(mt_loss=SSE(binary=True, sum_features=False),
                   ref_loss=lambda y, p, sample_weight: self.NUM_SAMPLES * mse(y, p, sample_weight=sample_weight),
                   task='probability',
                   classes=self.NUM_CLASSES,
                   weights=True)

    def test_binary_mae(self):
        self._test(mt_loss=MAE(binary=True, sum_features=False),
                   ref_loss=lambda y, p, sample_weight: mae(y, p, sample_weight=sample_weight),
                   task='probability',
                   classes=self.NUM_CLASSES,
                   weights=False)

    def test_binary_mae_weights(self):
        self._test(mt_loss=MAE(binary=True, sum_features=False),
                   ref_loss=lambda y, p, sample_weight: mae(y, p, sample_weight=sample_weight),
                   task='probability',
                   classes=self.NUM_CLASSES,
                   weights=True)

    def test_binary_mse(self):
        self._test(mt_loss=MSE(binary=True, sum_features=False),
                   ref_loss=lambda y, p, sample_weight: mse(y, p, sample_weight=sample_weight),
                   task='probability',
                   classes=self.NUM_CLASSES,
                   weights=False)

    def test_binary_mse_weights(self):
        self._test(mt_loss=MSE(binary=True, sum_features=False),
                   ref_loss=lambda y, p, sample_weight: mse(y, p, sample_weight=sample_weight),
                   task='probability',
                   classes=self.NUM_CLASSES,
                   weights=True)

    def test_bh(self):
        self._test(mt_loss=HammingDistance(),
                   ref_loss=lambda y, p, sample_weight: 1 - ps(y, p, sample_weight=sample_weight, average='micro'),
                   task='indicator',
                   classes=2,
                   weights=False)

    def test_bh_weights(self):
        self._test(mt_loss=HammingDistance(),
                   ref_loss=lambda y, p, sample_weight: 1 - ps(y, p, sample_weight=sample_weight, average='micro'),
                   task='indicator',
                   classes=2,
                   weights=True)

    def test_bce(self):
        self._test(mt_loss=CrossEntropy(),
                   ref_loss=ce,
                   task='probability',
                   classes=2,
                   weights=False)

    def test_bce_weights(self):
        self._test(mt_loss=CrossEntropy(),
                   ref_loss=ce,
                   task='probability',
                   classes=2,
                   weights=True)

    def test_reversed_bce(self):
        self._test(mt_loss=ReversedCrossEntropy(),
                   ref_loss=lambda y, p, sample_weight: self._ce_handler(y, p, sample_weight, bnr=True, sym=False),
                   task='reversed',
                   classes=2,
                   weights=False)

    def test_reversed_bce_weights(self):
        self._test(mt_loss=ReversedCrossEntropy(),
                   ref_loss=lambda y, p, sample_weight: self._ce_handler(y, p, sample_weight, bnr=True, sym=False),
                   task='reversed',
                   classes=2,
                   weights=True)

    def test_symmetric_bce(self):
        self._test(mt_loss=SymmetricCrossEntropy(),
                   ref_loss=lambda y, p, sample_weight: self._ce_handler(y, p, sample_weight, bnr=True, sym=True),
                   task='symmetric',
                   classes=2,
                   weights=False)

    def test_symmetric_bce_weights(self):
        self._test(mt_loss=SymmetricCrossEntropy(),
                   ref_loss=lambda y, p, sample_weight: self._ce_handler(y, p, sample_weight, bnr=True, sym=True),
                   classes=2,
                   task='symmetric',
                   weights=True)

    def test_ch(self):
        self._test(mt_loss=HammingDistance(),
                   ref_loss=lambda y, p, sample_weight: 1 - ps(y, p, sample_weight=sample_weight, average='micro'),
                   task='indicator',
                   classes=self.NUM_CLASSES,
                   weights=False)

    def test_ch_weights(self):
        self._test(mt_loss=HammingDistance(),
                   ref_loss=lambda y, p, sample_weight: 1 - ps(y, p, sample_weight=sample_weight, average='micro'),
                   task='indicator',
                   classes=self.NUM_CLASSES,
                   weights=True)

    def test_cce(self):
        self._test(mt_loss=CrossEntropy(),
                   ref_loss=ce,
                   task='probability',
                   classes=self.NUM_CLASSES,
                   weights=False)

    def test_cce_weights(self):
        self._test(mt_loss=CrossEntropy(),
                   ref_loss=ce,
                   task='probability',
                   classes=self.NUM_CLASSES,
                   weights=True)

    def test_reversed_cce(self):
        self._test(mt_loss=ReversedCrossEntropy(),
                   ref_loss=lambda y, p, sample_weight: self._ce_handler(y, p, sample_weight, bnr=False, sym=False),
                   task='reversed',
                   classes=self.NUM_CLASSES,
                   weights=False)

    def test_reversed_cce_weights(self):
        self._test(mt_loss=ReversedCrossEntropy(),
                   ref_loss=lambda y, p, sample_weight: self._ce_handler(y, p, sample_weight, bnr=False, sym=False),
                   task='reversed',
                   classes=self.NUM_CLASSES,
                   weights=True)

    def test_symmetric_cce(self):
        self._test(mt_loss=SymmetricCrossEntropy(),
                   ref_loss=lambda y, p, sample_weight: self._ce_handler(y, p, sample_weight, bnr=False, sym=True),
                   task='symmetric',
                   classes=self.NUM_CLASSES,
                   weights=False)

    def test_symmetric_cce_weights(self):
        self._test(mt_loss=SymmetricCrossEntropy(),
                   ref_loss=lambda y, p, sample_weight: self._ce_handler(y, p, sample_weight, bnr=False, sym=True),
                   task='symmetric',
                   classes=self.NUM_CLASSES,
                   weights=True)
