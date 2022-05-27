"""Moving Targets Losses."""
from typing import Optional, Any, Callable, Tuple

import numpy as np

from moving_targets.masters.backends import Backend
from moving_targets.util import probabilities
from moving_targets.util.errors import not_implemented_message
from moving_targets.util.masking import mask_data, get_mask


class Loss:
    """Basic interface for a Moving Targets Master Loss."""

    def __init__(self, binary: bool, name: str):
        """
        :param binary:
            Whether the model variables are expected to be binary or not.

            Knowing that the model variables are binary allows to deal with error computation in a faster way since it
            will lead to a simplified scenario in which the model variables can only have an integer value in {0, 1},
            while the numeric variables can only have a continuous value in [0, 1].

        :param name:
            The name of the loss.
        """
        # handle squared values computation based on whether or not variables are binary, indeed, given the model
        # variables <v> and the learners predictions <p>, for binary variables we have that:
        #    square(v, p) = [v * (1 - p) + (1 - v) * p]^2 =
        #                 = [v * (1 - p)]^2        + [(1 - v) * p]^2	    - 2 * [v * (1 - p) * (1 - v) * p] =
        #                 = [v^2 * (1 - 2p + p^2)] + [p^2 * (1 - 2v + v^2)] - 2 * [v * (1 - v) * (1 - p) * p] =
        #                 = [v - 2vp + vp^2]       + [p^2 - 2vp^2 + vp^2]   - 2 * [(v - v) * (p - p^2)] = --> as v^2 = v
        #                 = [v - 2vp + vp^2]       + [p^2 - vp^2]           - 2 * [0] =
        #                 =  v - 2vp + p^2
        # thus we can reformulate this as a linear expressions, which can be handled in a faster way
        if binary:
            square = lambda b, v, p: v - 2 * v * p + p ** 2
        else:
            square = lambda b, v, p: b.square(v - p)

        self.binary: bool = binary
        """Whether the model variables are expected to be binary or not."""

        self.square: Callable = square
        """A callable function of type f(<b>, <v>, <p>) -> <(p - v) ** 2> representing the square strategy, where <b>
        is the backend instance, <v> is the array of backend variables, and <p> is the array of predictions."""

        self.__name__: str = name
        """The name of the loss."""

    def __call__(self,
                 backend: Backend,
                 variables: np.ndarray,
                 targets: np.ndarray,
                 predictions: np.ndarray,
                 sample_weight: Optional[np.ndarray] = None,
                 mask: Optional[float] = None) -> Tuple[Any, Any]:
        """Computes the two terms of the loss value, which are obtained by approximating the actual loss <L> to its
        first order taylor expansion L(z, y) in the point <p>, where <z> are the model variables, <y> the original
        targets, and <p> the learners predictions.

        The taylor expansion of L(z, y) at point p is defined as: L(p, y) + dL(p, y) / dp * (z - p) + o((z - p) ^ 2).
        Since the term "L(p, y)" is constant, we can discard it when minimizing <L>, thus the final loss can be seen as
        a balance between the term "dL(p, y) / dp * (z - p)" and the term "(z - p) ^ 2", which is achieved through the
        balancing parameter <alpha>, thus obtaining: min_z {alpha * dL(p, y) / dp * (z - p) + (z - p) ^ 2}.

        In case of multiple features, the term dL(p, y) / dp * (z - p) is replaced by the sum of partial derivatives
        dL(p, y) / dp_i multiplied by the respective component (z_i - p_i), while the (z - p) ^ 2 term will be computed
        as a mean over all the features according to the norm-two definition, leading to a final formulation which is:
        min_z {alpha * (dL(p, y) / dp_1 * (z_1 - p_1) + ... + dL(p, y) / dp_F * (z_F - p_F)) + |z - p|_2^2}, or else:
        min_z {alpha * sum(nabla_L(p, y) * (z - p)) + (z - p) ^ 2}.

        :param backend:
            The `Backend` instance used to compute the loss.

        :param variables:
            The array of backend model variables (z).

        :param targets:
            The array of original targets (y).

        :param predictions:
            The array of learners predictions (p).

        :param sample_weight:
            The (optional) array of sample weights.

        :param mask:
            An (optional) masking value used to mask the original targets.

        :return:
            A tuple containing the nabla term nabla_L(p, y) @ (z - p)^T and the squared term (z - p) @ (z - p)^T.
        """
        mask = get_mask(targets, mask)
        # reshape inputs in order to create a NxF matrix, where N is the number of samples and F the number of features
        variables = np.reshape(variables, (len(variables), -1))
        targets = np.reshape(targets, (len(targets), -1))
        predictions = np.reshape(predictions, (len(predictions), -1))
        # flatten sample weights (or create a vector of ones is none are passed), then normalize them
        sample_weight = np.ones(len(variables)) if sample_weight is None else sample_weight.flatten()
        sample_weight = len(sample_weight) * np.array(sample_weight) / np.sum(sample_weight)
        # compute the nabla term, which is a summation of the terms dL / dp_i * (z_i - p_i), with i in {1, ..., F}
        # also mask each vector using the previously computed mask as reference
        m_targ, m_pred, m_var, m_weights = mask_data(targets, predictions, variables, sample_weight, mask=mask)
        nabla_term = self.nabla(targets=m_targ, predictions=m_pred)
        nabla_term = backend.sum(nabla_term * (m_var - m_pred), axis=1)
        nabla_term = backend.mean(m_weights * nabla_term)
        # compute the squared term, which is the mean of the terms (z_i - p_i) ^ 2, with i in {1, ..., F}
        squared_term = self.square(b=backend, v=variables, p=predictions)
        squared_term = backend.mean(squared_term, axis=1)
        squared_term = backend.mean(sample_weight * squared_term)
        # finally return the two losses
        return nabla_term, squared_term

    def nabla(self, targets: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        """Computes the nabla_L(p, y) values, with <p> being the predictions and <y> being the original targets.

        :param targets:
            The array of original targets (y).

        :param predictions:
            The array of learners predictions (p).

        :return:
            The array of values of nabla_L(p, y) for each of the samples.
        """
        raise NotImplementedError(not_implemented_message(name='nabla'))


class MAE(Loss):
    """Mean Absolute Error Loss."""

    def __init__(self, binary: bool = False, name: str = 'mean_absolute_error'):
        """
        :param binary:
            Whether the model variables are expected to be binary or not.

        :param name:
            The name of the loss.
        """
        super(MAE, self).__init__(binary=binary, name=name)

    def nabla(self, targets: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        # given a target vector with <F> features, the loss is: L(p, t) = sum(|p - t|) / F
        # thus, its partial derivative will be: dL / dp_i = sign(p - t) / F
        return np.sign(predictions - targets) / targets.shape[1]


class MSE(Loss):
    """Mean Squared Error Loss."""

    def __init__(self, binary: bool = False, name: str = 'mean_squared_error'):
        """
        :param binary:
            Whether the model variables are expected to be binary or not.

            Knowing that the model variables are binary allows to deal with error computation in a faster way since it
            will lead to a simplified scenario in which the model variables can only have an integer value in {0, 1},
            while the numeric variables can only have a continuous value in [0, 1].

        :param name:
            The name of the loss.
        """
        super(MSE, self).__init__(binary=binary, name=name)

    def nabla(self, targets: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        # given a target vector with <F> features, the loss is: L(p, t) = sum((p - t) ^ 2) / F
        # thus, its partial derivative will be: dL / dp_i = 2 * (p - t) / F
        return 2 * (predictions - targets) / targets.shape[1]


class HammingDistance(Loss):
    """Hamming Distance."""

    def __init__(self, labelling: bool = False, name: str = 'hamming_distance'):
        """
        :param labelling:
            Whether this is a loss for a labelling or a classification task.

            This parameter is used to retrieve classes/labels from the vector of predicted probabilities in multiclass
            or multilabel scenarios. For binary tasks this has no effect.

        :param name:
            Whether the master represents a labelling or a classification task.
        """
        super(HammingDistance, self).__init__(binary=True, name=name)

        self.labelling: bool = labelling
        """Whether this is a loss for a labelling or a classification task."""

    def nabla(self, targets: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        # given a target vector with <F> features, the loss is: L(p, t) = sum((1 - t) * p + t * (1 - p)) / F
        # which can be eventually rewritten as: L(p, t) = sum(p - 2 * t * p + t) / F
        # thus, its partial derivative will be: dL / dp_i = (1 - 2 * t_i) / F
        return (1 - 2 * targets) / targets.shape[1]

    def __call__(self,
                 backend: Backend,
                 variables: np.ndarray,
                 targets: np.ndarray,
                 predictions: np.ndarray,
                 sample_weight: Optional[np.ndarray] = None,
                 mask: Optional[float] = None) -> Tuple[Any, Any]:
        assert np.all(targets == targets.astype(int)), f"HammingDistance can handle integer targets only, got {targets}"
        # since the hamming distance works on discrete targets only, we retrieve the class values from the predictions
        if self.labelling:
            # in case of labelling tasks, this is enough to get a bi-dimensional matrix which matches the targets shape
            predictions = probabilities.get_classes(prob=predictions, labelling=True)
        else:
            # otherwise, if we have a multiclass task we need to onehot encode the probabilities to match the shape
            classes = 2 if targets.squeeze().ndim == 1 else targets.shape[1]
            predictions = probabilities.get_classes(prob=predictions, labelling=False)
            predictions = probabilities.get_onehot(vector=predictions, classes=classes)
        return super(HammingDistance, self).__call__(backend, variables, targets, predictions, sample_weight, mask)


class CrossEntropy(Loss):
    """Negative Log-Likelihood Loss."""

    def __init__(self, binary: bool = True, clip_value: float = 1e-3, name: str = 'crossentropy'):
        """
        :param binary:
            Whether the model variables are expected to be binary or not.

        :param clip_value:
            The clipping value to be used to avoid numerical errors.

        :param name:
            The name of the metric.
        """
        super(CrossEntropy, self).__init__(binary=binary, name=name)

        self.clip_value: float = clip_value
        """The clipping value to be used to avoid numerical errors."""

    def nabla(self, targets: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        # given a target vector with <F> features, the loss is: L(p, t) = sum(-t * ln(p))
        # thus, its partial derivative will be: dL / dp_i = -t / p
        return -targets / predictions.clip(min=self.clip_value, max=1 - self.clip_value)

    def __call__(self,
                 backend: Backend,
                 variables: np.ndarray,
                 targets: np.ndarray,
                 predictions: np.ndarray,
                 sample_weight: Optional[np.ndarray] = None,
                 mask: Optional[float] = None) -> Tuple[Any, Any]:
        # in order to deal with both binary and categorical crossentropy, if the prediction array has either size (N,)
        # or (N, 1), we convert it (along with the targets and variables vectors) to the form [1 - a, a]
        if predictions.squeeze().ndim == 1:
            variables = np.concatenate([1 - variables.reshape((-1, 1)), variables.reshape((-1, 1))], axis=1)
            targets = np.concatenate([1 - targets.reshape((-1, 1)), targets.reshape((-1, 1))], axis=1)
            predictions = np.concatenate([1 - predictions.reshape((-1, 1)), predictions.reshape((-1, 1))], axis=1)
        return super(CrossEntropy, self).__call__(backend, variables, targets, predictions, sample_weight, mask)


aliases: dict = {
    # Mean Absolute Error
    'mae': MAE,
    'mean_absolute_error': MAE,
    'mean absolute error': MAE,
    # Mean Squared Error
    'mse': MSE,
    'mean_squared_error': MSE,
    'mean squared error': MSE,
    # Hamming Distance
    'hd': HammingDistance,
    'hamming_distance': HammingDistance,
    'hamming distance': HammingDistance,
    'bhd': HammingDistance,
    'binary_hamming': HammingDistance,
    'binary hamming': HammingDistance,
    'chd': HammingDistance,
    'categorical_hamming': HammingDistance,
    'categorical hamming': HammingDistance,
    # CrossEntropy
    'ce': CrossEntropy,
    'crossentropy': CrossEntropy,
    'bce': CrossEntropy,
    'binary_crossentropy': CrossEntropy,
    'binary crossentropy': CrossEntropy,
    'cce': CrossEntropy,
    'categorical_crossentropy': CrossEntropy,
    'categorical crossentropy': CrossEntropy,
    'll': CrossEntropy,
    'log_likelihood': CrossEntropy,
    'log likelihood': CrossEntropy,
    'nll': CrossEntropy,
    'negative_log_likelihood': CrossEntropy,
    'negative log likelihood': CrossEntropy,
}
"""Dictionary which associates to each loss alias the respective class type."""
