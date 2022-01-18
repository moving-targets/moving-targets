from typing import Union

import numpy as np

from moving_targets.masters.backends import NumpyBackend
from moving_targets.masters.optimizers.optimizer import Optimizer
from moving_targets.util import probabilities
from moving_targets.util.errors import not_implemented_message


class BetaSatisfiability(Optimizer):
    """Dynamic Optimizer which should be used to control the behaviour of beta when dealing with satisfiability.

    Indeed, when using the beta step, it may be necessary to adopt an adaptive strategy in order to avoid the MACS
    process to stop due to the constraint 'p_loss <= beta' leading to infeasibility, since there are some cases in
    which the constraint satisfiability does not imply a null p_loss due to, e.g., (1) the usage of class probabilities
    instead of class targets in classification problems, in fact, since most of the classification losses (apart from
    HammingDistance) use class probabilities while the satisfiability is usually computed on class targets instead,
    there will always be a minimal amount of error due to the loss discrepancy between binary and continuous variables,
    or (2) the presence of some lower/upper bounds in regression problems that are not considered in the constraint
    satisfiability definition but are required by the model variables definition, which may happen for example if we
    know that our targets must be non-negative (thus we force our lower bound to be 0) but the learner might return
    some negative predictions that will introduce an error in the p_loss.

    Still, this minimal error can be measured as the loss between the predictions and their "model" representation,
    which involves variable bounds in the case of regression and the handling of targets and probabilities in the case
    of classification. This latter part is delegated to this method, which must return the accordingly constrained
    variables in order to let the master compute this error relying on its inner loss.
    """

    def __init__(self, base: Union[float, Optimizer]):
        """
        :param base:
            Either a fixed floating point value representing the initial value for the hyper-parameter to optimize, or
            a wrapped custom optimizer which returns a dynamic value to reduce by the given factor after each iteration.

        """
        super(BetaSatisfiability, self).__init__(base=base)

    def _expected_variables(self, macs, x, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Computes the expected "model" representation of the predictions.

        :param macs:
            The `MACS` instance encapsulating the master.

        :param x:
            The training samples.

        :param y:
            The training labels.

        :param p:
            The `Learner` predictions.

        :return:
            An array representing the expected "model" representation of the predictions.
        """
        raise NotImplementedError(not_implemented_message(name='_expected_variables'))

    def __call__(self, macs, x, y: np.ndarray, p: np.ndarray) -> float:
        value = super(BetaSatisfiability, self).__call__(macs, x, y, p)
        if p is not None:
            # in order to implement the adaptive strategy used to avoid model infeasibility due to the beta constraint
            # we rely on the NumpyBackend instance which can compute the value of the given loss between two numpy
            # arrays; this value will be our minimal error to be added to beta
            v = self._expected_variables(macs, x, y, p)
            value += macs.master.p_loss(backend=NumpyBackend(), numeric_variables=p, model_variables=v)
        return value


class BetaClassSatisfiability(BetaSatisfiability):
    """Beta Satisfiability Optimizer which adjusts the hyper-parameter value by computing the minimal error amount
    between the predicted probabilities and their class representation.

    This may be useful when dealing with a classification loss that uses predicted probabilities while checking for the
    model satisfiability on the predicted classes, since in that case there will be an amount of error in the p_loss
    that is not dependent from the constraint satisfaction.
    """

    def __init__(self, base: Union[float, Optimizer], multi_label: bool = False):
        """
        :param base:
            Either a fixed floating point value representing the initial value for the hyper-parameter to optimize, or
            a wrapped custom optimizer which returns a dynamic value to reduce by the given factor after each iteration.

        :param multi_label:
            Whether the classification task must handle multiple labels or not.
        """
        super(BetaClassSatisfiability, self).__init__(base=base)

        self.multi_label: bool = multi_label
        """Whether the classification task must handle multiple labels or not."""

    def _expected_variables(self, macs, x, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        # the classes are obtained using the 'get_classes' method
        c = probabilities.get_classes(prob=p, multi_label=self.multi_label)
        # if the task is not multi-label, the classes will be one-dimensional, thus we onehot encode for compatibility
        return c if self.multi_label else probabilities.get_onehot(c, classes=len(np.unique(y)), onehot_binary=False)


class BetaBoundedSatisfiability(BetaSatisfiability):
    """Beta Satisfiability Optimizer which adjusts the hyper-parameter value by computing the minimal error amount
    between the model predictions and their constrained representation which must satisfy the lower and upper bounds.

    This may be useful when dealing with regression losses that is unconstrained (due to the employed machine learning
    model) while putting some lower/upper bounds constraints in the solver, since in that case there will be an amount
    of error in the p_loss that is not dependent from the constraint satisfaction.
    """

    def __init__(self, base: Union[float, Optimizer], lb: float, ub: float):
        """
        :param base:
            Either a fixed floating point value representing the initial value for the hyper-parameter to optimize, or
            a wrapped custom optimizer which returns a dynamic value to reduce by the given factor after each iteration.

        :param lb:
            The model variables lower bounds.

        :param ub:
            The model variables upper bounds.
        """
        super(BetaBoundedSatisfiability, self).__init__(base=base)

        self.lb: float = lb
        """The model variables lower bounds."""

        self.ub: float = ub
        """The model variables upper bounds."""

    def _expected_variables(self, macs, x, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        # the constrained representation is obtained by projecting the predictions in the feasible space via clipping
        return p.clip(min=self.lb, max=self.ub)
