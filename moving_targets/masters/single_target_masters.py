"""Abstract Default Masters."""
from typing import Any, Optional, Callable, Union, List

import numpy as np

from moving_targets.masters.backends import Backend, get_backend as get_bk, NumpyBackend
from moving_targets.masters.losses import regression_loss as reg_loss, classification_loss as cls_loss
from moving_targets.masters.losses.loss import Loss
from moving_targets.masters.master import Master
from moving_targets.util import probabilities
from moving_targets.util.errors import not_implemented_message
from moving_targets.util.typing import Number


class SingleTargetMaster(Master):
    """Template class for a Moving Targets Master for Single-Target Tasks."""

    def __init__(self,
                 backend: Backend,
                 satisfied: Callable,
                 vtype: str,
                 lb: Number,
                 ub: Number,
                 alpha: Optional[float],
                 beta: Optional[float],
                 y_loss: Loss,
                 p_loss: Loss,
                 stats: Union[bool, List[str]] = False):
        """
        :param backend:
            The `Backend` instance encapsulating the optimization solver.

        :param satisfied:
            A `Callable` function of type f(<x>, <y>, <p>) -> <bool> that returns True if and only if given the current
            input (<x>), output (<y>), and predictions (<p>), the expected constraints are satisfied.

        :param vtype:
            The model variables vtype.

        :param lb:
            The model variables lower bounds.

        :param ub:
            The model variables upper bounds.

        :param alpha:
            The default alpha value, or None if no alpha step is wanted.

        :param beta:
            The default beta value, or None if no beta step is wanted.

        :param y_loss:
            A `Loss` instance used to compute the y_loss.

        :param p_loss:
            A `Loss` instance used to compute the p_loss.

        :param stats:
            Either a boolean value indicating whether or not to log statistics, or a list of parameters in ['alpha',
            'beta', 'use_beta', 'y_loss', 'p_loss', 'objective'] whose statistics must be logged.
        """
        super(SingleTargetMaster, self).__init__(backend=backend, stats=stats)

        assert alpha is not None or beta is not None, "Either alpha or beta must be not None"

        self.vtype: str = vtype
        """The model variables vtype."""

        self.lb: Number = lb
        """The model variables lower bounds."""

        self.ub: Number = ub
        """The model variables upper bounds."""

        self._y_loss: Loss = y_loss
        """The `Loss` instance used to compute the y_loss."""

        self._p_loss: Loss = p_loss
        """The `Loss` instance used to compute the p_loss."""

        self._alpha: Optional[float] = alpha
        """The default alpha value."""

        self._beta: Optional[float] = beta
        """The default beta value, or None if no beta step is wanted."""

        self._satisfied: Callable = satisfied
        """The `Callable` function that returns True if and only the expected constraints are satisfied."""

    def _beta_error_variables(self, x, y, p, v) -> np.ndarray:
        """When using the beta step, we need to adopt an adaptive strategy in order to avoid the MACS process to stop
        due to the constraint 'p_loss <= beta' leading to infeasibility.

        Indeed, there are some cases in which the constraint satisfiability does not imply a null p_loss due to, e.g.:
          > the adoption of class probabilities instead of class targets in classification problems, in fact, since
            most of the classification losses (apart from HammingDistance) use class probabilities while the
            satisfiability is usually computed on class targets instead, there will always be a minimal amount of error
            due to the loss discrepancy between binary and continuous variables
          > the presence of some lower/upper bounds in regression problems that are not considered in the constraint
            satisfiability definition but are required by the model variables definition, which may happen for example
            if we know that our targets must be non-negative (thus we force our lower bound to be 0) but the learner
            might return some negative predictions that will introduce an error in the p_loss

        However, this minimal error can be measured as the loss between the predictions and their "p_loss"
        representation, which involve variable bounds in the case of regression and the handling of targets and
        probabilities in the case of classification. This latter part is delegated to this method, which must return
        the "fake" model variables in order to let the master compute this error relying on its inner "_p_loss".

        :param x:
            The training samples.

        :param y:
            The training labels.

        :param p:
            The `Learner` predictions.

        :param v:
            The model variables.

        :return:
            The array of "fake" model variables obtained from the predictions.
        """
        raise NotImplementedError(not_implemented_message('_numpy_model_variables'))

    def build(self, x, y, p) -> Any:
        assert y.ndim == 1, "This master works for single-targets tasks only"
        return self.backend.add_variables(len(y), vtype=self.vtype, lb=self.lb, ub=self.ub, name='y')

    def alpha(self, x, y, p, v) -> float:
        return self._alpha or 1.0

    def beta(self, x, y, p, v) -> float:
        beta = self._beta or 1.0
        if p is not None:
            # here we implement the adaptive strategy used to avoid model infeasibility due to the beta constraint: in
            # order to do that, we rely on the NumpyBackend instance which can compute the value of the given loss
            # between two numpy arrays; this value will be our minimal error to be added to beta
            bev = self._beta_error_variables(x, y, p, v)
            beta += self._p_loss(backend=NumpyBackend(), numeric_variables=p, model_variables=bev)
        return beta

    def use_beta(self, x, y, p, v) -> bool:
        # if p is None (i.e., initial_step == 'projection' and iteration == 0), we will not log anything since both the
        # strategies will collapse to the same formulation, i.e., minimizing the y_loss only
        if p is None:
            return True
        elif self._alpha is None:
            use_beta = True
        elif self._beta is None:
            use_beta = False
        else:
            use_beta = self._satisfied(x, y, p)
        return use_beta

    def y_loss(self, x, y, p, v) -> Any:
        return self._y_loss(backend=self.backend, numeric_variables=y, model_variables=v)

    def p_loss(self, x, y, p, v) -> Any:
        return self._p_loss(backend=self.backend, numeric_variables=p, model_variables=v)

    def solution(self, x, y, p, v) -> Any:
        return self.backend.get_values(expressions=v)


class SingleTargetRegression(SingleTargetMaster):
    """A standard Moving Targets Master for Single-Target Regression Tasks with no constraints."""

    def __init__(self,
                 backend: Union[str, Backend],
                 satisfied: Callable,
                 lb: Number = -float('inf'),
                 ub: Number = float('inf'),
                 alpha: Optional[float] = 1.0,
                 beta: Optional[float] = 1.0,
                 y_loss: Union[str, Loss] = 'mse',
                 p_loss: Union[str, Loss] = 'mse',
                 stats: Union[bool, List[str]] = False):
        """
        :param backend:
            The `Backend` instance encapsulating the optimization solver.

        :param satisfied:
            A `Callable` function of type f(<x>, <y>, <p>) -> <bool> that returns True if and only if given the current
            input (<x>), output (<y>), and predictions (<p>), the expected constraints are satisfied.

        :param lb:
            The model variables lower bounds.

        :param ub:
            The model variables upper bounds.

        :param alpha:
            The default alpha value, or None if no alpha step is wanted.

        :param beta:
            The default beta value, or None if no beta step is wanted.

        :param y_loss:
            A `Loss` instance used to compute the y_loss.

        :param p_loss:
            A `Loss` instance used to compute the p_loss.

        :param stats:
            Either a boolean value indicating whether or not to log statistics, or a list of parameters in ['alpha',
            'beta', 'use_beta', 'y_loss', 'p_loss', 'objective'] whose statistics must be logged.
        """
        backend = get_bk(backend) if isinstance(backend, str) else backend
        y_loss = reg_loss(y_loss) if isinstance(y_loss, str) else y_loss
        p_loss = reg_loss(p_loss) if isinstance(p_loss, str) else p_loss
        super(SingleTargetRegression, self).__init__(backend=backend,
                                                     satisfied=satisfied,
                                                     y_loss=y_loss,
                                                     p_loss=p_loss,
                                                     vtype='continuous',
                                                     lb=lb,
                                                     ub=ub,
                                                     alpha=alpha,
                                                     beta=beta,
                                                     stats=stats)

    def _beta_error_variables(self, x, y, p, v) -> np.ndarray:
        # for the regression case, the error may be introduced due to the presence of lower/upper bounds which are not
        # considered in the explicit constraint satisfiability routine, thus we will return clipped predictions
        return p.clip(min=self.lb, max=self.ub)


class SingleTargetClassification(SingleTargetMaster):
    """A standard Moving Targets Master for Single-Target Classification Tasks with no constraints."""

    def __init__(self,
                 backend: Union[str, Backend],
                 satisfied: Callable,
                 alpha: Optional[float] = 1.0,
                 beta: Optional[float] = 1.0,
                 y_loss: Union[str, Loss] = 'hd',
                 p_loss: Union[str, Loss] = 'mse',
                 stats: Union[bool, List[str]] = False):
        """
        :param backend:
            The `Backend` instance encapsulating the optimization solver.

        :param satisfied:
            A `Callable` function of type f(<x>, <y>, <p>) -> <bool> that returns True if and only if given the current
            input (<x>), output (<y>), and predictions (<p>), the expected constraints are satisfied.

        :param alpha:
            The default alpha value, or None if no alpha step is wanted.

        :param beta:
            The default beta value, or None if no beta step is wanted.

        :param y_loss:
            A `Loss` instance used to compute the y_loss.

        :param p_loss:
            A `Loss` instance used to compute the p_loss.

        :param stats:
            Either a boolean value indicating whether or not to log statistics, or a list of parameters in ['alpha',
            'beta', 'use_beta', 'y_loss', 'p_loss', 'objective'] whose statistics must be logged.
        """
        backend = get_bk(backend) if isinstance(backend, str) else backend
        y_loss = cls_loss(y_loss) if isinstance(y_loss, str) else y_loss
        p_loss = cls_loss(p_loss) if isinstance(p_loss, str) else p_loss
        vtype = 'continuous' if (y_loss.use_continuous_targets or p_loss.use_continuous_targets) else 'binary'
        super(SingleTargetClassification, self).__init__(backend=backend,
                                                         satisfied=satisfied,
                                                         y_loss=y_loss,
                                                         p_loss=p_loss,
                                                         vtype=vtype,
                                                         lb=0,
                                                         ub=1,
                                                         alpha=alpha,
                                                         beta=beta,
                                                         stats=stats)

    def _beta_error_variables(self, x, y, p, v) -> np.ndarray:
        # for the classification case, if the loss uses class probabilities the minimal error must be computed between
        # the probabilities and their "class" representation (which are obtained using the 'get_classes' method and
        # then are onehot encoded for compatibility), otherwise since the predictions are already class targets there
        # will be no need to introduce an error, thus we can simply return the prediction themselves
        classes = probabilities.get_classes(prob=p)
        return probabilities.get_onehot(vector=classes, classes=len(np.unique(y)), onehot_binary=False)

    def build(self, x, y, p) -> Any:
        assert y.ndim == 1, "This Master works for univariate classification tasks only"
        samples, classes = len(y), len(np.unique(y))
        # for the binary case, it relies on the super implementation it should return a one-dimensional vector
        # otherwise, it creates a bi-dimensional matrix and impose constraint variables in each row to sum up to one
        if classes == 2:
            return super(SingleTargetClassification, self).build(x, y, p)
        else:
            variables = self.backend.add_variables(samples, classes, vtype=self.vtype, lb=self.lb, ub=self.ub, name='y')
            self.backend.add_constraints([self.backend.sum(row) == 1 for row in variables])
            return variables

    def solution(self, x, y, p, v) -> Any:
        # the get_classes() util function is used both to retrieve the class index in the multiclass classification
        # scenario and to retrieve the binary class in case continuous model variables have been used
        solutions = super(SingleTargetClassification, self).solution(x, y, p, v)
        return probabilities.get_classes(solutions)
