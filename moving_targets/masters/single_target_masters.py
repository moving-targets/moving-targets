"""Abstract Default Masters."""
from typing import Any, Callable, Union, List

import numpy as np

from moving_targets.masters.backends import Backend, get_backend as get_bk
from moving_targets.masters.losses import regression_loss as reg_loss, classification_loss as cls_loss
from moving_targets.masters.losses.loss import Loss
from moving_targets.masters.master import Master
from moving_targets.masters.optimizers import Optimizer
from moving_targets.util import probabilities
from moving_targets.util.typing import Number


class SingleTargetMaster(Master):
    """Template class for a Moving Targets Master for Single-Target Tasks."""

    def __init__(self,
                 backend: Backend,
                 satisfied: Callable,
                 vtype: str,
                 lb: Number,
                 ub: Number,
                 alpha: Union[None, Number, Optimizer],
                 beta: Union[None, Number, Optimizer],
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
            Either a constant alpha value, an alpha optimizer, or None if no alpha step is wanted.

        :param beta:
            Either a constant beta value, a beta optimizer, or None if no beta step is wanted.

        :param y_loss:
            A `Loss` instance used to compute the y_loss.

        :param p_loss:
            A `Loss` instance used to compute the p_loss.

        :param stats:
            Either a boolean value indicating whether or not to log statistics, or a list of parameters in ['alpha',
            'beta', 'use_beta', 'y_loss', 'p_loss', 'objective'] whose statistics must be logged.
        """
        super(SingleTargetMaster, self).__init__(backend=backend, alpha=alpha, beta=beta, stats=stats)

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

        self._satisfied: Callable = satisfied
        """The `Callable` function that returns True if and only the expected constraints are satisfied."""

    def build(self, x, y, p) -> Any:
        assert y.ndim == 1, "This master works for single-targets tasks only"
        return self.backend.add_variables(len(y), vtype=self.vtype, lb=self.lb, ub=self.ub, name='y')

    def use_beta(self, x, y, p, v) -> bool:
        # if p is None (i.e., initial_step == 'projection' and iteration == 0), this method should not be called at all
        # since the strategies will collapse to the same formulation, i.e., minimizing the y_loss only
        if p is None:
            raise ValueError("The method 'use_beta' should not be called when there are no predictions.")
        elif self._alpha is None:
            return True
        elif self._beta is None:
            return False
        else:
            return self._satisfied(x, y, p)

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
                 alpha: Union[None, Number, Optimizer] = 1.0,
                 beta: Union[None, Number, Optimizer] = None,
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
            Either a constant alpha value, an alpha optimizer, or None if no alpha step is wanted.

        :param beta:
            Either a constant beta value, a beta optimizer, or None if no beta step is wanted.

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


class SingleTargetClassification(SingleTargetMaster):
    """A standard Moving Targets Master for Single-Target Classification Tasks with no constraints."""

    def __init__(self,
                 backend: Union[str, Backend],
                 satisfied: Callable,
                 alpha: Union[None, Number, Optimizer] = 1.0,
                 beta: Union[None, Number, Optimizer] = None,
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
            Either a constant alpha value, an alpha optimizer, or None if no alpha step is wanted.

        :param beta:
            Either a constant beta value, a beta optimizer, or None if no beta step is wanted.

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
