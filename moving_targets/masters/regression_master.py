"""Basic Regression Master."""

from abc import ABC
from typing import Union, List

import numpy as np

from moving_targets.masters.backends import Backend
from moving_targets.masters.backends import get_backend
from moving_targets.masters.losses import RegressionLoss, Loss, aliases
from moving_targets.masters.losses import SAE, SSE, MAE, MSE
from moving_targets.masters.master import Master
from moving_targets.masters.optimizers import BetaBoundedSatisfiability
from moving_targets.masters.optimizers import Optimizer


class RegressionMaster(Master, ABC):
    """An abstract Moving Targets Master for Regression Tasks."""

    def __init__(self,
                 backend: Union[str, Backend],
                 lb: float = -float('inf'),
                 ub: float = float('inf'),
                 alpha: Union[None, float, Optimizer] = 1.0,
                 beta: Union[None, float, Optimizer] = None,
                 y_loss: Union[str, RegressionLoss] = 'mse',
                 p_loss: Union[str, RegressionLoss] = 'mse',
                 stats: Union[bool, List[str]] = False):
        """
        :param backend:
            The `Backend` instance encapsulating the optimization solver.

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

        self.lb = lb
        """The model variables lower bounds."""

        self.ub = ub
        """The model variables upper bounds."""

        super(RegressionMaster, self).__init__(
            backend=get_backend(backend) if isinstance(backend, str) else backend,
            y_loss=self._get_loss(loss=y_loss),
            p_loss=self._get_loss(loss=p_loss),
            alpha=None if alpha is None else Optimizer(base=alpha),
            beta=None if beta is None else BetaBoundedSatisfiability(base=beta, lb=lb, ub=ub),
            stats=stats
        )

    def build(self, x, y: np.ndarray) -> np.ndarray:
        return self.backend.add_continuous_variables(*y.shape, lb=self.lb, ub=self.ub, name='y')

    @staticmethod
    def _get_loss(loss: Union[str, Loss]) -> Loss:
        """Handles the given loss by checking that it is both known and supported.

        :param loss:
            The given loss.

        :return:
            A `Loss` instance.

        :raise `AssertionError`:
            If either the loss alias is unknown or does not represent a regression loss.
        """
        if isinstance(loss, Loss):
            # if the loss is already a loss instance we simply return it
            return loss
        else:
            # otherwise we retrieve the class from the alias then check if it is both known and supported
            class_type = aliases.get(loss)
            assert class_type is not None, f"Unknown loss alias '{loss}'"
            assert class_type in {SAE, SSE, MAE, MSE}, f"Loss type {class_type.__name__} is not a regression loss."
            return class_type(name=loss)
