"""Basic Master Interface."""
import time
from typing import Optional, Set
from typing import Union, List

import numpy as np

from moving_targets.callbacks import StatsLogger
from moving_targets.masters.backends import Backend
from moving_targets.masters.losses import Loss
from moving_targets.masters.optimizers import Optimizer
from moving_targets.util.errors import not_implemented_message
from moving_targets.util.typing import Dataset


class Master(StatsLogger):
    """Basic interface for a Moving Targets Master."""

    @staticmethod
    def _parameters() -> Set[str]:
        return {'alpha', 'beta', 'use_beta', 'y_loss', 'p_loss', 'objective', 'elapsed_time'}

    def __init__(self,
                 backend: Backend,
                 y_loss: Loss,
                 p_loss: Loss,
                 alpha: Optional[Optimizer],
                 beta: Optional[Optimizer],
                 stats: Union[bool, List[str]]):
        """
        :param backend:
            The `Backend` instance encapsulating the optimization solver.

        :param y_loss:
            A `Loss` instance used to compute the y_loss.

        :param p_loss:
            A `Loss` instance used to compute the p_loss.

        :param alpha:
            Either an optimizer for the alpha value, or None if no alpha step is wanted.

        :param beta:
            Either an optimizer for the beta value, or None if no beta step is wanted.

        :param stats:
            Either a boolean value indicating whether or not to log statistics, or a list of parameters in ['alpha',
            'beta', 'use_beta', 'y_loss', 'p_loss', 'objective'] whose statistics must be logged.
        """

        super(Master, self).__init__(stats=stats, name='Master')

        self.backend: Backend = backend
        """The `Backend` instance encapsulating the optimization solver."""

        self.y_loss: Loss = y_loss
        """The `Loss` instance used to compute the y_loss."""

        self.p_loss: Loss = p_loss
        """The `Loss` instance used to compute the p_loss."""

        self.alpha: Optional[Optimizer] = alpha
        """The alpha optimizer, or None if no alpha step is wanted."""

        self.beta: Optional[Optimizer] = beta
        """The beta optimizer, or None if no beta step is wanted."""

        self._macs: Optional = None
        """Reference to the MACS object encapsulating the `Master`."""

        self._time: Optional[float] = None
        """An auxiliary variable to keep track of the elapsed time between iterations."""

    def log(self, **cache):
        self._macs.log(**cache)

    def on_adjustment_start(self, macs, x, y: np.ndarray, val_data: Optional[Dataset]):
        self._macs = macs
        self._time = time.time()

    def on_adjustment_end(self, macs, x, y: np.ndarray, adjusted_y: np.ndarray, val_data: Optional[Dataset]):
        self._log_stats(elapsed_time=time.time() - self._time)
        self._time = None
        self._macs = None

    def build(self, x, y: np.ndarray) -> np.ndarray:
        """Creates the model variables and adds the problem constraints.

        :param x:
            The training samples.

        :param y:
            The training labels.

        :return:
            The model variables.
        """
        raise NotImplementedError(not_implemented_message(name='build'))

    def use_beta(self, x, y: np.ndarray, p: np.ndarray) -> bool:
        """Decides whether or not to use the beta step for the given iteration.

        :param x:
            The training samples.

        :param y:
            The training labels.

        :param p:
            The `Learner` predictions.

        :return:
            A boolean value representing whether or not to use the beta step for the given iteration.
        """
        raise ValueError("This master has not implemented any strategy to choose between the alpha and beta steps. " +
                         "Please override a custom 'use_beta' method or set either alpha or beta to None.")

    def adjust_targets(self, x, y: np.ndarray) -> np.ndarray:
        """Core function of the `Master` object which builds the model and returns the adjusted targets.

        :param x:
            The matrix/dataframe of training samples.

        :param y:
            The vector of training labels.

        :return:
            The vector of adjusted targets, potentially with a dictionary of additional information.
        """
        assert self._macs is not None, "No reference to the MACS object encapsulating the Master"
        self.backend.build()
        v = self.build(x=x, y=y)
        y_loss = self.y_loss(backend=self.backend, numeric_variables=y, model_variables=v)
        # if macs is not fitted (i.e., we are in the first 'projection' iteration), there is no need to choose a
        # strategy since both of them will collapse to the same formulation, namely minimizing the y_loss only,
        # otherwise we use the beta step if either alpha is None, or beta is not None and use_beta() returns true
        if self._macs.fitted:
            p = self._macs.predict(x)
            p_loss = self.p_loss(backend=self.backend, numeric_variables=p, model_variables=v)
            # leave use_beta at the end to avoid function evaluation if one of the two conditions hold before
            if self.alpha is None or (self.beta is not None and self.use_beta(x=x, y=y, p=p)):
                beta = self.beta(macs=self._macs, x=x, y=y, p=p)
                self._log_stats(use_beta=1, beta=beta)
                self.backend.add_constraint(constraint=p_loss <= beta)
                self.backend.minimize(cost=y_loss)
            else:
                alpha = self.alpha(macs=self._macs, x=x, y=y, p=p)
                self._log_stats(use_beta=0, alpha=alpha)
                self.backend.minimize(cost=y_loss + (1.0 / alpha) * p_loss)
        else:
            p_loss = None
            self.backend.minimize(cost=y_loss)
        # if the problem is infeasible return no None, otherwise log stats and return the adjusted labels
        adjusted = None
        if self.backend.solve().solution is not None:
            adjusted = self.backend.get_values(expressions=v)
            self._log_stats(
                y_loss=self.backend.get_value(y_loss),
                p_loss=None if p_loss is None else self.backend.get_value(p_loss),
                objective=self.backend.get_objective()
            )
        self.backend.clear()
        return adjusted
