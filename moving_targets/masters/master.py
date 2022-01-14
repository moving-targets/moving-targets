"""Basic Master Interface."""
import warnings
from typing import Any, Optional, List, Union, Set

import numpy as np

from moving_targets.callbacks import StatsLogger
from moving_targets.masters.backends.backend import Backend
from moving_targets.masters.optimizers import ConstantValue
from moving_targets.masters.optimizers.optimizer import Optimizer
from moving_targets.util.errors import not_implemented_message
from moving_targets.util.typing import Dataset


class Master(StatsLogger):
    """Basic interface for a Moving Targets Master."""

    @staticmethod
    def _parameters() -> Set[str]:
        return {'alpha', 'beta', 'use_beta', 'y_loss', 'p_loss', 'objective'}

    def __init__(self,
                 backend: Backend,
                 alpha: Union[None, float, Optimizer],
                 beta: Union[None, float, Optimizer],
                 stats: Union[bool, List[str]] = False):
        """
        :param backend:
            The `Backend` instance encapsulating the optimization solver.

        :param alpha:
            Either a constant alpha value, an alpha optimizer, or None if no alpha step is wanted.

        :param beta:
            Either a constant beta value, a beta optimizer, or None if no beta step is wanted.

        :param stats:
            Either a boolean value indicating whether or not to log statistics, or a list of parameters in ['alpha',
            'beta', 'use_beta', 'y_loss', 'p_loss', 'objective'] whose statistics must be logged.
        """
        def _get_optimizer(value):
            if isinstance(value, float) or isinstance(value, int):
                return ConstantValue(initial_value=value)
            else:
                return value

        super(Master, self).__init__(stats=stats, logger='Master')

        self.backend: Backend = backend
        """The `Backend` instance encapsulating the optimization solver."""

        self._alpha: Optional[Optimizer] = _get_optimizer(value=alpha)
        """The alpha optimizer, or None if no alpha step is wanted."""

        self._beta: Optional[Optimizer] = _get_optimizer(value=beta)
        """The beta optimizer, or None if no beta step is wanted."""

        self._macs: Optional = None
        """Reference to the MACS object encapsulating the `Master`."""

    def log(self, **cache):
        self._macs.log(**cache)

    def on_process_start(self, macs, x, y: np.ndarray, val_data: Optional[Dataset]):
        self._macs = macs

    def on_process_end(self, macs, val_data: Optional[Dataset]):
        self._macs = None

    def build(self, x, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Creates the model variables and adds the problem constraints.

        :param x:
            The training samples.

        :param y:
            The training labels.

        :param p:
            The `Learner` predictions.

        :return:
            The model variables.
        """
        raise NotImplementedError(not_implemented_message(name='build'))

    def use_beta(self, x, y: np.ndarray, p: np.ndarray, v: np.ndarray) -> bool:
        """Decides whether or not to use the beta step for the given iteration.

        :param x:
            The training samples.

        :param y:
            The training labels.

        :param p:
            The `Learner` predictions.

        :param v:
            The model variables.

        :param v:
            The model variables.

        :return:
            A boolean value representing whether or not to use the beta step for the given iteration.
        """
        raise NotImplementedError(not_implemented_message(name='use_beta'))

    def y_loss(self, x, y: np.ndarray, p: np.ndarray, v: np.ndarray) -> Any:
        """Computes the loss of the model variables wrt real targets.

        :param x:
            The training samples.

        :param y:
            The training labels.

        :param p:
            The `Learner` predictions.

        :param v:
            The model variables.

        :return:
            A real number representing the y_loss.
        """
        raise NotImplementedError(not_implemented_message(name='y_loss'))

    def p_loss(self, x, y: np.ndarray, p: np.ndarray, v: np.ndarray) -> Any:
        """Computes the loss of the model variables wrt predictions.

        :param x:
            The training samples.

        :param y:
            The training labels.

        :param p:
            The `Learner` predictions.

        :param v:
            The model variables.

        :return:
            A real number representing the p_loss.
        """
        raise NotImplementedError(not_implemented_message(name='p_loss'))

    def solution(self, x, y: np.ndarray, p: np.ndarray, v: np.ndarray) -> Any:
        """Processes and returns the solutions given by the optimization model.

        :param x:
            The matrix/dataframe of training samples.

        :param y:
            The vector of training labels.

        :param p:
            The information returned by the 'build_model' function.

        :param v:
            The model variables.

        :return:
            Either a simple vector of adjusted targets or a tuple containing the vector and a dictionary of kwargs.
        """
        raise NotImplementedError(not_implemented_message(name='solution'))

    def adjust_targets(self, x, y) -> Any:
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
        pred = self._macs.predict(x) if self._macs.fitted else None
        var = self.build(x=x, y=y, p=pred)
        y_loss = self.y_loss(x=x, y=y, p=pred, v=var)
        p_loss = None if pred is None else self.p_loss(x=x, y=y, p=pred, v=var)
        # check for feasibility and behave depending on that (if we are in the projection initial step, there is no
        # need to choose since both the strategies will collapse to the same formulation)
        if p_loss is None:
            self.backend.minimize(cost=y_loss)
        elif self.use_beta(x=x, y=y, p=pred, v=var):
            beta = self._beta(macs=self._macs, x=x, y=y, p=pred)
            self._log_stats(use_beta=1, beta=beta)
            self.backend.add_constraint(constraint=p_loss <= beta)
            self.backend.minimize(cost=y_loss)
        else:
            alpha = self._alpha(macs=self._macs, x=x, y=y, p=pred)
            self._log_stats(use_beta=0, alpha=alpha)
            self.backend.minimize(cost=y_loss + (1.0 / alpha) * p_loss)
        # solve the problem and get the adjusted labels
        if self.backend.solve().solution is None:
            warnings.warn(f'Model is infeasible at iteration {self._macs.iteration}, stop training.')
            return None
        self._log_stats(
            y_loss=self.backend.get_value(y_loss),
            p_loss=None if p_loss is None else self.backend.get_value(p_loss),
            objective=self.backend.get_objective()
        )
        self.backend.clear()
        return self.solution(x=x, y=y, p=pred, v=var)
