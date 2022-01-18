"""Basic Master Interface."""
import time
from abc import ABC
from typing import Optional, Set
from typing import Union, List

import numpy as np

from moving_targets.callbacks import StatsLogger
from moving_targets.masters.backends import Backend, get_backend as get_bk
from moving_targets.masters.losses import Loss
from moving_targets.masters.losses import regression_loss, classification_loss, RegressionLoss, ClassificationLoss
from moving_targets.masters.optimizers import Optimizer, BetaBoundedSatisfiability, BetaClassSatisfiability
from moving_targets.util import probabilities
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
            backend=get_bk(backend) if isinstance(backend, str) else backend,
            y_loss=regression_loss(y_loss) if isinstance(y_loss, str) else y_loss,
            p_loss=regression_loss(p_loss) if isinstance(p_loss, str) else p_loss,
            alpha=None if alpha is None else Optimizer(base=alpha),
            beta=None if beta is None else BetaBoundedSatisfiability(base=beta, lb=lb, ub=ub),
            stats=stats
        )

    def build(self, x, y: np.ndarray) -> np.ndarray:
        return self.backend.add_continuous_variables(*y.shape, lb=self.lb, ub=self.ub, name='y')


class ClassificationMaster(Master, ABC):
    """An abstract Moving Targets Master for Classification Tasks."""

    def __init__(self,
                 backend: Union[str, Backend],
                 alpha: Union[None, float, Optimizer] = 1.0,
                 beta: Union[None, float, Optimizer] = None,
                 y_loss: Union[str, ClassificationLoss] = 'mse',
                 p_loss: Union[str, ClassificationLoss] = 'mse',
                 vtype: str = 'auto',
                 rtype: str = 'class',
                 task: Union[int, str] = 'auto',
                 stats: Union[bool, List[str]] = False):
        """
        :param backend:
            The `Backend` instance encapsulating the optimization solver.

        :param alpha:
            Either a constant alpha value, an alpha optimizer, or None if no alpha step is wanted.

        :param beta:
            Either a constant beta value, a beta optimizer, or None if no beta step is wanted.

        :param y_loss:
            A `Loss` instance used to compute the y_loss.

        :param p_loss:
            A `Loss` instance used to compute the p_loss.

        :param vtype:
            The model variables vtype, either 'continuous' or 'discrete', or 'auto' for automatic type inference
            depending on the given losses.

        :param rtype:
            The return type of the adjusted targets, either 'class' to get class targets, or 'probability' to get class
            probabilities. Please notice that when returning probabilities, the learner may be adjusted accordingly,
            since some learners do not accept continuous targets and/or bi-dimensional targets.

        :param task:
            The kind of classification task, either 'binary', 'multiclass', or 'multilabel', or 'auto' for automatic
            task inference depending on the given y array. If an integer is passed instead of a string, this will be
            considered as the number of explicit classes in the 'multiclass' (or 'binary', in case of 2 classes) task.

            If 'binary' is chosen, the y array should be one-dimensional and with two classes only.
            If 'multiclass' is chosen, the y array should be one-dimensional and with C > 2 classes.
            If 'multilabel' is chosen, the y array should be bi-dimensional with shape (N, C).

        :param stats:
            Either a boolean value indicating whether or not to log statistics, or a list of parameters in ['alpha',
            'beta', 'use_beta', 'y_loss', 'p_loss', 'objective'] whose statistics must be logged.
        """
        assert rtype in ['class', 'probability'], f"'rtype' should be either 'class' or 'probability', but is '{rtype}'"

        y_loss = classification_loss(y_loss) if isinstance(y_loss, str) else y_loss
        p_loss = classification_loss(p_loss) if isinstance(p_loss, str) else p_loss

        self.vtype: str = vtype
        """The model variables vtype."""

        self.rtype: str = rtype
        """The model variables vtype."""

        self.task: Union[int, str] = task
        """The kind of classification task."""

        # automatic type inference depending on the given losses
        if vtype == 'auto':
            self.vtype = 'continuous' if (y_loss.use_continuous_targets or p_loss.use_continuous_targets) else 'binary'
        elif vtype == 'discrete':
            self.vtype = 'binary'
        elif vtype != 'continuous':
            raise AssertionError(f"'vtype' should be either 'continuous', 'discrete', or 'auto', but is {vtype}")

        super(ClassificationMaster, self).__init__(
            backend=get_bk(backend) if isinstance(backend, str) else backend,
            y_loss=y_loss,
            p_loss=p_loss,
            alpha=None if alpha is None else Optimizer(base=alpha),
            # multi_label is initially set to False, then it will be changed after the task is inferred if necessary
            beta=None if beta is None else BetaClassSatisfiability(base=beta, multi_label=False),
            stats=stats
        )

    def on_process_start(self, macs, x, y: np.ndarray, val_data: Optional[Dataset]):
        # automatic task inference depending on the y array
        y = y.squeeze()
        if self.task == 'auto':
            self.task = 'multi' if y.ndim == 2 else len(np.unique(y))
        elif self.task == 'multilabel':
            self.task = 'multi'
        elif self.task == 'multiclass':
            self.task = len(np.unique(y))
        elif self.task == 'binary':
            self.task = 2
        elif isinstance(self.task, int):
            assert self.task > 1, f"If an integer is passed, {self.task} must be greater than 1"
        else:
            raise AssertionError(f"'task' should be either 'binary', 'multiclass', 'multilabel', 'auto' or a " +
                                 f"positive integer, but it is {self.task}")
        # change "multi_label" parameter value of the 'BetaClassSatisfiability' optimizer if that is the inferred task
        if self.beta is not None and self.task == 'multi':
            self.beta.multi_label = True

    def build(self, x, y: np.ndarray) -> np.ndarray:
        if self.task == 2 or self.task == 'multi':
            # for the binary and the multilabel cases, we create either a one- or bi-dimensional array of model
            # variables depending on the shape of y without the need of imposing any constraint
            variables = self.backend.add_variables(*y.shape, vtype=self.vtype, lb=0, ub=1, name='y')
        else:
            # on the contrary, for the multiclass case, we create a bi-dimensional matrix and constraint variables in
            # each row to sum up to one (the number of classes is stored in the 'task' field)
            variables = self.backend.add_variables(len(y.squeeze()), self.task, vtype=self.vtype, lb=0, ub=1, name='y')
            self.backend.add_constraints([self.backend.sum(row) == 1 for row in variables])
        return variables

    def adjust_targets(self, x, y: np.ndarray) -> np.ndarray:
        # if class targets must be returned, we use the 'get_classes()' function for the binary/multiclass cases
        # while we simply round the predictions for the multilabel case, otherwise there is no need for post-processing
        adjusted = super(ClassificationMaster, self).adjust_targets(x, y)
        if self.rtype == 'class':
            return probabilities.get_classes(adjusted, multi_label=self.task == 'multi')
        return adjusted
