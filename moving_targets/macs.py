"""Core of the Moving Targets algorithm."""
import time
import warnings
from typing import List, Dict, Callable, Union, Optional, Any, Set

import numpy as np

from moving_targets.callbacks.callback import Callback
from moving_targets.callbacks.console_logger import ConsoleLogger
from moving_targets.callbacks.file_logger import FileLogger
from moving_targets.callbacks.history import History
from moving_targets.callbacks.logger import Logger, StatsLogger
from moving_targets.learners.learner import Learner
from moving_targets.masters.masters import Master
from moving_targets.metrics.metric import Metric
from moving_targets.util.masking import mask_data, get_mask
from moving_targets.util.typing import Dataset, is_numeric


class MACS(StatsLogger):
    """Model-Agnostic Constraint Satisfaction.

    This class contains the core algorithm of Moving Targets. It leverages the `Learner` and `Master` instances to
    iteratively refine the predictions during the training phase and, eventually, it evaluates the solutions based on
    the given list of `Metric` objects.
    """

    @staticmethod
    def _parameters() -> Set[str]:
        return {'iteration', 'elapsed_time'}

    def __init__(self,
                 master: Master,
                 learner: Learner,
                 init_step: str = 'pretraining',
                 metrics: List[Metric] = (),
                 mask: Union[None, float, str] = 'learner',
                 stats: Union[bool, List[str]] = False):
        """
        :param master:
            A `Master` instance.

        :param learner:
            A `Learner` instance.

        :param init_step:
            The initial step of the algorithm, which can be either 'pretraining' or 'projection'.

        :param metrics:
            A list of `Metric` instances to evaluate the final solution.

        :param mask:
            Either an (optional) masking value, or a string between 'learner' and 'master' to use the same mask of the
            respective object. The MACS masking value will not be used during the actual fitting process, but rather
            it will be passed to the metric objects.

        :param stats:
            Either a boolean value indicating whether or not to log statistics, or a list of parameters in
            ['iteration', 'elapsed_time'] whose statistics must be logged.
        """
        super(MACS, self).__init__(stats=stats, name='MACS')
        assert init_step in ['pretraining', 'projection'], f"'{init_step}' is not a valid initial step"

        # handle mask
        if mask == 'learner':
            mask = learner.mask
        elif mask == 'master':
            mask = master.mask
        elif not (mask is None or is_numeric(mask)):
            raise AssertionError(f'Unexpected mask "{mask}", possible values are None, "learner", "master", or a float')

        self.master: Master = master
        """The `Master` instance."""

        self.learner: Learner = learner
        """The `Learner` instance."""

        self.init_step: str = init_step
        """The initial step of the algorithm, which can be either 'pretraining' or 'projection'."""

        self.iteration: Optional[int] = None
        """The current `MACS` iteration."""

        self.fitted: bool = False
        """Whether or not the `Learner` has been fitted at least once."""

        self.metrics: List[Metric] = list(metrics)
        """A list of `Metric` instances to evaluate the final solution."""

        self.mask: Optional[float] = mask
        """The (optional) masking value used to mask the original targets."""

        self._history: History = History()
        """The internal `History` object which is returned at the end of the training."""

        self._time: Optional[float] = None
        """An auxiliary variable to keep track of the elapsed time between iterations."""

    def fit(self,
            x,
            y: np.ndarray,
            iterations: int = 1,
            sample_weight: Optional[np.ndarray] = None,
            val_data: Optional[Dataset] = None,
            callbacks: List[Callback] = (),
            verbose: Union[int, bool] = 2) -> History:
        """Fits the `Learner` based on the Moving Targets iterative procedure.

        :param x:
            The matrix/dataframe of training samples.

        :param y:
            The vector of training labels.

        :param iterations:
            The number of algorithm iterations.

        :param sample_weight:
            The (optional) array of sample weights.

        :param val_data:
            A dictionary containing the validation data, indicated as a tuple (xv, yv).

            .. code-block:: python

                val_data = dict(train=(xtr, ytr), validation=(xvl, yvl), test=(xts, yts))

        :param callbacks:
            A list of `Callback` instances.

        :param verbose:
            Either a boolean or an int representing the verbosity value, such that:

            - `0` or `False` create no logger;
            - `1` creates a simple console logger with elapsed time only;
            - `2` or `True` create a more complete console logger with cached data at the end of each iterations.

        :return:
            An instance of the `History` object containing the training history.

        :raise `AssertionError`:
            If the number of iteration is negative, or is zero and the initial step is 'pretraining'.
        """
        # check user input
        assert iterations >= 0, f'the number of iterations should be non-negative, but it is {iterations}'
        assert self.init_step == 'pretraining' or iterations > 0, 'if projection, iterations cannot be zero'
        assert verbose in [True, False, 0, 1, 2], 'unknown verbosity'
        val_data = {} if val_data is None else (val_data if isinstance(val_data, dict) else {'val': val_data})

        # handle callbacks and verbosity (check for type instance since 1 == True and vice versa)
        callbacks = list(callbacks)
        if verbose == 1 and not isinstance(verbose, bool):
            callbacks += [ConsoleLogger()]
        elif verbose == 2 or verbose is True:
            callbacks += [FileLogger(routines=['on_pretraining_end', 'on_iteration_end'])]
        self._update(callbacks, lambda c: c.on_process_start(macs=self, x=x, y=y, val_data=val_data))

        # handle pretraining
        p = None
        if self.init_step == 'pretraining':
            self.iteration = 0
            self._update(callbacks, lambda c: c.on_pretraining_start(macs=self, x=x, y=y, val_data=val_data))
            # ---------------------------------------------- LEARNER STEP ----------------------------------------------
            self.learner.fit(x=x, y=y, sample_weight=sample_weight)
            self.fitted = True
            p = self.learner.predict(x)
            # ---------------------------------------------- LEARNER STEP ----------------------------------------------
            self._update(callbacks, lambda c: c.on_pretraining_end(macs=self, x=x, y=y, p=p, val_data=val_data))

        # algorithm core
        for self.iteration in range(1, iterations + 1):
            self._update(callbacks, lambda c: c.on_iteration_start(macs=self, x=x, y=y, val_data=val_data))
            self._update(callbacks, lambda c: c.on_adjustment_start(macs=self, x=x, y=y, val_data=val_data))
            # ---------------------------------------------- MASTER  STEP ----------------------------------------------
            z = self.master.adjust_targets(x=x, y=y, p=p, sample_weight=sample_weight)
            if z is None:
                # in case of no valid solution, raise a warning and stop the training loop
                warnings.warn(f'No solution found at iteration {self.iteration}, stop training. ' +
                              f'This may have been cause either by time limit or by model infeasibility.')
                break
            # ---------------------------------------------- MASTER  STEP ----------------------------------------------
            self._update(callbacks, lambda c: c.on_adjustment_end(macs=self, x=x, y=y, z=z, val_data=val_data))
            self._update(callbacks, lambda c: c.on_training_start(macs=self, x=x, y=y, val_data=val_data))
            # ---------------------------------------------- LEARNER STEP ----------------------------------------------
            self.learner.fit(x=x, y=z, sample_weight=sample_weight)
            self.fitted = True
            p = self.learner.predict(x)
            # ---------------------------------------------- LEARNER STEP ----------------------------------------------
            self._update(callbacks, lambda c: c.on_training_end(macs=self, x=x, y=y, p=p, val_data=val_data))
            self._update(callbacks, lambda c: c.on_iteration_end(macs=self, x=x, y=y, val_data=val_data))
        self._update(callbacks, lambda c: c.on_process_end(macs=self, x=x, y=y, val_data=val_data))
        return self._history

    def predict(self, x) -> Any:
        """Uses the previously trained `Learner` to predict labels from input samples.

        :param x:
            The matrix/dataframe of input samples.

        :return:
            The vector of predicted labels.

        :raise `AssertionError`:
            If the `Learner` has not been fitted yet.
        """
        assert self.fitted, 'The model has not been fitted yet, please call method .fit()'
        return self.learner.predict(x)

    def evaluate(self, x, y) -> Dict[str, float]:
        """Evaluates the performances of the model based on the given set of metrics.

        :param x:
            The matrix/dataframe of training samples.

        :param y:
            The vector of training labels.

        :return:
            The dictionary of evaluated metrics.

        :raise `AssertionError`:
            If the `Learner` has not been fitted yet.
        """
        return self._compute_metrics(x=x, y=y, p=self.predict(x), metrics=self.metrics, prefix=None)

    def on_iteration_start(self, macs, x, y: np.ndarray, val_data: Optional[Dataset]):
        self._log_stats(iteration=self.iteration)
        self._time = time.time()

    def on_adjustment_end(self, macs, x, y: np.ndarray, z: np.ndarray, val_data: Optional[Dataset]):
        # log metrics on adjusted data
        self.log(**self._compute_metrics(x=x, y=y, p=z, metrics=self.metrics, prefix='adjusted'))

    def on_training_end(self, macs, x, y: np.ndarray, p: np.ndarray, val_data: Optional[Dataset]):
        # log metrics on training data
        self.log(**self._compute_metrics(x=x, y=y, p=p, metrics=self.metrics, prefix='predictions'))

    def on_iteration_end(self, macs, x, y: np.ndarray, val_data: Optional[Dataset]):
        # log metrics on validation data
        for name, (xv, yv) in val_data.items():
            self.log(**self._compute_metrics(x=xv, y=yv, p=self.predict(xv), metrics=self.metrics, prefix=name))
        # log stats
        self._log_stats(elapsed_time=time.time() - self._time)

    def _compute_metrics(self,
                         x,
                         y: np.ndarray,
                         p: np.ndarray,
                         metrics: List[Metric],
                         prefix: Optional[str] = None) -> Dict[str, float]:
        """Compute the metrics results on the given data.

        :param x:
            The input data.

        :param y:
            The output targets.

        :param p:
            The predicted targets.

        :param metrics:
            The list of `Metric` instances for the evaluation.

        :param prefix:
            A prefix to be prepended to the metric name in the results dictionary.

        :return:
            A dictionary of metric results indexed by metric name (optionally with a prefix).
        """
        # mask data based on the output target
        x, y, p = mask_data(x, y, p, mask=get_mask(y, self.mask))
        prefix = "" if prefix is None else f'{prefix}/'
        results = {}
        for metric in metrics:
            value = metric(x, y, p)
            if is_numeric(value):
                # if the metric is a single metric, directly assign the value
                results[f'{prefix}{metric.__name__}'] = value
            elif isinstance(value, dict):
                # if the metric is a multi metric, assign each value per feature by using an underscore as a separator
                for k, v in value.items():
                    results[f'{prefix}{metric.__name__}_{k}'] = v
            else:
                # handle array-like return type by giving a warning
                self.logger.warning(f"Metric '{metric.__name__}' should return either a float or a dictionary.")
                for i, v in enumerate(value):
                    results[f'{prefix}{metric.__name__}_{i}'] = v
        return results

    def _update(self, callbacks: List[Callback], routine: Callable):
        """Runs the given routine for each one of the given callbacks, plus the routine for the `MACS` object itself
        (which is run at the beginning) and the inner `History` callback (which is run at the end).

        :param callbacks:
            The list of callbacks specified during the `fit()` call.

        :param routine:
            The callback routine (e.g., <callback>.on_iteration_start()).
        """
        # run the callback routine for the MACS object itself plus the master and learner instances, for which we do
        # not need to call the 'log' method even though they are 'Logger' instances since their cache is already shared
        routine(self)
        routine(self.master)
        routine(self.learner)
        # run the callback routine for each external callback plus the history logger
        for callback in callbacks + [self._history]:
            # if the callback is a logger, share the internal cache via the 'log' method before calling the routine
            if isinstance(callback, Logger):
                callback.log(**self._cache)
            routine(callback)
        # eventually clear the internal cache
        self._cache = {}
