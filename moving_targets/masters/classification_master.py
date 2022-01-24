"""Basic Classification Master."""
from abc import ABC
from typing import Set, Callable, Optional
from typing import Union, List

import numpy as np

from moving_targets.masters.backends import Backend, get_backend
from moving_targets.masters.losses import SAE, SSE, MAE, MSE, Loss, aliases, HammingDistance, CrossEntropy, \
    ReversedCrossEntropy, SymmetricCrossEntropy
from moving_targets.masters.master import Master
from moving_targets.masters.optimizers import Optimizer, BetaClassSatisfiability
from moving_targets.util import probabilities
from moving_targets.util.typing import Dataset


class ClassificationMaster(Master, ABC):
    """An abstract Moving Targets Master for Classification Tasks."""

    TASKS: Set[str] = {'auto', 'binary', 'multiclass', 'multilabel'}
    """Set of accepted tasks."""

    VTYPES: Set[str] = {'auto', 'discrete', 'continuous'}
    """Set of accepted vtypes."""

    RTYPES: Set[str] = {'class', 'probability'}
    """Set of accepted rtypes."""

    def __init__(self,
                 backend: Union[str, Backend],
                 alpha: Union[None, float, Optimizer] = 1.0,
                 beta: Union[None, float, Optimizer] = None,
                 y_loss: Union[str, Loss] = 'hd',
                 p_loss: Union[str, Loss] = 'mse',
                 task: str = 'auto',
                 vtype: str = 'auto',
                 rtype: str = 'class',
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

        :param task:
            The kind of classification task, which can be either 'binary', 'multiclass', 'multilabel', or 'auto'.

            If 'binary' is chosen, the model variables will be stored in a one-dimensional vector.
            If 'multiclass' is chosen, the model variables will be stored in a matrix so that each row sum up to one.
            If 'multilabel' is chosen, the model variables will be stored in a matrix without any summing constraint.
            If 'auto' is chosen, it will choose between 'binary' or 'multiclass' depending on the number of classes
            (if you need to handle a multilabel classification task, you should explicitly declare it).

        :param vtype:
            The model variables vtype, either 'continuous' or 'discrete', or 'auto' for automatic type inference
            depending on the given losses.

        :param rtype:
            The return type of the adjusted targets, either 'class' to get class targets, or 'probability' to get class
            probabilities. Please notice that when returning probabilities, the learner may be adjusted accordingly,
            since some learners do not accept continuous targets and/or bi-dimensional targets.

        :param stats:
            Either a boolean value indicating whether or not to log statistics, or a list of parameters in ['alpha',
            'beta', 'use_beta', 'y_loss', 'p_loss', 'objective'] whose statistics must be logged.

        :raise `TypeError`:
            If two losses cannot handle the same (forced) variable types or if a loss cannot handle an externally
            forced variable type.
        """
        assert task in self.TASKS, f"'task' should be in {self.TASKS}, got '{task}'"
        assert vtype in self.VTYPES, f"'vtype' should be in {self.VTYPES}, got '{vtype}'"
        assert rtype in self.RTYPES, f"'rtype' should be in {self.RTYPES}, got '{rtype}'"
        # handle losses aliases and automatic type inference, e.g.:
        #   > if we have (y_loss = 'hd', p_loss = 'mse', vtype = 'auto') we will set binary = True since hamming
        #     distance can only handle binary targets
        #   > similarly, if we have (y_loss = 'hd', p_loss = 'rce', vtype = 'auto'), because even though reversed
        #     crossentropy prefers continuous targets, hamming distance forces them to be binary
        #   > on the contrary if, for example, we have (y_loss = 'mse', p_loss = 'rce', vtype = 'auto'), then we will
        #     set binary = False since reversed crossentropy prefers continuous targets and mean squared error does not
        #     force any variable type
        #   > however, if we get have (y_loss = 'mse', p_loss = 'rce', vtype = 'binary') then binary = True because
        #     the variables type is externally forced
        #   > moreover, if a given loss is already a Loss instance, this is supposed to expect binary targets if the
        #     variable types are not forced by the user
        #   > finally, if we have something like (y_loss = 'hd', p_loss = 'rce', vtype = 'continuous'), then we will
        #     raise a TypeError because targets are forced to be continuous but hamming distance cannot handle them
        y_loss, p_loss = self._get_loss(loss=y_loss, task=task), self._get_loss(loss=p_loss, task=task)
        force_binary = vtype == 'discrete' or y_loss.force_binary or p_loss.force_binary
        force_continuous = vtype == 'continuous' or y_loss.force_continuous or p_loss.force_continuous
        if not force_binary or not force_continuous:
            # if either one between binary and continuous is not forced, we can proceed and set the correct type
            binary = force_binary or (y_loss.prefer_binary and p_loss.prefer_binary)
            y_loss, p_loss = y_loss.loss_fn(b=binary), p_loss.loss_fn(b=binary)
            vtype = 'binary' if binary else 'continuous'
        elif vtype == 'auto':
            # if vtype is 'auto', it means that the two losses forced two different types, thus they cannot go together
            raise TypeError(f"y_loss '{y_loss}' and p_loss '{p_loss}' cannot handle the same vtypes")
        else:
            # otherwise, it means that the vtype that was externally forced cannot be handled by some of the losses
            raise TypeError(f"Either y_loss '{y_loss}' or p_loss '{p_loss}' cannot deal with vtype='{vtype}'")

        self.task: str = task
        """The kind of classification task."""

        self.vtype: str = vtype
        """The model variables vtype."""

        self.rtype: str = rtype
        """The model variables vtype."""

        super(ClassificationMaster, self).__init__(
            backend=get_backend(backend) if isinstance(backend, str) else backend,
            y_loss=y_loss,
            p_loss=p_loss,
            alpha=None if alpha is None else Optimizer(base=alpha),
            beta=None if beta is None else BetaClassSatisfiability(base=beta, task='classification'),
            stats=stats
        )

    def on_process_start(self, macs, x, y: np.ndarray, val_data: Optional[Dataset]):
        # set task in case of automatic task detection based on the number of classes
        if self.task == 'auto':
            assert y.squeeze().ndim == 1, "'auto' task can handle 1d targets only, use 'multilabel' for 2d targets"
            self.task = 'binary' if probabilities.count_classes(vector=y) == 2 else 'multiclass'

    def build(self, x, y: np.ndarray) -> np.ndarray:
        # we can simply build variables by shape (the y targets will be already onehot encoded for multiclass tasks)
        variables = self.backend.add_variables(*y.shape, vtype=self.vtype, lb=0, ub=1, name='y')
        if self.task == 'multiclass':
            # if we are dealing with multiclass classification, we create a bi-dimensional matrix and constraint the
            # variables in each row to sum up to one since we want class probabilities
            self.backend.add_constraints([self.backend.sum(row) == 1 for row in variables])
        return variables

    def adjust_targets(self, x, y: np.ndarray) -> np.ndarray:
        if self.task == 'multiclass':
            # for multiclass classification tasks, we need to acknowledge that MT losses must work with both original
            # targets and predictions thus, since predictions usually come as 2d floating point array of probabilities
            # while class targets usually come as an integer vector, we must onehot encode the latter for compatibility
            # (this will not be necessary for multilabel classification since the original targets will be already 2d)
            y = probabilities.get_onehot(y, onehot_binary=True)
            task = 'classification'
        else:
            # for binary tasks, the parameter 'task' of get_discrete() is ignored, thus we can simply pass 'labelling'
            task = 'labelling'
        adjusted = super(ClassificationMaster, self).adjust_targets(x, y)
        return probabilities.get_discrete(adjusted, task=task) if self.rtype == 'class' else adjusted

    class _LossInfo:
        """Data class containing information about a loss."""

        def __init__(self,
                     loss_fn: Callable,
                     force_binary: bool = False,
                     force_continuous: bool = False,
                     prefer_binary: bool = True):
            """
            :param loss_fn:
                A callable function of type f(<binary>) -> <Loss> that builds the loss instance.

            :param force_binary:
                Whether the loss can handle only binary variables or not.

            :param force_continuous:
                Whether the loss can handle only continuous variables or not.

            :param prefer_binary:
                Whether the loss prefers to handle binary variables by default or not.
            """
            self.loss_fn: Callable = loss_fn
            """A callable function of type f(<binary>) -> <Loss> that builds the loss instance."""

            self.force_binary: bool = force_binary
            """Whether the loss can handle only binary variables or not."""

            self.force_continuous: bool = force_continuous
            """Whether the loss can handle only continuous variables or not."""

            self.prefer_binary: bool = prefer_binary
            """Whether the loss prefers to handle binary variables by default or not."""

    @classmethod
    def _get_loss(cls, loss: Union[str, Loss], task: str) -> _LossInfo:
        """Handles the given loss by checking, then returns its respective `_LossInfo` instance.

        :param loss:
            The given loss.

        :param task:
            The given task.

        :return:
            A `Loss` instance.

        :raise `AssertionError`:
            If the loss alias is unknown.
        """
        if isinstance(loss, str):
            # if we get passed a string, we retrieve the class type and build a correct instance accordingly
            class_type = aliases.get(loss)
            if class_type == HammingDistance:
                # hamming distance can handle only binary targets, also, for this loss we need to specify the task
                hamming_task = 'labelling' if task == 'multilabel' else 'classification'
                return cls._LossInfo(loss_fn=lambda b: HammingDistance(task=hamming_task, name=loss), force_binary=True)
            elif class_type == CrossEntropy:
                # crossentropy can handle only binary targets
                return cls._LossInfo(loss_fn=lambda b: CrossEntropy(name=loss), force_binary=True)
            elif class_type in [ReversedCrossEntropy, SymmetricCrossEntropy]:
                # reversed and symmetric crossentropy prefer continuous targets but can handle both
                return cls._LossInfo(loss_fn=lambda b: class_type(binary=b, name=loss), prefer_binary=False)
            elif class_type in [SAE, SSE, MAE, MSE]:
                return cls._LossInfo(loss_fn=lambda b: class_type(binary=b, name=loss))
            else:
                raise AssertionError(f"Unknown loss alias '{loss}'")
        else:
            # otherwise, if the loss is already a loss instance, we ignore the <binary> parameter and return it
            return cls._LossInfo(loss_fn=lambda b: loss)
