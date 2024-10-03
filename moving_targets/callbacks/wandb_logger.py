"""Weights&Biases Callback"""

from typing import Dict, Optional

import numpy as np

from moving_targets.callbacks.logger import Logger
from moving_targets.util.errors import MissingDependencyError
from moving_targets.util.typing import Dataset


class WandBLogger(Logger):
    """Logs the training information on a Weights&Biases instance."""

    def __init__(self, project: str, entity: str, run_name: str, log_vectors: bool = True, **wandb_kwargs):
        """
        :param project:
            Weights&Biases project name.

        :param entity:
            Weights&Biases entity name.

        :param run_name:
            Weights&Biases run name.

        :param log_vectors:
            Whether to store the whole predictions and the adjustment targets at the end of every iteration.

        :param wandb_kwargs:
            Weights&Biases run configuration.
        """
        super(WandBLogger, self).__init__()

        try:
            import wandb
            self._wandb = wandb
            """The lazily imported wandb instance."""
        except ModuleNotFoundError:
            raise MissingDependencyError(package='wandb')

        self._project: str = project
        """The Weights&Biases project name."""

        self._entity: str = entity
        """The Weights&Biases entity name."""

        self._run_name: str = run_name
        """The Weights&Biases run name."""

        self._log_vectors: bool = log_vectors
        """Whether to store the whole predictions and the adjustment targets at the end of every iteration."""

        self.config: Dict = wandb_kwargs
        """The Weights&Biases run configuration."""

    def on_process_start(self, macs, x, y: np.ndarray, val_data: Optional[Dataset]):
        self._wandb.init(project=self._project, entity=self._entity, name=self._run_name, config=self.config)

    def on_iteration_end(self, macs, x, y: np.ndarray, val_data: Optional[Dataset]):
        self._wandb.log({k: self._cache[k] for k in sorted(self._cache)})
        self._cache = {}

    def on_training_end(self, macs, x, y: np.ndarray, p: np.ndarray, val_data: Optional[Dataset]):
        if self._log_vectors:
            self._wandb.log({'predictions': list(p)})

    def on_adjustment_end(self, macs, x, y: np.ndarray, z: np.ndarray, val_data: Optional[Dataset]):
        if self._log_vectors:
            self._wandb.log({'adjusted': list(z)})

    def on_process_end(self, macs, x, y: np.ndarray, val_data: Optional[Dataset]):
        self._wandb.finish()
