"""Console Logger Callback."""

import time
from typing import Optional

import numpy as np

from moving_targets.callbacks.callback import Callback
from moving_targets.util.typing import Dataset


class ConsoleLogger(Callback):
    """Callback which logs basic information on screen during the `MACS` training."""

    def __init__(self):
        """"""
        super(ConsoleLogger, self).__init__()

        self._time: Optional[float] = None
        """An internal variable used to compute elapsed time between routines."""

    def on_process_start(self, macs, x, y: np.ndarray, val_data: Optional[Dataset]):
        print('Starting the MACS process...')

    def on_iteration_start(self, macs, x, y: np.ndarray, val_data: Optional[Dataset]):
        # Prints the current iteration and stores the initial time.
        print(f'  > it. {macs.iteration:02}:', end=' ')
        self._time = time.time()

    def on_iteration_end(self, macs, x, y: np.ndarray, val_data: Optional[Dataset]):
        # Prints the time elapsed from the iteration start.
        print(f'{time.time() - self._time:.4f} s')
        self._time = None

    def on_process_end(self, macs, x, y: np.ndarray, val_data: Optional[Dataset]):
        print('... process ended correctly!')
