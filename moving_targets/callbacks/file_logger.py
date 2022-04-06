"""File Logger Callback"""

import sys
from typing import List, Set, Optional

import numpy as np

from moving_targets.callbacks.logger import Logger
from moving_targets.util.typing import Dataset

SEPARATOR: str = '--------------------------------------------------'


class FileLogger(Logger):
    """Logs the training data and information on a specified filepath or on the standard output."""

    def __init__(self, filepath: Optional[str] = None, routines: Optional[List[str]] = None, log_empty: bool = False,
                 sort_keys: bool = False, separator: str = SEPARATOR, end: str = '\n'):
        """
        :param filepath:
            Path string in which to put the log. If None, writes in the standard output.

        :param routines:
            List of routine names after which log the cached data. If None, logs after each routine.

        :param log_empty:
            Whether or not to log the routine name if there is no cached data.

        :param sort_keys:
            Whether or not to sort the cache data by key before logging.

        :param separator:
            String separator written between each routine.

        :param end:
            Line end.
        """
        super(FileLogger, self).__init__()

        self.filepath: str = filepath
        """Path string in which to put the log. If None, writes in the standard output."""

        self.routines: Optional[Set[str]] = None if routines is None else set(routines)
        """List of routine names after which log the cached data. If None, logs after each routine."""

        self.log_empty: bool = log_empty
        """Whether or not to log the routine name if there is no cached data."""

        self.sort_keys: bool = sort_keys
        """Whether or not to sort the cache data by key before logging."""

        self.separator: str = separator
        """String separator written between each routine."""

        self.end: str = end
        """Line end."""

        self._logged_once: bool = False
        """An internal variable used to write the initial line separator."""

    def on_process_start(self, macs, x, y: np.ndarray, val_data: Optional[Dataset]):
        # reset file content for overwriting
        if self.filepath is not None:
            open(self.filepath, 'w').close()

        self._write_on_file(macs, 'PROCESS START', 'on_process_start')

    def on_process_end(self, macs, x, y: np.ndarray, val_data: Optional[Dataset]):
        self._write_on_file(macs, 'PROCESS END', 'on_process_end')

    def on_pretraining_start(self, macs, x, y: np.ndarray, val_data: Optional[Dataset]):
        self._write_on_file(macs, 'PRETRAINING START', 'on_pretraining_start')

    def on_pretraining_end(self, macs, x, y: np.ndarray, p: np.ndarray, val_data: Optional[Dataset]):
        self._write_on_file(macs, 'PRETRAINING END', 'on_pretraining_end')

    def on_iteration_start(self, macs, x, y: np.ndarray, val_data: Optional[Dataset]):
        self._write_on_file(macs, f'ITERATION START', 'on_iteration_start')

    def on_iteration_end(self, macs, x, y: np.ndarray, val_data: Optional[Dataset]):
        self._write_on_file(macs, f'ITERATION END', 'on_iteration_end')

    def on_training_start(self, macs, x, y: np.ndarray, val_data: Optional[Dataset]):
        self._write_on_file(macs, 'TRAINING START', 'on_training_start')

    def on_training_end(self, macs, x, y: np.ndarray, p: np.ndarray, val_data: Optional[Dataset]):
        self._write_on_file(macs, 'TRAINING END', 'on_training_end')

    def on_adjustment_start(self, macs, x, y: np.ndarray, val_data: Optional[Dataset]):
        self._write_on_file(macs, 'ADJUSTMENT START', 'on_adjustment_start')

    def on_adjustment_end(self, macs, x, y: np.ndarray, z: np.ndarray, val_data: Optional[Dataset]):
        self._write_on_file(macs, 'ADJUSTMENT END', 'on_adjustment_end')

    def _write_on_file(self, macs, message: str, routine_name: str):
        """This is the core function, it is in charge of opening the chosen file and write the cached data when needed.

        :param macs:
            Reference to the `MACS` object encapsulating the master.

        :param message:
            An initial message to be printed before the cached data.

        :param routine_name:
            The name of the routine that just ended.
        """
        # in case no routine is inserted (i.e., all of them are considered) and either we are asked to log
        # independently from the presence of cached data or there is actually some cached data to log
        if (self.routines is None or routine_name in self.routines) and (self.log_empty or len(self._cache) > 0):
            # open file
            file = sys.stdout if self.filepath is None else open(self.filepath, 'a', encoding='utf8')
            # write initial separator if needed
            if not self._logged_once:
                file.write(f'{self.separator}{self.end}')
                self._logged_once = True
            # write message and cached items if present
            file.write(f'ITERATION {macs.iteration:02} - {message}{self.end}')
            cache = {k: self._cache[k] for k in sorted(self._cache)} if self.sort_keys else self._cache
            for k, v in cache.items():
                file.write(f'> {str(k)} = {str(v)}{self.end}')
            # write write separator and empty cache
            file.write(f'{self.separator}{self.end}')
            self._cache = {}
            # close file if not stdout
            if self.filepath is not None:
                file.close()
