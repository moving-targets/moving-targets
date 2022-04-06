"""Data Logger Callback."""

from typing import Optional, List

import numpy as np
import pandas as pd

from moving_targets.callbacks.callback import Callback
from moving_targets.util.typing import Dataset


class DataLogger(Callback):
    """Callback which stores the input data along with the predictions and adjustments after each iteration."""

    def __init__(self, filepath: Optional[str] = None):
        """
        :param filepath:
            Path string in which to write the stored data. Depending on the path extension, the data will be stored as
             a 'csv', 'tsv' or 'text' file. If None is passed, does not write any data.
        """
        super(DataLogger, self).__init__()

        self.filepath: Optional[str] = filepath
        """Path string in which to write the stored data."""

        self.data: pd.DataFrame = pd.DataFrame()
        """Dataframe object in which to store the input and output data."""

        self.iterations: List[int] = []
        """The list of macs iterations."""

    def on_process_start(self, macs, x, y: np.ndarray, val_data: Optional[Dataset]):
        self.data = pd.concat((self.data, x), axis=1)
        self.data['y'] = y

    def on_iteration_start(self, macs, x, y: np.ndarray, val_data: Optional[Dataset]):
        self.iterations.append(macs.iteration)

    def on_training_end(self, macs, x, y: np.ndarray, p: np.ndarray, val_data: Optional[Dataset]):
        self.data[f'p{macs.iteration}'] = p

    def on_adjustment_end(self, macs, x, y: np.ndarray, z: np.ndarray, val_data: Optional[Dataset]):
        self.data[f'z{macs.iteration}'] = z

    def on_process_end(self, macs, x, y: np.ndarray, val_data: Optional[Dataset]):
        if self.filepath is None:
            return
        elif self.filepath.endswith('.csv'):
            self.data.to_csv(self.filepath, index=False)
        elif self.filepath.endswith('.tsv'):
            self.data.to_csv(self.filepath, index=False, sep='\t')
        else:
            with open(self.filepath, 'w') as f:
                f.write(self.data.to_string())
