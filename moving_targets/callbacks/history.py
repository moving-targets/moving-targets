"""History Callback"""
import warnings
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from moving_targets.callbacks.logger import Logger
from moving_targets.util.typing import Dataset


class History(Logger):
    """Collects the training information and eventually plots it."""

    def __init__(self):
        """"""
        super(History, self).__init__()

        self._history: List = []
        """An internal list which will be used to create the final DataFrame object at the end of the training."""

    def on_iteration_end(self, macs, x, y: np.ndarray, val_data: Optional[Dataset]):
        # Creates a dataframe from the cached items and appends them to a list indexed by iteration.
        self._history.append(pd.DataFrame([self._cache.values()], columns=self._cache.keys()))
        self._cache = {}

    def on_process_end(self, macs, x, y: np.ndarray, val_data: Optional[Dataset]):
        # Creates a single dataframe by concatenating the sub-dataframes from each iteration.
        if len(self._history) > 0:
            self._history = pd.concat(self._history, ignore_index=True)

    def plot(self,
             features: Optional[List[str]] = None,
             num_subplots: Union[str, int] = '/',
             orient_rows: bool = False,
             tight_layout=True,
             figsize=(16, 9),
             **plt_kwargs):
        """Plots the training information which were previously collected in a dataframe.

        :param features:
            List of strings representing the names of the columns to plot. If None, plots all the columns.

        :param num_subplots:
            Number of row/columns in the final subplot, respectively to the value of the orient_rows argument.
            If a string is passed, that is used to display the subplots according to a common prefix (e.g.,
            if num_subplots = '/' and the features are ['trn/loss', 'trn/metric', 'val/loss', 'val/metric'],
            then there will be a row/column for the 'trn' prefix and another one for the 'val' prefix).

        :param orient_rows:
            Whether to orient the plots by column or by row. This influences also the 'num_subplots' parameter since
            when 'orient_rows' is set to True the 'num_subplots' indicates the number of rows while when 'orient_rows'
            is set to False the 'num_subplots' indicates the number of columns.

        :param tight_layout:
            Matplotlib `figure()` argument.

        :param figsize:
            Matplotlib `figure()` argument.

        :param plt_kwargs:
            Additional plot arguments to be passed to `figure()`.
        """
        if not isinstance(self._history, pd.DataFrame):
            warnings.warn('Process did not end correctly, therefore no dataframe can be plotted')
            return
        # HANDLE COLUMNS TO BE PLOTTED (FEATURES)
        #   1. we use the given columns if explicitly passed, otherwise we use the df columns
        #   2. the features are then obtained as those who are a subset of the number data type
        features = self._history.columns if features is None else features
        features = [f for f in features if f is None or np.issubdtype(self._history[f].dtype, np.number)]
        # HANDLE THE SUBPLOT POSITIONS
        #   1. if the given number of subplots is an integer, it arranges the features by taking them in groups of
        #      <num_subplots>, otherwise it means that the argument is a prefix separator thus it arranges the features
        #      by constructing a dictionary of features sharing the prefix and taking them as a group
        #   2. computes the dimensions of the newly arranged features list as the number of rows and the length of the
        #      longest column, then creates a numpy array of None values having that shape and eventually fills it
        #   3. if 'orient_column' is False, transpose the array of positions
        if isinstance(num_subplots, int):
            features = [features[i::num_subplots] for i in range(num_subplots)]
            # the behaviour here is inverted since we are taking the features by <num_subplots> steps
            orient_rows = not orient_rows
        else:
            prefixes = {}
            for column in features:
                prefix = column.split(num_subplots)[0]
                if prefix in prefixes:
                    prefixes[prefix].append(column)
                else:
                    prefixes[prefix] = [column]
            features = [v for v in prefixes.values()]
        num_rows = len(features)
        num_cols = np.max([len(row) for row in features])
        positions = np.array([[None] * num_cols] * num_rows)
        for i, row in enumerate(features):
            # filling up to len(row) is necessary since some rows may have less features than the maximum one
            positions[i, :len(row)] = row
        positions = positions if orient_rows else positions.transpose()
        # PLOT EACH FEATURE IN A SUBPLOT
        #   1. iterates by rows and columns
        #   2. if the feature name is not None, it plots the data
        ax = None
        num_rows, num_cols = positions.shape
        plt.figure(figsize=figsize, tight_layout=tight_layout, **plt_kwargs)
        for idx, feature in enumerate(positions.flatten()):
            if feature is not None:
                ax = plt.subplot(num_rows, num_cols, idx + 1, sharex=ax)
                x, y = self._history.index, self._history[feature]
                plt.plot(x, y)
                plt.scatter(x, y)
                ax.set(title=feature, xlabel='', ylabel='')
                # fix integer x_ticks based on autogenerated ones in order to avoid superimposed values
                ticks = np.unique(ax.get_xticks().round().astype(int))
                ax.set_xticks([t for t in ticks if t in range(x.min(), x.max() + 1)])
        plt.show()
