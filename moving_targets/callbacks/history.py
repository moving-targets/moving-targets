"""History Callback"""
import fnmatch
import warnings
from typing import List, Optional, Union, Tuple

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
        self._history.append(pd.DataFrame([self._cache.values()], columns=list(self._cache.keys())))
        self._cache = {}

    def on_process_end(self, macs, x, y: np.ndarray, val_data: Optional[Dataset]):
        # Creates a single dataframe by concatenating the sub-dataframes from each iteration.
        if len(self._history) > 0:
            self._history = pd.concat(self._history, ignore_index=True)

    def plot(self,
             features: Union[None, int, str, List[str], List[List[Optional[str]]]] = '/',
             excluded: Optional[List[str]] = None,
             orient_rows: bool = False,
             title: Optional[str] = None,
             titles_pad: int = 20,
             tight_layout: bool = True,
             figsize: Optional[Tuple[int, int]] = (16, 9),
             **plt_kwargs):
        """Plots the training information which were previously collected in a dataframe.

        :param features:
            Indicates which feature to plot as well as the plotting strategy.
            If None is passed, all the (numeric) features are automatically displayed according to the figure ratio.
            If an integer is passed, all the (numeric) features are plotted and the parameter indicates number of row
            or columns in the final subplot, respectively to the value of the orient_rows argument.
            If a string is passed, this is used to display the subplots according to a common prefix (for instance, if
            features = '/' and the features are ['trn/loss', 'trn/metric', 'val/loss', 'val/metric'], then there will
            be a row or column for the 'trn' prefix and another one for the 'val' prefix).
            If a list of strings is passed, these strings are considered as patters, therefore there will be a row or
            column for each feature matching with the pattern (each pattern can be either the exact name of a feature
            or it can contain a wildcard, e.g., '*/mse' or 'predictions/*').
            If a list of lists is passed, it displays the subplots accordingly to each entry in the grid (if an entry
            is None, the respective subplot will be left empty).

        :param excluded:
            The list of either patterns or full feature names to be excluded from the plots.

        :param orient_rows:
            Whether to orient the plots by column or by row. This influences also the 'num_subplots' parameter since
            when 'orient_rows' is set to True the 'num_subplots' indicates the number of rows while when 'orient_rows'
            is set to False the 'num_subplots' indicates the number of columns.

        :param title:
            The (optional) title of the whole plot.

        :param titles_pad:
            The padding of each subplot title to avoid overlapping with labels.

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
        # 1. first of all, retrieve all the columns that can be plotted (i.e., they are numeric)
        # 2. if <excluded> is not None, builds the set of excluded features matching the patterns and removes each of
        #    them from the list of columns
        # 3. if <features> is None, compute the best subplots disposition according to their number, the figure ratio,
        #    and the orientation, so that it is brought back to the case in which <features> is an integer
        # 4. if <features> is a string, retrieve all the prefixes matching that separator and store them into a list,
        #    so that it is brought back to the case in which <features> is a list of patterns
        columns = [c for c in self._history.columns if np.issubdtype(self._history[c].dtype, np.number)]
        if excluded is not None:
            excluded = {c for e in excluded for c in fnmatch.filter(columns, e)}
            columns = [c for c in columns if c not in excluded]
        if features is None:
            ratio = figsize[0] / figsize[1] if orient_rows else figsize[1] / figsize[0]
            features = int(max(np.sqrt(ratio * len(columns)).round(), 1))
        elif isinstance(features, str):
            prefixes = {f'{column.split(features)[0]}' for column in columns}
            features = [f'{prefix}{features}*' for prefix in prefixes]
        # HANDLE THE SUBPLOT POSITIONS
        # 1. if <features> is an integer, it arranges the subplots by taking them in groups of <features>; moreover the
        #    orientation behaviour is inverted since columns are taken by <features> steps
        # 2. otherwise, <features> must be either a list of strings or a list of lists; in the first case, each string
        #    is mapped to the list of columns matching for the pattern, otherwise it means that this is the list of
        #    lists directly passed by the user, thus some integrity checks are performed
        # 3. eventually, the list of lists is converted into a well-shaped numpy array by filling up each row to the
        #    length of the maximal rows since some of them may have less subplots than the maximum one
        if isinstance(features, int):
            columns = [columns[i::features] for i in range(features)]
            orient_rows = not orient_rows
        else:
            for i, entry in enumerate(features):
                if isinstance(entry, str):
                    features[i] = fnmatch.filter(columns, entry)
                else:
                    for f in entry:
                        if f is not None:
                            assert f in columns, f"Feature '{f} is either not present in history or it is not numeric"
            columns = features
        num_rows = len(columns)
        num_cols = np.max([len(row) for row in columns])
        positions = np.array([[None] * num_cols] * num_rows)
        for i, row in enumerate(columns):
            positions[i, :len(row)] = row
        positions = positions if orient_rows else positions.transpose()
        # PLOT EACH COLUMN IN A SUBPLOT
        #   1. iterates by rows and columns
        #   2. if the column is not None, it plots the data
        ax = None
        num_rows, num_cols = positions.shape
        plt.figure(figsize=figsize, tight_layout=tight_layout, **plt_kwargs)
        for idx, column in enumerate(positions.flatten()):
            if column is not None:
                ax = plt.subplot(num_rows, num_cols, idx + 1, sharex=ax)
                x, y = self._history.index, self._history[column]
                plt.plot(x, y)
                plt.scatter(x, y)
                ax.set(xlabel='', ylabel='')
                ax.set_title(column, pad=titles_pad)
                # fix integer x_ticks based on autogenerated ones in order to avoid superimposed values
                ticks = np.unique(ax.get_xticks().round().astype(int))
                ax.set_xticks([t for t in ticks if t in range(x.min(), x.max() + 1)])
        if title is not None:
            plt.suptitle(title)
        plt.show()
