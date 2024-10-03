"""Basic Callback Interface."""

from typing import Optional

import numpy as np

from moving_targets.util.typing import Dataset


class Callback:
    """Basic interface for a Moving Targets callback."""

    def on_process_start(self, macs, x, y: np.ndarray, val_data: Optional[Dataset]):
        """Routine called at the end of the `MACS` fitting process.

        :param macs:
            Reference to the `MACS` object encapsulating the master.

        :param x:
            The matrix/dataframe of training samples.

        :param y:
            The vector of training labels.

        :param val_data:
            A dictionary containing the validation data, indicated as a tuple (xv, yv).
        """
        pass

    def on_process_end(self, macs, x, y: np.ndarray, val_data: Optional[Dataset]):
        """Routine called at the beginning of the `MACS` fitting process.

        :param macs:
            Reference to the `MACS` object encapsulating the master.

        :param x:
            The matrix/dataframe of training samples.

        :param y:
            The vector of training labels.

        :param val_data:
            A dictionary containing the validation data, indicated as a tuple (xv, yv).
        """
        pass

    def on_pretraining_start(self, macs, x, y: np.ndarray, val_data: Optional[Dataset]):
        """Routine called at the beginning of the `MACS` pretraining phase.

        :param macs:
            Reference to the `MACS` object encapsulating the master.

        :param x:
            The matrix/dataframe of training samples.

        :param y:
            The vector of training labels.

        :param val_data:
            A dictionary containing the validation data, indicated as a tuple (xv, yv).

        """
        self.on_iteration_start(macs=macs, x=x, y=y, val_data=val_data)
        self.on_training_start(macs=macs, x=x, y=y, val_data=val_data)

    def on_pretraining_end(self, macs, x, y: np.ndarray, p: np.ndarray, val_data: Optional[Dataset]):
        """Routine called at the end of the `MACS` pretraining phase.

        :param macs:
            Reference to the `MACS` object encapsulating the master.

        :param x:
            The matrix/dataframe of training samples.

        :param y:
            The vector of training labels.

        :param p:
            The vector of learners predictions.

        :param val_data:
            A dictionary containing the validation data, indicated as a tuple (xv, yv).

        """
        self.on_training_end(macs=macs, x=x, y=y, p=p, val_data=val_data)
        self.on_iteration_end(macs=macs, x=x, y=y, val_data=val_data)

    def on_iteration_start(self, macs, x, y: np.ndarray, val_data: Optional[Dataset]):
        """Routine called at the beginning of a `MACS` iteration.

        :param macs:
            Reference to the `MACS` object encapsulating the master.

        :param x:
            The matrix/dataframe of training samples.

        :param y:
            The vector of training labels.

        :param val_data:
            A dictionary containing the validation data, indicated as a tuple (xv, yv).

        """
        pass

    def on_iteration_end(self, macs, x, y: np.ndarray, val_data: Optional[Dataset]):
        """Routine called at the end of a `MACS` iteration.

        :param macs:
            Reference to the `MACS` object encapsulating the master.

        :param x:
            The matrix/dataframe of training samples.

        :param y:
            The vector of training labels.

        :param val_data:
            A dictionary containing the validation data, indicated as a tuple (xv, yv).

        """
        pass

    def on_training_start(self, macs, x, y: np.ndarray, val_data: Optional[Dataset]):
        """Routine called at the beginning of a `MACS` training phase.

        :param macs:
            Reference to the `MACS` object encapsulating the master.

        :param x:
            The matrix/dataframe of training samples.

        :param y:
            The vector of training labels.

        :param val_data:
            A dictionary containing the validation data, indicated as a tuple (xv, yv).

        """
        pass

    def on_training_end(self, macs, x, y: np.ndarray, p: np.ndarray, val_data: Optional[Dataset]):
        """Routine called at the end of a `MACS` training phase.

        :param macs:
            Reference to the `MACS` object encapsulating the master.

        :param x:
            The matrix/dataframe of training samples.

        :param y:
            The vector of training labels.

        :param p:
            The vector of learners predictions.

        :param val_data:
            A dictionary containing the validation data, indicated as a tuple (xv, yv).

        """
        pass

    def on_adjustment_start(self, macs, x, y: np.ndarray, val_data: Optional[Dataset]):
        """Routine called at the beginning of a `MACS` adjustment phase.

        :param macs:
            Reference to the `MACS` object encapsulating the master.

        :param x:
            The matrix/dataframe of training samples.

        :param y:
            The vector of training labels.

        :param val_data:
            A dictionary containing the validation data, indicated as a tuple (xv, yv).

        """
        pass

    def on_adjustment_end(self, macs, x, y: np.ndarray, z: np.ndarray, val_data: Optional[Dataset]):
        """Routine called at the end of a `MACS` adjustment phase.

        :param macs:
            Reference to the `MACS` object encapsulating the master.

        :param x:
            The matrix/dataframe of training samples.

        :param y:
            The vector of training labels.

        :param z:
            The vector of adjusted labels.

        :param val_data:
            A dictionary containing the validation data, indicated as a tuple (xv, yv).

        """
        pass
