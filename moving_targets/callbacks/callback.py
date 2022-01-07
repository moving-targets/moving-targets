"""Basic Callback Interface."""

from typing import Optional

from moving_targets.util.typing import Dataset


class Callback:
    """Basic interface for a Moving Targets callback."""

    def __init__(self):
        """"""
        super(Callback, self).__init__()

    def on_process_start(self, macs, x, y, val_data: Optional[Dataset], **additional_kwargs):
        """Routine called at the end of the `MACS` fitting process.

        :param macs:
            Reference to the `MACS` object encapsulating the master.

        :param x:
            The matrix/dataframe of training samples.

        :param y:
            The vector of training labels.

        :param val_data:
            A dictionary containing the validation data, indicated as a tuple (xv, yv).

        :param additional_kwargs:
            Additional arguments which are implementation-dependent.
        """
        pass

    def on_process_end(self, macs, val_data: Optional[Dataset], **additional_kwargs):
        """Routine called at the beginning of the `MACS` fitting process.

        :param macs:
            Reference to the `MACS` object encapsulating the master.

        :param val_data:
            A dictionary containing the validation data, indicated as a tuple (xv, yv).

        :param additional_kwargs:
            Additional arguments which are implementation-dependent.
        """
        pass

    def on_pretraining_start(self, macs, x, y, val_data: Optional[Dataset], **additional_kwargs):
        """Routine called at the beginning of the `MACS` pretraining phase.

        :param macs:
            Reference to the `MACS` object encapsulating the master.

        :param x:
            The matrix/dataframe of training samples.

        :param y:
            The vector of training labels.

        :param val_data:
            A dictionary containing the validation data, indicated as a tuple (xv, yv).

        :param additional_kwargs:
            Additional arguments which are implementation-dependent.
        """
        self.on_iteration_start(macs=macs, x=x, y=y, val_data=val_data, **additional_kwargs)
        self.on_training_start(macs=macs, x=x, y=y, val_data=val_data, **additional_kwargs)

    def on_pretraining_end(self, macs, x, y, val_data: Optional[Dataset], **additional_kwargs):
        """Routine called at the end of the `MACS` pretraining phase.

        :param macs:
            Reference to the `MACS` object encapsulating the master.

        :param x:
            The matrix/dataframe of training samples.

        :param y:
            The vector of training labels.

        :param val_data:
            A dictionary containing the validation data, indicated as a tuple (xv, yv).

        :param additional_kwargs:
            Additional arguments which are implementation-dependent.
        """
        self.on_training_end(macs=macs, x=x, y=y, val_data=val_data, **additional_kwargs)
        self.on_iteration_end(macs=macs, x=x, y=y, val_data=val_data, **additional_kwargs)

    def on_iteration_start(self, macs, x, y, val_data: Optional[Dataset], **additional_kwargs):
        """Routine called at the beginning of a `MACS` iteration.

        :param macs:
            Reference to the `MACS` object encapsulating the master.

        :param x:
            The matrix/dataframe of training samples.

        :param y:
            The vector of training labels.

        :param val_data:
            A dictionary containing the validation data, indicated as a tuple (xv, yv).

        :param additional_kwargs:
            Additional arguments which are implementation-dependent.
        """
        pass

    def on_iteration_end(self, macs, x, y, val_data: Optional[Dataset], **additional_kwargs):
        """Routine called at the end of a `MACS` iteration.

        :param macs:
            Reference to the `MACS` object encapsulating the master.

        :param x:
            The matrix/dataframe of training samples.

        :param y:
            The vector of training labels.

        :param val_data:
            A dictionary containing the validation data, indicated as a tuple (xv, yv).

        :param additional_kwargs:
            Additional arguments which are implementation-dependent.
        """
        pass

    def on_training_start(self, macs, x, y, val_data: Optional[Dataset], **additional_kwargs):
        """Routine called at the beginning of a `MACS` training phase.

        :param macs:
            Reference to the `MACS` object encapsulating the master.

        :param x:
            The matrix/dataframe of training samples.

        :param y:
            The vector of training labels.

        :param val_data:
            A dictionary containing the validation data, indicated as a tuple (xv, yv).

        :param additional_kwargs:
            Additional arguments which are implementation-dependent.
        """
        pass

    def on_training_end(self, macs, x, y, val_data: Optional[Dataset], **additional_kwargs):
        """Routine called at the end of a `MACS` training phase.

        :param macs:
            Reference to the `MACS` object encapsulating the master.

        :param x:
            The matrix/dataframe of training samples.

        :param y:
            The vector of training labels.

        :param val_data:
            A dictionary containing the validation data, indicated as a tuple (xv, yv).

        :param additional_kwargs:
            Additional arguments which are implementation-dependent.
        """
        pass

    def on_adjustment_start(self, macs, x, y, val_data: Optional[Dataset], **additional_kwargs):
        """Routine called at the beginning of a `MACS` adjustment phase.

        :param macs:
            Reference to the `MACS` object encapsulating the master.

        :param x:
            The matrix/dataframe of training samples.

        :param y:
            The vector of training labels.

        :param val_data:
            A dictionary containing the validation data, indicated as a tuple (xv, yv).

        :param additional_kwargs:
            Additional arguments which are implementation-dependent.
        """
        pass

    def on_adjustment_end(self, macs, x, y, adjusted_y, val_data: Optional[Dataset], **additional_kwargs):
        """Routine called at the end of a `MACS` adjustment phase.

        :param macs:
            Reference to the `MACS` object encapsulating the master.

        :param x:
            The matrix/dataframe of training samples.

        :param y:
            The vector of training labels.

        :param adjusted_y:
            The vector of adjusted labels.

        :param val_data:
            A dictionary containing the validation data, indicated as a tuple (xv, yv).

        :param additional_kwargs:
            Additional arguments which are implementation-dependent.
        """
        pass
