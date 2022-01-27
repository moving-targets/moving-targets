"""Custom scalers."""

from typing import Union, Dict, List, Optional, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer


class Scaler:
    """Custom scaler that is able to treat each feature separately."""

    def __init__(self, *methods: str, default_method: str = 'none', **custom_methods: str):
        """
        :param methods:
            If a single value is passed, all the features are scaled with the same method, otherwise, if a list or a
            dictionary is passed, each feature is scaled with the respective method.

        :param default_method:
            The default scaling which is used for features for which an explicit method is not passed.

            Admitted methods are (1) 'std' to standardize the data, (2) 'norm' to normalize the data, (3) 'onehot' to
            onehot encode categorical features, and (4) 'none' to perform no scaling.
        """
        super(Scaler, self).__init__()
        assert default_method == 'none' or len(methods) == 0, "if the default method is not None, custom scaling " \
                                                              "methods for specific features must be passed via " \
                                                              "keyword (i.e., the '*methods' parameter must be empty)"
        assert len(custom_methods) == 0 or len(methods) <= 1, "if at least one custom scaling method for specific " \
                                                              "features is passed, no unnamed method must be passed " \
                                                              "as well (i.e., the '*methods' parameter must be empty)"

        # if a single method is passed (and default_method is known to be None), then we consider it the default one
        self.default_method: str = methods[0] if len(methods) == 1 else default_method
        """The default scaling method."""

        # if more than a single method is passed we use them, otherwise we use the custom_methods dictionary
        self.methods: Union[List[str], Dict[str, str]] = list(methods) if len(methods) > 1 else custom_methods
        """The custom scaling methods."""

        self._dtypes: Any = None
        """The input data types."""

        self.input_columns: List[Any] = []
        """The list of input column names."""

        self.output_columns: List[Any] = []
        """The list of output column names (this is useful for onehot encoding)."""

        self._onehot: Dict[Any, LabelBinarizer] = {}
        """The dictionary of onehot encoders indexed by column name/number."""

        self._translation: Optional[np.ndarray] = None
        """The translation vector."""

        self._scaling: Optional[np.ndarray] = None
        """The scaling vector."""

    def fit(self, data) -> Any:
        """Fits the scaler parameters.

        :param data:
            The matrix/dataframe of samples.

        :return:
            The scaler itself.
        """
        # handle input, then clear and fit metadata
        self.output_columns = []
        self.input_columns = []
        self._onehot = {}
        self._dtypes, data = (data.dtypes, data) if isinstance(data, pd.DataFrame) else (data.dtype, pd.DataFrame(data))

        # handle methods:
        # > if self.methods is a list, pair the methods with the columns
        # > otherwise, use the custom methods when passed, or the default method otherwise
        if isinstance(self.methods, list):
            assert len(self.methods) == len(data.columns), "previously passed methods do not match data features length"
            methods = {c: m for m, c in zip(self.methods, data.columns)}
        else:
            methods = {c: self.methods.get(c) or self.default_method for c in data.columns}

        # compute factors
        translation, scaling = [], []
        self.input_columns = list(data.columns)
        for column, method in methods.items():
            if method == 'onehot':
                encoder = LabelBinarizer()
                encoder.fit(list(data[column]))
                if encoder.y_type_ == 'binary':
                    self.output_columns.append(f'{column}_{encoder.classes_[1]}')
                    translation.append(0)
                    scaling.append(1)
                else:
                    self.output_columns += [f'{column}_{c}' for c in encoder.classes_]
                    translation += [0] * len(encoder.classes_)
                    scaling += [1] * len(encoder.classes_)
                self._onehot[column] = encoder
            else:
                self.output_columns.append(column)
                values = data[column].astype(np.float64).values
                if method in ['std', 'standardize']:
                    translation.append(values.mean())
                    scaling.append(values.std())
                elif method in ['norm', 'normalize', 'minmax']:
                    translation.append(values.min())
                    scaling.append(values.max() - values.min())
                elif method == 'none':
                    translation.append(0)
                    scaling.append(1)
                else:
                    raise ValueError(f'Method {method} is not supported')

        # handle case with null scaling factor to avoid division by zero
        self._scaling = np.where([s != 0 for s in scaling], scaling, 1.0).astype(np.float64)
        self._translation = np.array(translation).astype(np.float64)
        return self

    def transform(self, data) -> Any:
        """Transforms the data according to the scaler parameters.

        :param data:
            The matrix/dataframe of samples.

        :return:
            The scaled data.
        """
        is_1d, is_numpy = data.ndim == 1, True if isinstance(data, np.ndarray) else False
        data = pd.DataFrame(data, columns=self.input_columns)
        # handle one hot encoding
        columns, dataframes = self.input_columns.copy(), [data]
        for column, encoder in self._onehot.items():
            # define class names
            if encoder.y_type_ == 'binary':
                class_names = [f'{column}_{encoder.classes_[1]}']
            else:
                class_names = [f'{column}_{c}' for c in encoder.classes_]
            # retrieve column index and replace it with the list of class names
            index = columns.index(column)
            columns = columns[:index] + class_names + ([] if len(columns) == index + 1 else columns[index + 1:])
            # compute onehot classes, create dataframe, and append it to the list
            classes = encoder.transform(list(data[column]))
            classes = pd.DataFrame(classes, index=data.index, columns=class_names)
            dataframes.append(classes)
        # build the full dataframe by appending the classes at the end and reordering the columns, then scale it
        data = pd.concat(dataframes, axis=1)[columns]
        data = (data.astype(np.float64) - self._translation) / self._scaling
        data = data.squeeze() if is_1d else data
        return data.values if is_numpy else data

    def fit_transform(self, data) -> Any:
        """Fits the scaler parameters.

        :param data:
            The matrix/dataframe of samples.

        :return:
            The scaled data.
        """
        return self.fit(data).transform(data)

    def inverse_transform(self, data) -> Any:
        """Inverts the scaling according to the scaler parameters.

        :param data:
            The previously scaled matrix/dataframe of samples.

        :return:
            The original data.
        """
        is_1d, is_numpy = data.ndim == 1, True if isinstance(data, np.ndarray) else False
        data = pd.DataFrame(data, columns=self.output_columns)
        data = (data.astype(np.float64) * self._scaling) + self._translation
        # handle one hot encoding
        columns = self.output_columns.copy()
        for column, encoder in self._onehot.items():
            # define class names
            if encoder.y_type_ == 'binary':
                class_names = [f'{column}_{encoder.classes_[1]}']
            else:
                class_names = [f'{column}_{c}' for c in encoder.classes_]
            # remove onehot class names from columns and add column name instead
            first_index = columns.index(class_names[0])
            last_index = first_index + len(class_names)
            columns = columns[:first_index] + [column] + columns[last_index:]
            # add new column in the dataframe for the categorical feature
            data[column] = encoder.inverse_transform(data[class_names].values)
        # select only the correct columns and assign dtypes
        data = data[columns].astype(self._dtypes)
        data = data.squeeze() if is_1d else data
        return data.values if is_numpy else data
