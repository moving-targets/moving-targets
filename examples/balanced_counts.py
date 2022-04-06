import importlib.resources

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from moving_targets import MACS
from moving_targets.callbacks import DataLogger
from moving_targets.learners import LogisticRegression
from moving_targets.masters import ClassificationMaster
from moving_targets.metrics import Accuracy, ClassFrequenciesStd, CrossEntropy
from moving_targets.util import probabilities


# AS A FIRST STEP, WE NEED TO DEFINE OUR MASTER PROBLEM, WHICH IN THIS CASE WOULD BE THAT OF BALANCED COUNTS
class BalancedCounts(ClassificationMaster):
    def __init__(self, backend='gurobi', loss='hd', alpha='harmonic', stats=False):
        # backend : the backend, which used to solve the master step
        # loss    : the loss function, which is used to compute the master objective
        # alpha   : the alpha optimizer, which is used to balance between the gradient term and the squared term
        # stats   : whether to include master statistics in the history object

        super().__init__(backend=backend, loss=loss, alpha=alpha, stats=stats, types='discrete', labelling=False)

    # here we define the problem formulation, i.e., variables and constraints
    def build(self, x, y, p):
        # retrieve model variables from the super method, which can be either:
        #   - a 1d vector of binary variables in case of binary classification (i.e., num classes = 2)
        #   - a 2d vector of binary variables in case of multiclass classification (i.e., num classes > 2)
        # then reshape it into a column vector to better handle both cases
        variables = super(BalancedCounts, self).build(x, y, p).reshape((len(y), -1))

        # use the number of samples and classes to compute the upper bound for the number of counts of a class then
        # constraint the sum of the model variables on each column to be lower than that value
        num_samples, num_classes = (len(y), 2) if y.ndim == 1 else y.shape
        max_count = np.ceil(num_samples / num_classes)
        self.backend.add_constraints([c <= max_count for c in self.backend.sum(variables, axis=0)])

        # return the variables at the end after reshaping them into their default shape (which is the same as y)
        return variables.reshape(y.shape)


# THEN, WE MAY ADD A CUSTOM CALLBACK TO SEE HOW OUR TRAINING HAS PROCEEDED
class ClassesHistogram(DataLogger):
    def __init__(self, num_columns=4, **plt_kwargs):
        # num_columns : the number of columns to display the subplots
        # plt_kwargs  : custom arguments to be passed to the 'matplotlib.pyplot.plot' function

        super().__init__()
        self.num_columns = num_columns
        self.plt_kwargs = plt_kwargs

    def on_training_end(self, macs, x, y, p, val_data):
        # store class targets instead of class probabilities
        super(ClassesHistogram, self).on_training_end(macs, x, y, probabilities.get_classes(p), val_data)

    def on_process_end(self, macs, x, y, val_data):
        # at the end of the process, we plot the results
        plt.figure(**self.plt_kwargs)
        num_rows = int(np.ceil(len(self.iterations) / self.num_columns))
        ax = None
        # for each iteration, we we plot the classes counts for both the predictions and the adjusted targets
        for it in self.iterations:
            ax = plt.subplot(num_rows, self.num_columns, it + 1, sharex=ax, sharey=ax)
            # this check is necessary to handle the pretraining step, where no adjusted target is present
            column, name = ('y', 'targets') if it == 0 else (f'z{it}', 'adjusted')
            data = np.concatenate((self.data[column].values, self.data[f'p{it}'].values))
            hue = np.concatenate((len(self.data) * [name], len(self.data) * ['predictions']))
            sns.countplot(x=data, hue=hue, ax=ax)
            ax.set(xlabel='class', ylabel='count')
            ax.set_title(f'iteration: {it}')
        plt.show()


# FINALLY, WE CAN CREATE OUR SCRIPT TO TEST THE CORRECTNESS OF THE MODEL
if __name__ == '__main__':
    np.random.seed(0)
    sns.set_style('whitegrid')
    sns.set_context('notebook')

    # retrieve data and split train/test
    with importlib.resources.path('res', 'redwine.csv') as filepath:
        df = pd.read_csv(filepath)
        x_df, y_df = df.drop('quality', axis=1), df['quality'].astype('category').cat.codes.values
        x_tr, x_ts, y_tr, y_ts = train_test_split(x_df, y_df, stratify=y_df, shuffle=True)

    # create a moving targets instance and fit it, then plot the training history
    cbs = [ClassesHistogram(figsize=(16, 9), tight_layout=True)]
    model = MACS(
        init_step='pretraining',
        learner=LogisticRegression(x_scaler='std', max_iter=10000),
        master=BalancedCounts(),
        metrics=[Accuracy(), CrossEntropy(), ClassFrequenciesStd()]
    )
    history = model.fit(x=x_tr, y=y_tr, iterations=10, val_data={'test': (x_ts, y_ts)}, callbacks=cbs, verbose=True)
    history.plot()
