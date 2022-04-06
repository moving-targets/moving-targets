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
from moving_targets.masters.backends import GurobiBackend
from moving_targets.metrics import DIDI, CrossEntropy, Accuracy
# AS A FIRST STEP, WE NEED TO DEFINE OUR MASTER PROBLEM, WHICH IN THIS CASE WOULD BE THAT OF FAIR CLASSIFICATION
from moving_targets.util import probabilities


class FairClassification(ClassificationMaster):
    def __init__(self, protected, violation=0.2, backend='gurobi', loss='hd', alpha='harmonic', stats=False):
        # protected  : the name of the protected feature
        # violation  : the maximal accepted level of violation of the constraint.
        # backend    : the backend, which used to solve the master step
        # loss       : the loss function, which is used to compute the master objective
        # alpha      : the alpha optimizer, which is used to balance between the gradient term and the squared term
        # stats      : whether to include master statistics in the history object

        super().__init__(backend=backend, loss=loss, alpha=alpha, stats=stats, types='discrete', labelling=False)
        self.protected = protected
        self.violation = violation

    # here we define the problem formulation, i.e., variables and constraints
    def build(self, x, y, p):
        # retrieve model variables from the super method and, for compatibility between the binary/multiclass scenarios,
        # optionally transform 1d variables (representing a binary classification task) into a 2d matrix
        super_vars = super(FairClassification, self).build(x, y, p)
        variables = np.transpose([1 - super_vars, super_vars]) if super_vars.ndim == 1 else super_vars

        # as a first step, we need to compute the deviations between the average output for the total dataset and the
        # average output respectively to each protected class
        indicator_matrix = DIDI.get_indicator_matrix(x=x, protected=self.protected)
        groups, classes = len(indicator_matrix), len(np.unique(y))
        deviations = self.backend.add_continuous_variables(groups, classes, lb=0.0, name='deviations')
        # this is the average number of samples from the whole dataset <class[c]> as target class
        total_avg = [self.backend.mean(variables[:, c]) for c in range(classes)]
        for g, protected_group in enumerate(indicator_matrix):
            # this is the subset of the variables having <label> as protected feature (i.e., the protected group)
            protected_vars = variables[protected_group]
            if len(protected_vars) == 0:
                continue
            # this is the average number of samples within the protected group having <class[c]> as target
            protected_avg = [self.backend.mean(protected_vars[:, c]) for c in range(classes)]
            # eventually, the partial deviation is computed as the absolute value (which is linearized) of the
            # difference between the total average samples and the average samples within the protected group
            self.backend.add_constraints([deviations[g, c] >= total_avg[c] - protected_avg[c] for c in range(classes)])
            self.backend.add_constraints([deviations[g, c] >= protected_avg[c] - total_avg[c] for c in range(classes)])
        # finally, we compute the DIDI as the sum of this deviations, which is constrained to be lower or equal to the
        # given value (also, since we are computing the percentage DIDI, we need to scale for the original train_didi)
        didi = self.backend.sum(deviations)
        train_didi = DIDI.classification_didi(indicator_matrix=indicator_matrix, targets=y)
        self.backend.add_constraint(didi <= self.violation * train_didi)

        # return the original model variables at the end
        return super_vars


# THEN, WE MAY ADD A CUSTOM CALLBACK TO SEE HOW OUR TRAINING HAS PROCEEDED
class FairnessPlots(DataLogger):
    def __init__(self, protected, num_columns=4, **plt_kwargs):
        # protected   : the name of the protected feature
        # num_columns : the number of columns to display the subplots
        # plt_kwargs  : custom arguments to be passed to the 'matplotlib.pyplot.plot' function

        super().__init__()
        self.protected = protected
        self.num_columns = num_columns
        self.plt_kwargs = plt_kwargs

    def on_process_start(self, macs, x, y, val_data):
        # retrieve the subset of input features regarding the protected groups and store them in the inner 'data'
        # variable by replacing the input data which will not be useful during the plotting
        group = x[[c for c in x.columns if c.startswith(self.protected)]].values.squeeze().astype(int)
        self.data['group'] = group.argmax(axis=1) if group.ndim == 2 else group

    def on_training_end(self, macs, x, y, p, val_data):
        # store class targets instead of class probabilities
        super(FairnessPlots, self).on_training_end(macs, x, y, probabilities.get_classes(p), val_data)

    def on_process_end(self, macs, x, y, val_data):
        plt.figure(**self.plt_kwargs)
        num_rows = int(np.ceil(len(self.iterations) / self.num_columns))
        ax = None
        for it in self.iterations:
            ax = plt.subplot(num_rows, self.num_columns, it + 1, sharex=ax, sharey=ax)
            # we do not use count plot since we would like to show the percentage of predicted classes per group in
            # order to better see how moving targets affect the class balancing, thus we use instead a standard bar
            # plot to plot the bars with normalized value counts
            groups = self.data.rename(columns={f'p{it}': 'prediction'}).groupby('group')
            groups['prediction'].value_counts(normalize=True).mul(100).unstack().plot(kind='bar', stacked=True, ax=ax)
            ax.set(xlabel='group', ylabel='%')
            ax.set_title(f'iteration: {it}')
        plt.show()


if __name__ == '__main__':
    sns.set_style('whitegrid')
    sns.set_context('notebook')
    np.random.seed(0)

    # retrieve data and split train/test
    with importlib.resources.path('res', 'adult.csv') as filepath:
        df = pd.read_csv(filepath)
        x_df, y_df = df.drop('income', axis=1), df['income'].astype('category').cat.codes.values
        x_tr, x_ts, y_tr, y_ts = train_test_split(x_df, y_df, stratify=y_df, shuffle=True)

    # create a moving targets instance and fit it, then plot the training history
    # moreover, we pass a custom backend with a time limit
    cbs = [FairnessPlots(protected='race', figsize=(16, 9), tight_layout=True)]
    model = MACS(
        init_step='pretraining',
        learner=LogisticRegression(x_scaler='std', max_iter=10000),
        master=FairClassification(protected='race', backend=GurobiBackend(time_limit=10)),
        metrics=[Accuracy(), CrossEntropy(), DIDI(protected='race', classification=True, percentage=True)]
    )
    history = model.fit(x=x_tr, y=y_tr, iterations=10, val_data={'test': (x_ts, y_ts)}, callbacks=cbs, verbose=True)
    history.plot()
