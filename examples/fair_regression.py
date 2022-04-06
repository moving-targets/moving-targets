import importlib.resources

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from moving_targets import MACS
from moving_targets.callbacks import DataLogger
from moving_targets.learners import LinearRegression
from moving_targets.masters import RegressionMaster
from moving_targets.metrics import DIDI, R2, MSE


# AS A FIRST STEP, WE NEED TO DEFINE OUR MASTER PROBLEM, WHICH IN THIS CASE WOULD BE THAT OF FAIR CLASSIFICATION
class FairRegression(RegressionMaster):
    def __init__(self, protected, violation=0.2, backend='gurobi', loss='mse',
                 alpha='harmonic', lb=0, ub=float('inf'), stats=False):
        # protected  : the name of the protected feature
        # violation  : the maximal accepted level of violation of the constraint.
        # backend    : the backend, which used to solve the master step
        # loss       : the loss function, which is used to compute the master objective
        # alpha      : the alpha optimizer, which is used to balance between the gradient term and the squared term
        # lb         : the model variables lower bounds
        # ub         : the model variables upper bounds
        # stats      : whether to include master statistics in the history object

        super().__init__(backend=backend, loss=loss, alpha=alpha, lb=lb, ub=ub, stats=stats)
        self.protected = protected
        self.violation = violation

    # here we define the problem formulation, i.e., variables and constraints
    def build(self, x, y, p):
        # retrieve model variables from the super method
        variables = super(FairRegression, self).build(x, y, p)

        # as a first step, we need to compute the deviations between the average output for the total dataset and the
        # average output respectively to each protected class
        indicator_matrix = DIDI.get_indicator_matrix(x=x, protected=self.protected)
        deviations = self.backend.add_continuous_variables(len(indicator_matrix), lb=0.0, name='deviations')
        # this is the average output target for the whole dataset
        total_avg = self.backend.mean(variables)
        for g, protected_group in enumerate(indicator_matrix):
            # this is the subset of the variables having <label> as protected feature (i.e., the protected group)
            protected_vars = variables[protected_group]
            if len(protected_vars) == 0:
                continue
            # this is the average output target for the protected group
            protected_avg = self.backend.mean(protected_vars)
            # eventually, the partial deviation is computed as the absolute value (which is linearized) of the
            # difference between the total average samples and the average samples within the protected group
            self.backend.add_constraint(deviations[g] >= total_avg - protected_avg)
            self.backend.add_constraint(deviations[g] >= protected_avg - total_avg)

        # finally, we compute the DIDI as the sum of this deviations, which is constrained to be lower or equal to the
        # given value (also, since we are computing the percentage DIDI, we need to scale for the original train_didi)
        didi = self.backend.sum(deviations)
        train_didi = DIDI.regression_didi(indicator_matrix=indicator_matrix, targets=y)
        self.backend.add_constraint(didi <= self.violation * train_didi)

        # return the variables at the end
        return variables


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
        # retrieve the subset of input features regarding the protected groups (in case of multiple groups, the index
        # must be obtained via argmax) and store them in the inner 'data' variable
        super(FairnessPlots, self).on_process_start(macs, x, y, val_data)
        group = x[[c for c in x.columns if c.startswith(self.protected)]].values.squeeze().astype(int)
        self.data['group'] = group.argmax(axis=1) if group.ndim == 2 else group

    def on_process_end(self, macs, x, y, val_data):
        plt.figure(**self.plt_kwargs)
        num_rows = int(np.ceil(len(self.iterations) / self.num_columns))
        ax = None
        for it in self.iterations:
            ax = plt.subplot(num_rows, self.num_columns, it + 1, sharex=ax, sharey=ax)
            # this check is necessary to handle the pretraining step, where no adjusted target is present
            column, name = ('y', 'targets') if it == 0 else (f'z{it}', 'adjusted')
            data = pd.DataFrame.from_dict({
                'group': np.concatenate((self.data['group'].values, self.data['group'].values)),
                'targets': np.concatenate((self.data[column].values, self.data[f'p{it}'].values)),
                'hue': np.concatenate((len(self.data) * [name], len(self.data) * ['predictions']))
            })
            sns.boxplot(data=data, x='group', y='targets', hue='hue', ax=ax)
            ax.set_title(f'iteration: {it}')
        plt.show()


if __name__ == '__main__':
    sns.set_style('whitegrid')
    sns.set_context('notebook')
    np.random.seed(0)

    # retrieve data and split train/test
    # (we keep the input data as dataframes instead of arrays to compute the indicator matrix from the data columns)
    with importlib.resources.path('res', 'communities.csv') as filepath:
        df = pd.read_csv(filepath)
        x_df, y_df = df.drop('violentPerPop', axis=1), df['violentPerPop'].values
        x_tr, x_ts, y_tr, y_ts = train_test_split(x_df, y_df, shuffle=True)

    # create a moving targets instance and fit it, then plot the training history
    cbs = [FairnessPlots(protected='race', figsize=(16, 9), tight_layout=True)]
    model = MACS(
        init_step='pretraining',
        learner=LinearRegression(x_scaler='std', y_scaler='norm'),
        master=FairRegression(protected='race'),
        metrics=[R2(), MSE(), DIDI(protected='race', classification=False, percentage=True)]
    )
    history = model.fit(x=x_tr, y=y_tr, iterations=10, val_data={'test': (x_ts, y_ts)}, callbacks=cbs, verbose=True)
    history.plot()
