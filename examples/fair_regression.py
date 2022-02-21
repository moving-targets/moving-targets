import importlib.resources

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from moving_targets import MACS
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
        # ------------------------------------------------------------------------------------------------------
        # didi       : a DIDI metric instance used to compute both the indicator matrices and the satisfiability

        super().__init__(backend=backend, loss=loss, alpha=alpha, lb=lb, ub=ub, stats=stats)
        self.didi = DIDI(protected=protected, classification=False, percentage=True)
        self.violation = violation

    # here we define the problem formulation, i.e., variables and constraints
    def build(self, x, y, p):
        # retrieve model variables from the super method
        variables = super(FairRegression, self).build(x, y, p)

        # as a first step, we need to compute the deviations between the average output for the total dataset and the
        # average output respectively to each protected class
        indicator_matrix = DIDI.get_indicator_matrix(x=x, protected=self.didi.protected)
        deviations = self.backend.add_continuous_variables(len(indicator_matrix), lb=0.0, name='deviations')
        # this is the average output target for the whole dataset
        total_avg = self.backend.mean(variables)
        for g, protected_group in enumerate(indicator_matrix):
            # this is the subset of the variables having <label> as protected feature (i.e., the protected group)
            protected_vars = variables[protected_group]
            if len(protected_vars) > 0:
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
    model = MACS(
        init_step='pretraining',
        learner=LinearRegression(x_scaler='std', y_scaler='norm'),
        master=FairRegression(protected='race'),
        metrics=[R2(), MSE(), DIDI(protected='race', classification=False, percentage=True)]
    )
    history = model.fit(x=x_tr, y=y_tr, iterations=10, val_data={'test': (x_ts, y_ts)}, verbose=True)
    history.plot()
