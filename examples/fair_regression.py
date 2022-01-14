import importlib.resources

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from moving_targets import MACS
from moving_targets.learners import LinearRegression
from moving_targets.masters import SingleTargetRegression
from moving_targets.metrics import DIDI, R2, MSE


# AS A FIRST STEP, WE NEED TO DEFINE OUR MASTER PROBLEM, WHICH IN THIS CASE WOULD BE THAT OF FAIR CLASSIFICATION
class FairRegression(SingleTargetRegression):
    def __init__(self, protected, backend='gurobi', loss='mse', violation=0.2, lb=0, ub=float('inf'), alpha=1, beta=1):
        # protected  : the name of the protected feature
        # backend    : the backend instance or backend alias
        # loss       : both the y_loss and the p_loss
        # violation  : the maximal accepted level of violation of the constraint.
        # lb         : the model variables lower bounds.
        # ub         : the model variables upper bounds.
        # alpha      : the non-negative real number which is used to calibrate the two losses in the alpha step
        # beta       : the non-negative real number which is used to constraint the p_loss in the beta step
        # time_limit : the maximal time for which the master can run during each iteration.
        # ------------------------------------------------------------------------------------------------------
        # didi       : a DIDI metric instance used to compute both the indicator matrices and the satisfiability

        self.violation = violation
        self.didi = DIDI(protected=protected, classification=False, percentage=True)
        # the constraint is satisfied if the percentage DIDI is lower or equal to the expected violation; moreover,
        # since we know that the predictions must be positive, so we clip them to 0.0 in order to avoid (wrong)
        # negative predictions to influence the satisfiability computation
        super().__init__(satisfied=lambda x, y, p: self.didi(x=x, y=y, p=p.clip(0.0)) <= self.violation,
                         backend=backend, lb=lb, ub=ub, alpha=alpha, beta=beta, y_loss=loss, p_loss=loss)

    # here we define the problem formulation, i.e., variables and constraints
    def build(self, x, y, p):
        # retrieve model variables from the super method
        variables = super(FairRegression, self).build(x, y, p)

        # as a first step, we need to compute the deviations between the average output for the total dataset and the
        # average output respectively to each protected class
        indicator_matrix = DIDI.get_indicator_matrix(x=x, protected=self.didi.protected)
        deviations = self.backend.add_continuous_variables(len(indicator_matrix), lb=0.0, name='deviations')
        # this is the average output target for the whole dataset
        total_avg = self.backend.sum(variables) / len(variables)
        for g, protected_group in enumerate(indicator_matrix):
            # this is the subset of the variables having <label> as protected feature (i.e., the protected group)
            protected_vars = variables[protected_group]
            if len(protected_vars) == 0:
                continue
            # this is the average output target for the protected group
            protected_avg = self.backend.sum(protected_vars) / len(protected_vars)
            # eventually, the partial deviation is computed as the absolute value (which is linearized) of the
            # difference between the total average samples and the average samples within the protected group
            self.backend.add_constraint(deviations[g] >= total_avg - protected_avg)
            self.backend.add_constraint(deviations[g] >= protected_avg - total_avg)

        # finally, we compute the DIDI as the sum of this deviations, which is constrained to be lower or equal to the
        # given value (also, since we are computing the percentage DIDI, we need to scale for the original train_didi)
        didi = self.backend.sum(deviations)
        train_didi = DIDI.regression_didi(indicator_matrix=indicator_matrix, targets=y)
        self.backend.add_constraint(didi <= self.violation * train_didi)
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
        learner=LinearRegression(),
        master=FairRegression(protected='race'),
        metrics=[R2(), MSE(), DIDI(protected='race', classification=False, percentage=True)]
    )
    history = model.fit(x=x_tr, y=y_tr, iterations=10, val_data={'test': (x_ts, y_ts)}, verbose=True)
    history.plot(figsize=(16, 9))
