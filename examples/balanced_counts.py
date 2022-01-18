import importlib.resources

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from moving_targets import MACS
from moving_targets.callbacks import Callback
from moving_targets.learners import LogisticRegression
from moving_targets.masters import ClassificationMaster
from moving_targets.metrics import Accuracy, ClassFrequenciesStd, CrossEntropy
from moving_targets.util import probabilities


# AS A FIRST STEP, WE NEED TO DEFINE OUR MASTER PROBLEM, WHICH IN THIS CASE WOULD BE THAT OF BALANCED COUNTS
class BalancedCounts(ClassificationMaster):
    def __init__(self, backend='gurobi', loss='mse', alpha=1, beta=1):
        # backend : the backend instance or backend alias
        # loss    : the loss function computed between the model variables and the learner predictions
        # alpha   : the non-negative real number which is used to calibrate the two losses in the alpha step
        # beta    : the non-negative real number which is used to constraint the p_loss in the beta step

        super().__init__(backend=backend, alpha=alpha, beta=beta, y_loss='hd', p_loss=loss)

    # here we define the problem formulation, i.e., variables and constraints
    def build(self, x, y):
        # retrieve model variables from the super method, which can be either:
        #   - a 1d vector of binary variables in case of binary classification (i.e., num classes = 2)
        #   - a 2d vector of binary variables in case of multiclass classification (i.e., num classes > 2)
        variables = super(BalancedCounts, self).build(x, y)

        # compute the upper bound for number of counts of a class, which will be used to constraint the model variables
        classes = np.unique(y)
        max_count = np.ceil(len(y) / len(classes))

        # constraint the model variables by computing the sum of each column
        # (the np.atleast_2d() call is used to handle binary classification tasks, where variables is a 1d vector)
        constraints = []
        for column in np.atleast_2d(variables).transpose():
            class_count = self.backend.sum(column)
            constraints.append(class_count <= max_count)
        self.backend.add_constraints(constraints=constraints)

        # return the variables at the end
        return variables

    # here we define whether to use the alpha or beta strategy, and beta is used if the constraint is already satisfied
    def use_beta(self, x, y, p):
        # the constraint is satisfied if all the classes counts are lower than then average number of counts per class
        pred = probabilities.get_classes(p, multi_label=False)
        classes, counts = np.unique(pred, return_counts=True)
        max_count = np.ceil(len(y) / len(classes))
        return np.all(counts <= max_count)


# THEN, WE MAY ADD A CUSTOM CALLBACK TO SEE HOW OUR TRAINING
class ClassesHistogram(Callback):
    def __init__(self, num_columns=4, **plt_kwargs):
        # num_columns : the number of columns to display the subplots
        # plt_kwargs  : custom arguments to be passed to the 'matplotlib.pyplot.plot' function
        # ------------------------------------------------------------------------------------
        # data        : a dataframe with the learner predictions and the adjusted targets
        # iterations  : a list of iteration names

        super().__init__()
        self.data = None
        self.iterations = []
        self.num_columns = num_columns
        self.plt_kwargs = plt_kwargs

    def on_process_start(self, macs, x, y, val_data):
        # initially, we store the original targets
        self.data = pd.DataFrame.from_dict({'y': y})

    def on_training_end(self, macs, x, y, val_data):
        # when the training ends, we use the macs instance to store both the predicted classes and the iterations
        self.data[f'pred_{macs.iteration}'] = probabilities.get_classes(macs.predict(x), multi_label=False)
        self.iterations.append(macs.iteration)

    def on_adjustment_end(self, macs, x, y, adjusted_y: np.ndarray, val_data):
        # we store the adjusted targets as well, but there is no need to store the iteration since it was already done
        self.data[f'adj_{macs.iteration}'] = adjusted_y

    def on_process_end(self, macs, val_data):
        # at the end of the process, we plot the results
        plt.figure(**self.plt_kwargs)
        num_rows = int(np.ceil(len(self.iterations) / self.num_columns))
        ax = None
        # for each iteration, we we plot the classes counts for both the predictions and the adjusted targets
        for idx, it in enumerate(self.iterations):
            ax = plt.subplot(num_rows, self.num_columns, idx + 1, sharex=ax, sharey=ax)
            # this check is necessary to handle the pretraining step, where no adjusted target is present
            if f'adj_{it}' in self.data:
                data = np.concatenate((self.data[f'pred_{it}'].values, self.data[f'adj_{it}'].values))
                hue = np.concatenate((len(self.data) * ['pred'], len(self.data) * ['adj']))
            else:
                data, hue = self.data[f'pred_{it}'].values, np.array(len(self.data) * ['pred'])
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
        learner=LogisticRegression(max_iter=10000),
        master=BalancedCounts(),
        metrics=[Accuracy(), CrossEntropy(), ClassFrequenciesStd()]
    )
    history = model.fit(x=x_tr, y=y_tr, iterations=10, val_data={'test': (x_ts, y_ts)}, callbacks=cbs, verbose=True)
    history.plot(figsize=(16, 9))
