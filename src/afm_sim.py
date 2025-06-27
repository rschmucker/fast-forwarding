"""Student simulation implemented using AFM."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from src.util import read_file, prepare_kc_mapping, sigmoid
from src.util import evaluate_model

COLUMN_NAMES = ["name", "intercept", "slope"]


def logistic_regression(X, w):
    z = np.dot(X, w)
    return sigmoid(z)


def cost_function(w, X, y):  # BCE loss
    m = len(y)
    predictions = logistic_regression(X, w)

    # Clip predictions to avoid log(0)
    predictions = np.clip(predictions, 1e-10, 1 - 1e-10)

    cost_class1 = -y * np.log(predictions)
    cost_class0 = -(1 - y) * np.log(1 - predictions)
    cost = cost_class1 + cost_class0
    return cost.sum() / m


class AFMSim():
    """Implements student simulation using AFM."""

    def __init__(self, data_path: str, fit=False, verb=False) -> None:
        """Initialize new AFM simulator object

        Args:
            data_path (str): path to datashop dataset
            verbose (bool): show model fit statistics
        """
        self.data_path = data_path
        self.params = {}
        if fit:
            self.df = read_file(data_path)
            self.kc_to_id = prepare_kc_mapping(self.df)
            self.fit_model_man(verbose=verb)
        else:
            self.para_df = pd.read_csv(data_path)
            self.kc_to_id = {}
            self.params["bias"] = 0
            self.params["mean_student"] = 0
            self.params["std_student"] = 0
            self.params["kc_diffs"] = {}
            self.params["learn_params"] = {}
            i = 0
            for kc, intercept, slope in self.para_df[COLUMN_NAMES].values:
                assert kc not in self.kc_to_id, "found duplicate KC"
                self.kc_to_id[kc] = i
                self.params["kc_diffs"][i] = intercept
                self.params["learn_params"][i] = slope
                i += 1

        # for student knowledge state
        self.ability = self.sample_ability()
        self.attempts = {k: 0 for k in self.kc_to_id}

    def prepare_training_data(self):
        """Prepare data for AFM training.

        Returns:
            X (np.array): Features
            y (np.array): Labels
        """
        bias = np.ones(shape=(self.df.shape[0], 1))
        one_hots = np.zeros(shape=(self.df.shape[0], len(self.kc_to_id)))
        attempt_counts = np.zeros(shape=(self.df.shape[0], len(self.kc_to_id)))
        row = 0
        columns = ['KC (Default)', 'Opportunity (Default)']
        for kcs, counts in self.df[columns].values:
            for kc, count in zip(kcs.split("~~"), counts.split("~~")):
                one_hots[row][self.kc_to_id[kc]] = 1
                attempt_counts[row][self.kc_to_id[kc]] = int(count) - 1
            row += 1
        s_one_hot = pd.get_dummies(self.df["Anon Student Id"],
                                   columns=['Anon Student Id'])
        X = np.hstack([bias, one_hots, attempt_counts, s_one_hot.values])
        y = self.df['success']
        return X, y

    def fit_model_man(self, verbose=False):
        """Fit AFM model manually based on data file.

        Args:
            verbose (bool): Show model fit statistics
        """
        # fit model
        n_students = len(self.df["Anon Student Id"].unique())
        X, y = self.prepare_training_data()

        # Ensure the second half of the weight vector is non-negative
        constraint = (lambda w: w[1 + len(self.kc_to_id):-n_students])
        constr = {'type': 'ineq', 'fun': constraint}

        # optimize the model
        w_0 = np.random.normal(size=X.shape[1])
        print(w_0)
        sub_w = w_0[1 + len(self.kc_to_id):-n_students]
        w_0[1 + len(self.kc_to_id):-n_students] = np.random.random(len(sub_w))
        result = minimize(fun=cost_function, x0=w_0, args=(X, y),
                          constraints=constr)
        coefficients = result.x

        # extract relevant coefficients
        self.params["bias"] = coefficients[0]
        self.params["student_params"] = coefficients[-n_students:]
        self.params["kc_diffs"] = coefficients[1:1 + len(self.kc_to_id)]
        self.params["learn_params"] = \
            coefficients[1 + len(self.kc_to_id):-n_students]
        self.params["mean_student"] = np.mean(self.params["student_params"])
        self.params["std_student"] = np.std(self.params["student_params"])

        if verbose:
            print(self.kc_to_id)
            print("")
            for k in self.params:
                print(k)
                print(self.params[k])
                print("")

    def fit_model(self, verbose=False):
        """Fit AFM model based on data file.

        Args:
            verbose (bool): Show model fit statistics
        """
        # fit model
        X, y = self.prepare_training_data()
        self.model.fit(X, y)
        if verbose:
            print(evaluate_model(self.model, X, y))

        # extract relevant coefficients
        coefficients = self.model.coef_[0]
        n_students = len(self.df["Anon Student Id"].unique())
        self.params["bias"] = coefficients[0]
        self.params["student_params"] = coefficients[-n_students:]
        self.params["kc_diffs"] = coefficients[1:1 + len(self.kc_to_id)]
        self.params["learn_params"] = \
            coefficients[1 + len(self.kc_to_id):-n_students]
        self.params["mean_student"] = np.mean(self.params["student_params"])
        self.params["std_student"] = np.std(self.params["student_params"])

    def sample_ability(self):
        """Sample student ability parameter."""
        ability = np.random.normal(loc=self.params["mean_student"],
                                   scale=self.params["std_student"])
        return ability

    def reset_state(self):
        """Reset and initialize state before starting simulation."""
        self.ability = self.sample_ability()
        self.attempts = {k: 0 for k in self.kc_to_id}

    def get_state(self):
        """Get student knowledge state.

        Returns:
            masteries (dict): contains mastery of each individual KC
        """
        masteries = {}
        for k in self.kc_to_id:
            bias = self.params["bias"]
            alpha = self.ability
            diff = self.params["kc_diffs"][self.kc_to_id[k]]
            learn = self.params["learn_params"][self.kc_to_id[k]]
            atts = self.attempts[k]
            masteries[k] = sigmoid(bias + alpha + diff + (learn * atts))
        return masteries

    def simulate_response(self, kcs):
        """Simulate student response to particular problem.

        Args:
            kcs (list): kcs relevant for current problem/step

        Returns:
            p_cor (float): probability of correct response
            cor (int): sampled response correctness
        """
        agg_x = self.params["bias"] + self.ability
        for k in kcs:
            diff = self.params["kc_diffs"][self.kc_to_id[k]]
            learn = self.params["learn_params"][self.kc_to_id[k]]
            atts = self.attempts[k]
            agg_x += diff + (learn * atts)
        p_cor = sigmoid(agg_x)
        cor = np.random.binomial(1, p_cor)
        return p_cor, cor

    def update_state(self, kcs, cors):
        """Update student knowledge state.

        Args:
            kcs (list): name of attempted kcs
            cors (list): correctness for each kc (NOT USED BY AFM)
        """
        for k in kcs:
            self.attempts[k] += 1
