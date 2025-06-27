"""Helper functions for student simulations."""
import copy
import random
import numpy as np
import pandas as pd
from pyBKT.models import Model
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

BASE_MODEL = "./artifacts/base_bkt.pkl"
BKT_PARAMS = ["Skill name", "P-Known", "P-Learn", "P-Guess", "P-Slip"]
DUMMY_KC = "DUMMY"


def set_random_seeds(seed):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)


def evaluate_model(model, X, y):
    """Evaluate model and compute key metrics.

    Args:
        model (sklearn model): model to evaluate
        X (np.array): data matrix
        y (np.array): binary target

    Returns
        evals (dict): dictionary with evaluation metrics
    """
    # Predict probabilities and classes
    y_probs = model.predict_proba(X)
    y_pred = model.predict(X)

    # Compute Accuracy
    accuracy = accuracy_score(y, y_pred)

    # Compute Log Loss (Negative Log Likelihood)
    loglikelihood = -log_loss(y, y_probs)

    # Compute AIC and BIC
    n = len(y)  # Number of observations
    k = X.shape[1]  # Number of features
    ll = loglikelihood * n  # Log likelihood needs to be scaled back
    aic = 2 * k - 2 * ll
    bic = np.log(n) * k - 2 * ll

    # Compute AUC
    auc = roc_auc_score(y, y_probs[:, 1])

    evals = {
        "accuracy": accuracy,
        "loglikelihood": loglikelihood,
        "aic": aic,
        "bic": bic,
        "auc": auc,
    }
    return evals


def read_file(data_path):
    """Read DataShop data file.

    Args:
        data_path (str): path to data file

    Returns:
        df (pd.DataFrame): loaded data file
    """
    df = pd.read_csv(data_path, sep="\t")
    df["success"] = \
        df["First Attempt"].apply(lambda x: 1 if x == "correct" else 0)
    df = df.dropna(subset=["Opportunity (Default)"])
    df = df.dropna(subset=["KC (Default)"])
    # NOTE: Meng & Robin decided against these stricter filters
    # df_s1 = df_s1[df_s1["First Attempt"].isin(["correct", "incorrect"])]
    # df.dropna(axis=0, how='any', inplace=True)
    return df


def prepare_kc_mapping(df):
    """Prepare KC mapping for data preproceessing.

    Args:
        df (pd.DataFrame): loaded DataShop data

    Returns
        kc_to_id (dict): mapping from KCs to integer IDs
    """
    kc_to_id, pointer = {}, 0
    for e in df['KC (Default)'].values:
        for kc in e.split("~~"):
            if kc not in kc_to_id:
                kc_to_id[kc] = pointer
                pointer += 1
    return kc_to_id


def sigmoid(x):
    """Compute sigmoid function value."""
    return 1 / (1 + np.exp(-x))


def init_BKT_models(param_path: str, seed=0):
    """Initialize BKT model for each skill.

    Args:
        param_path (str): path to prepared parameters

    Returns:
        bkt_models (dict): dictionary with BKT model for each KC
    """
    bkt_models = {}
    params = pd.read_csv(param_path)
    for kc, p_known, p_learn, p_guess, p_slip in params[BKT_PARAMS].values:
        model = Model(seed=seed)
        model.load(BASE_MODEL)
        assert len(list(model.coef_.keys())) == 1, "one BKT per skill"
        assert DUMMY_KC == list(model.coef_.keys())[0], \
            "Assert name of single KC is: " + BASE_MODEL
        assert 0 <= p_known <= 1, "Invalid p_known value: " + str(p_known)
        assert 0 <= p_learn <= 1, "Invalid p_learn value: " + str(p_learn)
        assert 0 <= p_guess <= 1, "Invalid p_guess value: " + str(p_guess)
        assert 0 <= p_slip <= 1, "Invalid p_slip value: " + str(p_slip)
        c_fit_m = copy.deepcopy(model.fit_model)

        # overwrite coef first
        model.coef_ = {DUMMY_KC: {'prior': p_known,
                                  'learns': np.array([p_learn]),
                                  'guesses': np.array([p_guess]),
                                  'slips': np.array([p_slip]),
                                  'forgets': np.array([0.0])}}

        # modify fit_model object
        c_fit_m[DUMMY_KC]["prior"] = p_known
        c_fit_m[DUMMY_KC]["learns"] = np.array([p_learn])
        c_fit_m[DUMMY_KC]["forgets"] = np.array([0.0])
        c_fit_m[DUMMY_KC]["guesses"] = np.array([p_guess])
        c_fit_m[DUMMY_KC]["slips"] = np.array([p_slip])
        c_fit_m[DUMMY_KC]["As"] = np.array([[(1 - p_learn), 0],
                                            [p_learn, 1]])
        c_fit_m[DUMMY_KC]["emissions"] = \
            np.array([[(1 - p_guess), p_guess], [p_slip, (1 - p_slip)]])
        c_fit_m[DUMMY_KC]["pi_0"] = np.array([[(1 - p_known)], [p_known]])
        model.fit_model = c_fit_m
        bkt_models[kc] = model
    return bkt_models
