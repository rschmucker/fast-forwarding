"""Student knowledge state inference implemented using BKT."""
import numpy as np
import pandas as pd
from src.util import init_BKT_models, DUMMY_KC


class BKTInference():
    """Implements student simulation using BKT."""

    def __init__(self, para_path: str) -> None:
        """Initialize new BKT inference object

        Args:
            para_path (str): path to BKT parameters
        """
        self.data_path = para_path
        para_df = pd.read_csv(para_path)
        self.paras = {}
        for i in range(para_df.shape[0]):
            self.paras[para_df["Skill name"].values[i]] = {
                "p_known": para_df["P-Known"].values[i],
                "p_learn": para_df["P-Learn"].values[i],
                "p_guess": para_df["P-Guess"].values[i],
                "p_slip": para_df["P-Slip"].values[i],
            }

        # for student knowledge state
        self.attempts = {k: [] for k in self.paras}

    def reset_state(self):
        """Reset and initialize state before starting simulation."""
        self.attempts = {k: [] for k in self.paras}

    def prepare_bkt_df(self, kc):
        """Prepare dataframe for BKT predictions.

        Args:
            kc (str): name of KC to focus on

        Returns:
            kc_df (pd.DataFrame): KC data reformated as DataFrame
        """
        kc_df = pd.DataFrame()
        kc_df["Anon Student Id"] = [0] * (len(self.attempts[kc]) + 1)
        kc_df["Row"] = np.arange(kc_df.shape[0])
        kc_df["KC(Default)"] = [DUMMY_KC] * kc_df.shape[0]
        kc_df["Correct First Attempt"] = self.attempts[kc] + [0]
        return kc_df

    def manual_bkt(self, kc, cors):
        """Compute mastery estimates based on manual implementation.

        Args:
            kc (str): name of kc
            cors (list): list of response correctness for particular KC
        """
        p_known, p_learn = self.paras[kc]["p_known"], self.paras[kc]["p_learn"]
        p_guess, p_slip = self.paras[kc]["p_guess"], self.paras[kc]["p_slip"]
        ms = [p_known]
        for c in cors:
            m_ref = ms[-1]
            if c == 1:  # correct
                numerator = (1 - p_slip) * m_ref
                denominator = ((1 - p_slip) * m_ref) + (p_guess * (1 - m_ref))
                cond = numerator / denominator
            else:  # model incorrect
                numerator = p_slip * m_ref
                denominator = (p_slip * m_ref) + ((1 - p_guess) * (1 - m_ref))
                cond = numerator / denominator
            m_t = cond + (p_learn * (1 - cond))
            ms.append(m_t)
        return ms

    def get_state(self):
        """Get student knowledge state.

        Returns:
            masteries (dict): contains mastery of each individual KC
        """
        masteries = {}
        for kc in self.paras:
            # kc_df = self.prepare_bkt_df(kc)
            # estimate mastery
            # pred = self.bkts[kc].predict(data=kc_df)
            # masteries[kc] = pred["state_predictions"].values[-1]
            masteries[kc] = self.manual_bkt(kc, self.attempts[kc])[-1]
            # man_pred = self.manual_bkt(kc)
            # assert abs(masteries[kc] - man_pred) < 0.01, "prediction miss"
        return masteries

    def update_state(self, kcs, cors):
        """Update student knowledge state.

        Args:
            kcs (list): name of attempted kcs
            cors (list): correctness for each kc (NOT USED BY AFM)
        """
        for i, k in enumerate(kcs):
            self.attempts[k].append(cors[i])
