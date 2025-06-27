"""Focus scheduler that selects KC random and problem at random."""
import random
import numpy as np
from src.problem_pool import ProblemPool


class FocusRndScheduler():
    """Implements focus scheduler with random KC."""

    def __init__(self, problem_path, threshold=0.95) -> None:
        """Initializes new focus scheduler with random KC.

        Args:
            problem_path (str): path to problem file
            threshold (float): mastery threshold
        """
        self.problem_pool = ProblemPool(problem_path)
        self.threshold = threshold

    def select_problem(self, kc_estimates):
        """Select next problem using focus with random kc.

        Args:
            kc_estimates (dict): Mastery estimates

        Returns:
            problem (list): Next problem as step list
        """
        a_p_ids = self.problem_pool.get_available_problems()
        if len(a_p_ids) == 0:  # exhausted problem pool
            return -1, -1

        # select KC at random based proportional to p_unknown
        kcs, unmastered_vals = [], []
        for kc in kc_estimates:
            has_problem = False
            for p in a_p_ids:
                if np.any([kc in s for s in self.problem_pool.problems[p]]):
                    has_problem = True
            if kc_estimates[kc] >= self.threshold:  # mastered
                continue
            elif not has_problem:  # no problem available
                continue
            else:  # consider this KC
                kcs.append(kc)
                unmastered_vals.append(1 - kc_estimates[kc])
        if len(kcs) == 0:  # student mastered everything OR no problems left
            # TODO: Implement early stopping
            p_id = random.choice(a_p_ids)
            self.problem_pool.mark_problem(p_id)
            return p_id, self.problem_pool.problems[p_id]

        kc_probs = [v / np.sum(unmastered_vals) for v in unmastered_vals]
        kc_sel = np.random.choice(kcs, p=kc_probs)

        # select problem based on focus score
        focus_scores, ids = [], []
        for p in a_p_ids:
            rel = np.any([kc_sel in s for s in self.problem_pool.problems[p]])
            if not rel:
                continue  # this p is not relevant 
            p_unknown, hardness = 0, 0
            for step in self.problem_pool.problems[p]:
                if kc_sel in step:
                    p_unknown += (1 - kc_estimates[kc_sel])
                hardness += sum([(1 - kc_estimates[kc]) for kc in step
                                 if kc_estimates[kc] < self.threshold])
            focus = p_unknown / hardness
            focus_scores.append(focus)
            ids.append(p)

        # select problem at random based on focus
        p_probs = [f / np.sum(focus_scores) for f in focus_scores]
        p_id = np.random.choice(ids, p=p_probs)
        self.problem_pool.mark_problem(p_id)
        return p_id, self.problem_pool.problems[p_id]
