"""Focus scheduler that selects KC deterministically and problem at random."""
import random
import numpy as np
from src.problem_pool import ProblemPool


class FocusDetScheduler():
    """Implements focus scheduler with deterministic KC."""

    def __init__(self, problem_path, threshold=0.95) -> None:
        """Initializes new focus scheduler with deterministic KC.

        Args:
            problem_path (str): path to problem file
            threshold (float): mastery threshold
        """
        self.problem_pool = ProblemPool(problem_path)
        self.threshold = threshold

    def select_problem(self, kc_estimates):
        """Select next problem using focus with deterministic kc.

        Args:
            kc_estimates (dict): Mastery estimates

        Returns:
            problem (list): Next problem as step list
        """
        a_p_ids = self.problem_pool.get_available_problems()
        if len(a_p_ids) == 0:  # exhausted problem pool
            return -1, -1

        # determine easiest not mastered KC
        kc_easy, easy_val = "-", 1.0
        for kc in kc_estimates:
            has_problem = False
            for p in a_p_ids:
                if np.any([kc in s for s in self.problem_pool.problems[p]]):
                    has_problem = True
            if kc_estimates[kc] >= self.threshold:  # mastered
                continue
            elif not has_problem:  # no problem available
                continue
            elif kc_estimates[kc] > easy_val:
                kc_easy = kc
                easy_val = kc_estimates[kc]
        if kc_easy == "-":  # student mastered everything OR no problems left
            # TODO: Implement early stopping
            p_id = random.choice(a_p_ids)
            self.problem_pool.mark_problem(p_id)
            return p_id, self.problem_pool.problems[p_id]

        # select problem based on focus score
        focus_scores, ids = [], []
        for p in a_p_ids:
            rel = np.any([kc_easy in s for s in self.problem_pool.problems[p]])
            if not rel:
                continue  # this p is not relevant 
            p_unknown, hardness = 0, 0
            for step in self.problem_pool.problems[p]:
                if kc_easy in step:
                    p_unknown += (1 - kc_estimates[kc_easy])
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
