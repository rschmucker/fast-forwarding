"""Scheduler that selects problems using mastery hard."""
import random
from src.problem_pool import ProblemPool


class MasteryHardScheduler():
    """Implements mastery hard scheduler as baseline."""

    def __init__(self, problem_path, threshold=0.95) -> None:
        """Initializes new mastery hard scheduler object.

        Args:
            problem_path (str): path to problem file
            threshold (float): mastery threshold
        """
        self.problem_pool = ProblemPool(problem_path)
        self.threshold = threshold

    def select_problem(self, kc_estimates):
        """Select next problem using mastery hard.

        Args:
            kc_estimates (dict): Mastery estimates

        Returns:
            problem (list): Next problem as step list
        """
        a_p_ids = self.problem_pool.get_available_problems()
        if len(a_p_ids) == 0:  # exhausted problem pool
            return -1, -1

        hard_p_id, hard_val = "-", 0.0
        for p_id in a_p_ids:
            # determine hardness
            hardness, opp_count = 0, 0
            for step in self.problem_pool.problems[p_id]:
                opp_count = sum([1 for kc in step
                                 if kc_estimates[kc] < self.threshold])
                hardness += sum([(1 - kc_estimates[kc]) for kc in step
                                 if kc_estimates[kc] < self.threshold])
            if (hardness > 0.0) and (opp_count > 0):
                hardness = hardness / opp_count
                if (hardness > hard_val):  # select most difficult problem
                    hard_p_id = p_id
                    hard_val = hardness

        if hard_p_id == "-":  # NOTE: handle all mastered case
            # TODO: implement early stopping
            hard_p_id = random.choice(a_p_ids)

        self.problem_pool.mark_problem(hard_p_id)
        return hard_p_id, self.problem_pool.problems[hard_p_id]
