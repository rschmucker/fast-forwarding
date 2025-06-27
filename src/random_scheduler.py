"""Scheduler that selects problems at random."""
import random
from src.problem_pool import ProblemPool


class RandomScheduler():
    """Implements random scheduler as baseline."""

    def __init__(self, problem_path) -> None:
        """Initializes new random scheduler object.

        Args:
            problem_path (str): path to problem file
        """
        self.problem_pool = ProblemPool(problem_path)

    def select_problem(self, kc_estimates):
        """Select next problem at random.

        Args:
            kc_estimates (dict): Mastery estimates (random is agnostic)

        Returns:
            problem (list): Next problem as step list
        """
        a_p_ids = self.problem_pool.get_available_problems()
        if len(a_p_ids) == 0:  # exhausted problem pool
            return -1, -1
        p_id = random.choice(a_p_ids)
        self.problem_pool.mark_problem(p_id)
        return p_id, self.problem_pool.problems[p_id]
