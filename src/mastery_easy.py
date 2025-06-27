"""Scheduler that selects problems using mastery easy."""
import random
from src.problem_pool import ProblemPool


class MasteryEasyScheduler():
    """Implements mastery easy scheduler as baseline."""

    def __init__(self, problem_path, threshold=0.95) -> None:
        """Initializes new mastery easy scheduler object.

        Args:
            problem_path (str): path to problem file
            threshold (float): mastery threshold
        """
        self.problem_pool = ProblemPool(problem_path)
        self.threshold = threshold

    def select_problem(self, kc_estimates):
        """Select next problem using mastery easy.

        Args:
            kc_estimates (dict): Mastery estimates

        Returns:
            problem (list): Next problem as step list
        """
        a_p_ids = self.problem_pool.get_available_problems()
        if len(a_p_ids) == 0:  # exhausted problem pool
            return -1, -1

        easy_p_id, easy_val = "-", 10e8
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
                if (hardness < easy_val):  # select most easiest problem
                    easy_p_id = p_id
                    easy_val = hardness

        if easy_p_id == "-":  # NOTE: handle all mastered case
            # TODO: implement early stopping
            easy_p_id = random.choice(a_p_ids)

        self.problem_pool.mark_problem(easy_p_id)
        return easy_p_id, self.problem_pool.problems[easy_p_id]

""" MASTERY HARD
Select most difficult problem based on same equation as above
"""

"""Select KC (deterministic) + random problem (proportional to focus)
Step 1: Select easiest KC (KC closest to being mastered 0.95)
Step 2: Compute focus scores for each problem that has relevant KC
    # 

p_unknown, hardness = 0, 0
for step in self.problem_pool.problems[p_id]:
    if kc_easy in step:
        p_unknown += (1 - kc_estimates[kc_easy])
    hardness += sum([(1 - kc_estimates[kc]) for kc in step
                        if kc_estimates[kc] < self.threshold])
focus = p_unknown / hardness

# prob(select p) = focus_p / (\sum_{j in P}(focus_j))
"""

"""Select KC (random) + select problem (random)

prob(select k) = (1 p_known(k)) / (\sum_{j in KCs}(1 - p_known(j)))
"""


"""
      # determine easiest not mastered KC
        easy_kc, easy_val = "-", 1.0
        for kc in kc_estimates:
            if kc_estimates[kc] >= self.threshold:
                continue
            elif kc_estimates[kc] > easy_val:
                easy_kc = kc
                easy_val = kc_estimates[kc]

        if easy_kc == "-":  # student mastered everything
            pass ## GIVE RANDOM PROBLEM
"""