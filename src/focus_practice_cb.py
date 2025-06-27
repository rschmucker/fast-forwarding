"""
Scheduler based on Huang, Y., Brusilovsky, P., Guerra, J., Koedinger, K., & Schunn, C. (2023). Supporting skill integration in an intelligent tutoring system for code tracing. Journal of Computer Assisted Learning, 39(2), 477-500.
Based on V1.1 implementation for data-driven optimization experiment.
V1.1 has the following changelog:
Changelog (compared to initial implementation strictly following the algorithm's description):
-- Add multinomal selection of problem based on focus score instead of deterministic selection
-- Compute focus score based on how many opportunities the selected skill has in a problem, rather than if they appear or not only
-- Version 1.1: Do not stop giving problems if the selected skill has no more problems but another unmastered skill has. Currently brute-forced, could be optimized.
"""
import random
import json
from collections import Counter
from src.problem_pool import ProblemPool
from src.bkt_inference import BKTInference
import numpy as np

class FocusedPracticeScheduler():
    """Implements mastery hard scheduler as baseline."""

    def __init__(self, problem_path, threshold=0.95) -> None:
        """Initializes new mastery hard scheduler object.

        Args:
            problem_path (str): path to problem file
            threshold (float): mastery threshold
        """
        self.problem_pool = ProblemPool(problem_path)
        self.threshold = threshold

    def get_skill_occurrences(self):
        def flatten(xss):
            return [x for xs in xss for x in xs]
        ans = {k: Counter(flatten(v)) for k, v in self.problem_pool.problems.items()}
        return ans

    def get_skill_distribution(self, bkt_kc_estimates):
        """
        For multinomial.
        """
        skills = bkt_kc_estimates#bkt_inf.get_state()
        skills = {k: v for k, v in skills.items() if v < 0.95}
        total_knowledge = sum(skills.values())
        ans = {k: v/total_knowledge for k, v in skills.items()}
        return ans

    def get_focus_scores(self, bkt_kc_estimates, focus_skill):
        problem_skill_dict = self.get_skill_occurrences()
        mastery = bkt_kc_estimates#bkt_inf.get_state()
        skill_unmastered_dict = {skill: 1-p_known for skill, p_known in mastery.items()}
        def focus_score(d_entry, focus_skill, skill_unmastered_dict):
            focus_unmastered_knowledge = skill_unmastered_dict[focus_skill]*d_entry[focus_skill]
            all_unmastered_knowledge = sum([skill_unmastered_dict[skill]*occurrence for skill, occurrence in d_entry.items()])
            return focus_unmastered_knowledge/all_unmastered_knowledge if all_unmastered_knowledge>0 else 0
        ans_focus_scores = {problem: focus_score(entry, focus_skill, skill_unmastered_dict) for problem, entry in problem_skill_dict.items()}
        
        #print(ans_focus_scores)
        if sum(ans_focus_scores.values())==0:
            self.problem_pool.add_unavailable_skill(focus_skill)
            return False
        
        # standardize focus score distribution for multinomial (p's must sum up to 1)
        total_focus_score = sum(ans_focus_scores.values())
        ans_focus_scores = {k: v/total_focus_score for k, v in ans_focus_scores.items()}
        return ans_focus_scores

    def select_skill(self, bkt_kc_estimates):
        """
        Selects a skill based on the distribution.
        Returns -1 if the distribution is empty.
        """
        d = self.get_skill_distribution(bkt_kc_estimates)
        
        if not d:
            return -1
        
        # Sort keys and corresponding probabilities to ensure consistent ordering
        sorted_items = sorted(d.items())  # This sorts by key
        sorted_keys = [item[0] for item in sorted_items]
        sorted_probabilities = [item[1] for item in sorted_items]
        
        # Ensure the probabilities sum to 1
        sorted_probabilities = np.array(sorted_probabilities)
        sorted_probabilities /= sorted_probabilities.sum()
        
        draw = np.random.choice(sorted_keys, 1, p=sorted_probabilities)
        
        return draw[0]


    def select_problem(self, bkt_kc_estimates):
        focus_skill = self.select_skill(bkt_kc_estimates)
        d = self.get_focus_scores(bkt_kc_estimates, focus_skill)
        while not d: 
            #print('retrying another skill')
            focus_skill = self.select_skill(bkt_kc_estimates)
            if focus_skill == -1: # no more skills left to practice in pool
                return -1, -1
            d = self.get_focus_scores(bkt_kc_estimates, focus_skill)

        # Sort keys and corresponding probabilities to ensure consistent ordering
        sorted_items = sorted(d.items())  # This sorts by key
        sorted_keys = [item[0] for item in sorted_items]
        sorted_probabilities = [item[1] for item in sorted_items]
        
        # Ensure the probabilities sum to 1
        sorted_probabilities = np.array(sorted_probabilities)
        sorted_probabilities /= sorted_probabilities.sum()
        
        draw = np.random.choice(sorted_keys, 1, p=sorted_probabilities)[0]
        self.problem_pool.mark_problem(draw)
        return draw, self.problem_pool.problems[draw]