"""Helper class to manage problem pool."""
import os
import json


class ProblemPool():
    """Manages problem pool for simulation."""

    def __init__(self, problem_path: str) -> None:
        """Create new ProblemPool object.

        Args:
            problem_path (str): path to problem file
        """
        assert os.path.exists(problem_path), "path needs to point to valid file"
        with open(problem_path, "r") as file:
            self.problems = json.load(file)
        self.problem_path = problem_path
        self.problem_shown = {p: False for p in self.problems}
        self.unavailable_skills = []

    def get_available_problems(self):
        """Return list of p_ids that have not been shown yet.

        Returns:
            p_ids (str): list of available problem IDs
        """
        p_ids = [p for p in self.problems if not self.problem_shown[p]]
        return p_ids

    def mark_problem(self, p_id):
        """Mark selected problem as shown.

        Args:
            p_id (str): problem ID to mark as selected
        """
        assert p_id in self.problems, "received unknown problem ID"   
        self.problem_shown[p_id] = True

    def replenish_pool(self):
        self.problem_shown = {p: False for p in self.problems}
        self.unavailable_skills = []
        
    def check_skill_available(self, skill):
        """Check if there is at least one new provel for a skill.

        Args:
            skill (str): KC name
        """
        available = self.get_available_problems()
        for p_id in available:
            for step in self.problems[p_id]:
                if skill in step:
                    return True
        return False

    def add_unavailable_skill(self, skill):
        self.unavailable_skills.append(skill)
