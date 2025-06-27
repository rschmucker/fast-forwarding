"""Scheduler that selects problems in predefined order."""
from src.problem_pool import ProblemPool

DETERMINISTIC_ORDER = [
'16=-8x', '18=3x', '6x=12', '6x=18', '-6=-2x', '20=-4x', '3x-1=-10', '7=-1+2x', '-3x-1=-10', '-2x+4=0', '-2x-5=5', '5x+10=0', '2(-3+7x)=8', '16=-4(x-2)', '5(-2+2x)=20', '15=-3(1+x)', '-2(2x-1)=6', '3(3x+1)=21', '3(-x-2)=9', '10=-2(2x+1)', '-4(2x-3)=4', '-2(x+1)=6', '3(-2+x)=9', '20=5(x-1)', '-16=-2x solver', '18=3x solver', '7x=14 solver', '-6(1+x)+8=-4', '18=-2+4(x-1)', '5=3-2(x-2)', '-2(x-3)+6=14', '2-3(x+2)=11', '7(x-6)+4=-24', '10=1+3(x-4)', '10=-3(2x+2)-8', '5(-6+x)-1=-16', '4+3(-1+x)=7', '4x+3=7x', 'x+9=4x', '-12x=-2x+20', '7x=2x-10', '3x+10=5x', '10x-6=4x', '8x-9=5x', '-4x=2x+6', '-2x+6=-3x', '5x=3x+10', '-6x=-2x+8', '3x-9=-6x', '2x+1=5x+10', '9x-3=-3x+21', '4x-10=-x+10', '-4x+10=2x-2', '3x+13=2x+6', '4x+2=-6x-8', '-2x+2=-5x+8', '-12-6x=-3-3x', '-2x+3=-6x+11', '-18-11x=-3-6x', '3x-4=x+6', '-10-7x=-1-4x', '1+2(2x-1)=7', '15=-2x+5', '4+3(-1+x)=13', '15=4x-1', '5(-6+x)-1=-11'
 ]

class DeterministicScheduler():
    """Implements deterministic scheduler as baseline."""

    def __init__(self, problem_path) -> None:
        """Initializes new deterministic scheduler object.

        Args:
            problem_path (str): path to problem file
        """
        self.problem_pool = ProblemPool(problem_path)
        self.pointer = 0

    def select_problem(self, kc_estimates):
        """Select next problem based on pre-defined order.

        Args:
            kc_estimates (dict): Mastery estimates (random is agnostic)

        Returns:
            p_id (str): name of problem
            steps (list): list with KCs for each step
        """
        if self.pointer == len(DETERMINISTIC_ORDER):  # exhausted problem pool
            return -1, -1
        p_id = DETERMINISTIC_ORDER[self.pointer]
        assert not self.problem_pool.problem_shown[p_id], "problem unavailable"
        self.problem_pool.mark_problem(p_id)
        self.pointer += 1
        return p_id, self.problem_pool.problems[p_id]
