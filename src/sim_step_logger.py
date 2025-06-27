"""Helper class to log and analyse simulation data."""
import numpy as np


class SimStepLogger():
    """Helper class to log and analyse simulation data."""

    def __init__(self):
        """Create new SimLogger object."""
        self.num_logs = 0
        self.buffer = []
        self.logs = []

    def save_buffer(self):
        """Save buffer to permanent logs and reset for next run."""
        self.logs.append(self.buffer)
        self.num_logs += 1
        self.buffer = []

    def to_buffer(self, p_log):
        """Add step-level log to buffer.

        Args:
            p_log (dict): step-level question, correctness, mastery information
        """
        self.buffer.append(p_log)

    def get_masteries(self):
        """Compute avg. mastery after each step across all logs.

        Returns:
            masteries (dict): average mastery for each KC over time
        """
        assert len(self.logs) >= 1, "need at least one stored log"
        masteries = {k: [] for k in self.logs[0][0]["masteries"]}
        n_steps = min([len(lo) for lo in self.logs])
        for i in range(n_steps):  # iterate after each step
            for kc in masteries:
                avg = np.mean([lo[i]["masteries"][kc] for lo in self.logs])
                masteries[kc].append(avg)
        return masteries

    def get_opportunities(self):
        """Compute avg. opportunity number after each step across all logs.

        Returns:
            opportunities (dict): average opportunity number for each KC
        """
        assert len(self.logs) >= 1, "need at least one stored log"
        opportunities = {k: [] for k in self.logs[0][0]["masteries"]}
        n_steps = min([len(lo) for lo in self.logs])
        for i in range(n_steps):  # iterate after each step
            for kc in opportunities:
                counts = []
                for lo in self.logs:  # go through all logs
                    c = 0
                    for j in range(i + 1):  # move to current step
                        c += int(kc in lo[j]["step"])
                    counts.append(c)
                opportunities[kc].append(np.mean(counts))
        return opportunities

    def get_kc_opportunity(self, n_steps=-1):
        """Get KC opportunity information for downstream analysis.

        Returns:
            log_dfs (list): Contains information for each individual run
            n_steps (int): number of problems to consider (-1 for all)
        """
        assert len(self.logs) >= 1, "need add least one stored log"
        # NOTE: MASTERY LEARNING MIGHT CHANGE THIS
        #assert n_steps <= min([len(lo) for lo in self.logs]), \
        #    "Not: n_probs < len(self.logs[0])"
        if n_steps == -1:  # -1 to consider all problems
            n_steps = 1000000
            # n_steps = min([len(lo) for lo in self.logs])
        else:
            n_steps += 1

        log_dfs = []
        for log in self.logs:
            dfs = {}
            # NOTE: This is only used to determine over and under practice??
            for k in log[0]["masteries"]:
                dfs[k] = {"att_number": [], "correct": []}
            for e in log[:n_steps]:
                for k in e["step"]:
                    dfs[k]["correct"].append(e["corrects"][0])
                    dfs[k]["att_number"].append(len(dfs[k]["correct"]))
            log_dfs.append(dfs)
        return log_dfs
