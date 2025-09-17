from .Evaluator import Evaluator
from TabuSearch.Solution import Solution
from .QBF import QBF
from .SetCover import SetCover as SC

class SC_QBF(Evaluator):
    """
    Set Covering Quadratic Binary Function (SC-QBF) problem.
    Maximizes a quadratic function over variables activated by selected sets,
    while ensuring full coverage.
    """

    def __init__(self, n: int, A: list[list[float]], sets: list[set[int]]):
        self.n = n
        self.A = A
        self.sets = sets

        self.SC = SC(sets, n)

    def is_feasible(self, sol: Solution) -> bool:
        return sol and sol.elements and self.SC.is_feasible(sol)

    def evaluate(self, sol: Solution, allow_partial: bool = True) -> float:
        if not self.is_feasible(sol) and not allow_partial:
            sol.cost = float("-inf")
            return sol.cost

        covered = sorted(self.SC.coverage(sol))
        cost = 0.0
        for idx_i, i in enumerate(covered):
            for j in covered[idx_i:]:
                cost += self.A[i][j]
        sol.cost = cost
        return cost

    def evaluate_insertion_cost(self, elem: int, sol: Solution) -> float:
        if elem in sol:
            return 0.0

        current_vars = self.SC.coverage(sol)
        new_vars = self.sets[elem] - current_vars
        if not new_vars:
            return 0.0

        # Only sum upper triangle (i <= j)
        all_vars = sorted(current_vars | new_vars)
        delta = 0.0
        for idx_i, i in enumerate(all_vars):
            for j in all_vars[idx_i:]:
                # Only add if i or j is in new_vars (i.e., new contribution)
                if i in new_vars or j in new_vars:
                    delta += self.A[i][j]
        return delta

    def evaluate_removal_cost(self, elem: int, sol: Solution) -> float:
        if elem not in sol:
            return 0.0

        current_vars = self.SC.coverage(sol)
        removed_vars = self.sets[elem] & current_vars
        remaining_vars = sorted(current_vars - removed_vars)

        # Only allow removal if still feasible
        if not self.SC.is_feasible(sol.remove(elem)):
            return float("-inf")

        # Only sum upper triangle (i <= j)
        delta = 0.0
        for idx_i, i in enumerate(current_vars):
            for j in current_vars[idx_i:]:
                if i in removed_vars or j in removed_vars:
                    delta -= self.A[i][j]
        return delta

    def evaluate_exchange_cost(self, elem_in: int, elem_out: int, sol: Solution) -> float:
        if elem_in == elem_out:
            return 0.0

        sol_after_removal = sol.remove(elem_out)
        if not self.SC.is_feasible(sol_after_removal.insert(elem_in)):
            return float("-inf")

        # Compute insertion and removal deltas using upper triangle logic
        delta = self.evaluate_removal_cost(elem_out, sol)
        delta += self.evaluate_insertion_cost(elem_in, sol_after_removal)
        return delta

    def get_domain_size(self) -> int:
        return len(self.sets)
