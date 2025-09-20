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

    def __init__(self, n: int, 
                 A: list[list[float]], 
                 sets: list[set[int]],
                 maximize: bool = True):
        self.n = n
        self.A = A
        self.sets = sets
        self.maximize = maximize

        self.SC = SC(sets, n)

    def is_feasible(self, sol: Solution) -> bool:
        return self.SC.is_feasible(sol)

    def evaluate(self, sol: Solution) -> float:
        if not self.is_feasible(sol):
            sol.cost = float("-inf") if self.maximize else float("inf")
            return sol.cost

        cost = 0.0
        for i in sol:
            for j in sol:
                if i <= j:  # Only sum upper triangle (i <= j)
                    cost += self.A[i][j]
            cost += self.A[i][i] # Diagonal term
        
        sol.cost = cost
        return cost

    def evaluate_insertion_cost(self, elem: int, sol: Solution) -> float:
        if elem in sol:
            return 0.0

        # Only sum upper triangle (i <= j)
        delta = 0.0
        for var in sol.elements:
            if var < elem:
                delta += self.A[var][elem]
            if var > elem:
                delta += self.A[elem][var]
        delta += self.A[elem][elem]

        return delta

    def evaluate_removal_cost(self, elem: int, sol: Solution) -> float:
        if elem not in sol:
            return 0.0

        # Only allow removal if still feasible
        if not self.SC.is_feasible(sol.remove(elem)):
            return float("-inf") if self.maximize else float("inf")

        current_vars = sol.elements
        removed_vars = current_vars - sol.elements

        # Only sum upper triangle (i <= j)
        delta = 0.0
        for var in sol.elements:
            if var < elem:
                delta -= self.A[var][elem]
            if var > elem:
                delta -= self.A[elem][var]
        delta -= self.A[elem][elem]

        return delta

    def evaluate_exchange_cost(self, elem_in: int, elem_out: int, sol: Solution) -> float:
        if elem_in == elem_out:
            return 0.0

        sol_after_removal = sol.remove(elem_out)
        if not self.SC.is_feasible(sol_after_removal.insert(elem_in)):
            return float("-inf") if self.maximize else float("inf")

        # Compute insertion and removal deltas using upper triangle logic
        delta = self.evaluate_removal_cost(elem_out, sol)
        delta += self.evaluate_insertion_cost(elem_in, sol_after_removal)
        return delta

    def get_domain_size(self) -> int:
        return len(self.sets)
