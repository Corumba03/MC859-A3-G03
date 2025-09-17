from .Evaluator import Evaluator
from .. import Solution


class QBF(Evaluator):
    """
    Quadratic Binary Function evaluator: f(x) = x^T A x
    """

    def __init__(self, n: int, A: list[list[float]]):
        self.size = n
        self.A = A

    def is_feasible(self, sol: Solution) -> bool:
        return True  # All binary vectors are feasible

    # Returns the number of decision variables
    def get_domain_size(self) -> int:
        return self.size

    # Returns the value of the objective function for a given solution
    def evaluate(self, sol: Solution) -> float:
        value = 0.0
        for i in sol.elements:
            for j in sol.elements:
                value += self.A[i][j]
        sol.cost = value
        return value

    # Returns the cost variation of inserting elem into sol
    def evaluate_insertion_cost(self, elem: int, sol: Solution) -> float:
        if elem in sol:
            return 0.0
        delta = sum(self.A[elem][j] + self.A[j][elem] for j in sol.elements)
        delta += self.A[elem][elem]
        return delta


    # Returns the cost variation of removing elem from sol
    def evaluate_removal_cost(self, elem: int, sol: Solution) -> float:
        if elem not in sol:
            return 0.0
        delta = -sum(self.A[elem][j] + self.A[j][elem] for j in sol.elements if j != elem)
        delta -= self.A[elem][elem]
        return delta


    # Returns the cost variation of exchanging elem_out with elem_in in sol
    def evaluate_exchange_cost(self, elem_in: int, elem_out: int, sol: Solution) -> float:
        if elem_in == elem_out:
            return 0.0
        if elem_in in sol and elem_out not in sol:
            return 0.0  # no change

        delta_remove = self.evaluate_removal_cost(elem_out, sol) if elem_out in sol else 0.0
        # For insertion, compute as if elem_out was already removed
        elems_after_removal = sol.elements - {elem_out}
        delta_insert = 0.0
        if elem_in not in elems_after_removal:
            delta_insert = sum(self.A[elem_in][j] + self.A[j][elem_in] for j in elems_after_removal)
            delta_insert += self.A[elem_in][elem_in]

        return delta_remove + delta_insert


    # ------------------------
    # Utilities
    # ------------------------

    def get_matrix(self) -> list[list[float]]:
        return self.A

    def print_matrix(self):
        for row in self.A:
            print(" ".join(f"{val:.2f}" for val in row))

