from .. import Solution

class SetCover():
    """
    Generic Set Cover manager.
    Provides methods to check feasibility and filter candidates.
    """

    sets = None  # List of sets (each set is a set of integers)
    num_elements = 0  # Total number of elements to be covered

    def __init__(self, sets: list[set[int]], num_elements: int):
        self.sets = sets
        self.num_elements = num_elements
    
    def get_domain_size(self) -> int:
        return len(self.sets)
    
    def get_sets(self) -> list[set[int]]:
        return self.sets

    def is_feasible(self, sol) -> bool:
        """
        Checks if a solution covers all required elements.
        Accepts either a Solution object or an iterable of indices.
        """
        if hasattr(sol, 'elements'):
            indices = sol.elements
        else:
            indices = sol
        covered = set()
        for i in indices:
            covered.update(self.sets[i])
        return len(covered) == self.num_elements

    def coverage(self, sol) -> set[int]:
        """
        Returns the set of elements covered by the current solution.
        Accepts either a Solution object or an iterable of indices.
        """
        if hasattr(sol, 'elements'):
            indices = sol.elements
        else:
            indices = sol
        covered = set()
        for i in indices:
            covered.update(self.sets[i])
        return covered
