class Solution:
    """
    Represents a solution in the GRASP framework.

    Stores a list of elements and its associated cost.

    Basically a wrapper around a set with some utility methods.
    """

    def __init__(self, other: "Solution" = None, maximize: bool = True):
        self.maximize = maximize 
        if other is None:
            self.elements = set()
            self.cost = float('inf') if not self.maximize else float('-inf')
        else:
            # Copy constructor
            self.elements = set(other.elements)
            self.cost = other.cost

    def add(self, elem):
        """Adds an element to the solution."""
        self.elements.add(elem)

    def delete(self, elem):
        """Removes an element from the solution."""
        self.elements.discard(elem)
    
    def exchange(self, elem_in, elem_out):
        """Returns a solution with elem_out removed and elem_in added."""
        if elem_out == elem_in:
            return self.copy()
        new_sol = self.copy()
        new_sol.delete(elem_out)
        new_sol.add(elem_in)
        new_sol.cost = float("-inf") if self.maximize else float("inf")
        return new_sol

    def remove(self, elem):
        """Returns a solution without the given element."""
        new_sol = self.copy()
        new_sol.delete(elem)
        new_sol.cost = float("-inf") if self.maximize else float("inf")
        return new_sol
    
    def insert(self, elem):
        """Returns a solution with the given element added."""
        new_sol = self.copy()
        new_sol.add(elem)
        new_sol.cost = float("-inf") if self.maximize else float("inf")
        return new_sol

    def __len__(self):
        return len(self.elements)

    def __iter__(self):
        return iter(self.elements)

    def __contains__(self, item):
        return item in self.elements

    def __str__(self):
        return f"Solution: cost=[{self.cost:.2f}], size=[{len(self.elements)}], elements={self.elements}"

    def copy(self) -> "Solution":
        """Returns a copy of this solution."""
        return Solution(self)
