from abc import abstractmethod
from collections import deque
from AbstractTS import AbstractTS
from TabuSearch.Solution import Solution
from ..problems import Evaluator

class TS(AbstractTS):
    """
    Tabu Search metaheuristic for maximization problems.
    """

    def __init__(self, obj_function: Evaluator, 
                 tenure: int = 7, 
                 iterations: int = 100,
                 maximize: bool = True,
                 tabu_size: int = 0,
                 search_type: str = 'first',
                 constructive_type: str = 'std',):
        """
        Constructor for the TS class.
        Args:
            obj_function: The objective function being maximized.
            tenure: The Tabu tenure parameter.
            iterations: The number of iterations which the TS will be executed.
            constructive_type: The type of constructive heuristic to use ('random' or 'cost_ratio').
            search_type: The type of local search to use ('first' or 'best').
        """
        super().__init__(obj_function, tenure, iterations, maximize=maximize)
        self.tabu_size = tabu_size
        self.search_type = search_type
        self.constructive_type = constructive_type
    
    def create_empty_sol(self):
        """
        Creates a new solution which is empty, i.e., does not contain any candidate solution element.
        Returns:
            An empty solution.
        """
        sol = Solution(maximize=self.maximize)
        for elem in range(self.objFunction.get_domain_size()):
            sol.add(elem)
        self.objFunction.evaluate(sol)
        return sol  

    def make_CL(self) -> set:
        """
        Creates the Candidate List, which is a list of candidate elements that can enter a solution.
        Returns:
            The Candidate List.
        """
        # All elements are candidates initially
        return set([elem for elem in range(self.objFunction.get_domain_size()) if elem in self.sol.elements])
    
    def update_CL(self) -> None:
        """
        No candidate list update is needed in this implementation.
        """
        self.CL = self.sol.elements

    def make_RCL(self) -> set:
        """
        Creates the Restricted Candidate List, which is a list of the best candidate elements that be removed from the solution.
        Returns:
            The Restricted Candidate List.
        """
        if not self.CL:
            self.RCL = []
            return self.RCL

        # Compute cost of adding each element separately in CL to the current solution
        deltas = {elem: self.obj_function.evaluate_removal_cost(elem, self.sol) for elem in self.CL}
        min_cost = min(deltas.values())
        max_cost = max(deltas.values())
        
        # Build RCL based on threshold
        if self.maximize:
            threshold = max_cost - (max_cost - min_cost)
            self.RCL = {c for c, delta in deltas.items() if delta >= threshold}
        else:
            threshold = min_cost + (max_cost - min_cost)
            self.RCL = {c for c, delta in deltas.items() if delta <= threshold}

        return self.RCL

    def make_TL(self) -> deque:
        """
        Creates the Tabu List, which is a deque of the Tabu candidate elements.
        Returns:
            The Tabu List.
        """
        pass


    def neighborhood_move(self):
        """
        The TS local search phase is responsible for repeatedly applying a neighborhood operation 
        while the solution is getting improved, until a local optimum is attained. When a local 
        optimum is attained the search continues by exploring moves which can make the current 
        solution worse. Cycling is prevented by not allowing forbidden (tabu) moves that would 
        otherwise backtrack to a previous solution.
        Returns:
            A local optimum solution.
        """
        
        pass
        