from abc import abstractmethod
import random
from collections import deque
from .AbstractTS import AbstractTS
from TabuSearch.Solution import Solution
from ..problems import Evaluator

class TS(AbstractTS):
    """
    Tabu Search metaheuristic for maximization problems.
    """

    def __init__(self, obj_function: Evaluator, 
                 tenure: int = 0, 
                 no_improv_iter: int = 100,
                 maximize: bool = True,
                 tabu_size: int = 0,
                 search_type: str = 'first',
                 constructive_type: str = 'std',
                 tabu_check: str = 'strict',
                 alt_strategy: str = None):
        """
        Constructor for the TS class.
        Args:
            obj_function: The objective function being maximized.
            tenure: The Tabu tenure parameter.
            no_improv_iter: The number of iterations without improvement to stop the search.
            constructive_type: The type of constructive heuristic to use ('random' or 'cost_ratio').
            search_type: The type of local search to use ('first' or 'best').
            tabu_check: The type of tabu check to use ('strict' or 'relaxed').
            alt_strategy: The alternative strategy to use when no improving move is found ('intensification' or 'diversification').
        """
        super().__init__(obj_function, tenure, no_improv_iter, maximize=maximize, constructive_type=constructive_type)
        self.tabu_size = tabu_size
        self.tabu_check = tabu_check
        self.search_type = search_type
        self.constructive_type = constructive_type
        self.alt_strategy = alt_strategy


    def create_empty_sol(self):
        """
        Creates a new solution which is empty, i.e., does not contain any candidate solution element.
        Returns:
            An empty solution.
        """
        sol = Solution(maximize=self.maximize)
        return sol  

    def make_CL(self) -> list:
        """
        Creates the Candidate List, which is a list of candidate elements that can enter a solution.
        Returns:
            The Candidate List.
        """
        # All elements are candidates initially
        return [elem for elem in range(self.obj_function.get_domain_size())]
    
    def update_CL(self) -> None:
        """
        No candidate list update is needed in this implementation.
        """
        self.CL = [elem for elem in range(self.obj_function.get_domain_size()) if elem not in self.sol.elements]

    def make_RCL(self) -> list:
        """
        Creates the Restricted Candidate List, which is a list of the best candidate elements that be removed from the solution.
        Returns:
            The Restricted Candidate List.
        """
        if not self.CL:
            self.RCL = []
            return self.RCL

        # Compute cost of adding each element separately in CL to the current solution
        deltas = {elem: self.obj_function.evaluate_insertion_cost(elem, self.sol) for elem in self.CL}
        min_cost = min(deltas.values())
        max_cost = max(deltas.values())
        # Build RCL based on threshold
        if self.maximize:
            threshold = max_cost - 0.5 * (max_cost - min_cost)
            self.RCL = [c for c, delta in deltas.items() if delta >= threshold]
        else:
            threshold = min_cost + 0.5 * (max_cost - min_cost)
            self.RCL = [c for c, delta in deltas.items() if delta <= threshold]

        return self.RCL

    def make_TL(self) -> None:
        """
        Creates the Tabu List, which is a deque of the Tabu candidate elements plus a dictionary for fast lookup.
        """
        self.TL_deq = deque(maxlen=self.tenure)
        self.TL_dict = {}

    def update_TL(self, move) -> None:
        """
        Updates the Tabu List by adding the most recent move.
        Args:
            move: The move to add to the Tabu List, represented as a tuple (elem_out, elem_in).
        """
        if move is None:
            return
        
        elem_in, elem_out = move
        # Add the move to the tabu dictionary and deque
        if elem_in not in self.TL_dict.keys():
            self.TL_dict[elem_in] = set([elem_out])
        else:
            self.TL_dict[elem_in].add(elem_out)
        self.TL_deq.append((elem_in, elem_out))

        # Maintain the size of the tabu list
        if len(self.TL_deq) > self.tenure:
            old_elem_in, old_elem_out = self.TL_deq.popleft()
            self.TL_dict[old_elem_in].discard(old_elem_out)

    def is_tabu_strict(self, move) -> bool:
        """
        Checks if a move is tabu. The check is strict, in other words, it prevents exchanges 
        even if only insertion or removal is tabu.
        Args:
            move: The move to check, represented as a tuple (elem_out, elem_in).
        Returns:
            True if the move is tabu, False otherwise.
        """
        elem_in, elem_out = move
        # Checks if the insertion of the element is tabu
        if elem_in != -1 and elem_in in self.TL_dict.keys():
            return True
        # Checks if the removal of the element is tabu
        if elem_out != -1 and any(elem_out in sublist for sublist in self.TL_dict.values()):
            return True
        
    def is_tabu_relaxed(self, move) -> bool:
        """
        Checks if a move is tabu. The check is relaxed, in other words, it allows exchanges 
        if only insertion or removal is tabu.
        Args:
            move: The move to check, represented as a tuple (elem_out, elem_in).
        Returns:
            True if the move is tabu, False otherwise.
        """
        elem_in, elem_out = move
        # Checks if the insertion of the element is tabu
        if elem_in in self.TL_dict.keys():
            return elem_out in self.TL_dict[elem_in]
        return False
    
    def is_tabu(self, move) -> bool:
        """
        Checks if a move is tabu based on the tabu_size parameter.
        Args:
            move: The move to check, represented as a tuple (elem_out, elem_in).
        Returns:
            True if the move is tabu, False otherwise.
        """
        if self.tabu_check == 'strict':
            return self.is_tabu_strict(move)
        elif self.tabu_check == 'relaxed':
            return self.is_tabu_relaxed(move)
        return False

    def aspiration_criteria(self, candidate):
        """
        Aspiration criteria: allows a tabu move if it results in a solution better than the best known solution.
        Args:
            candidate: The candidate element being considered for addition to the solution.
            candidate_cost: The cost of adding the candidate element to the solution.
        Returns:
            True if the move is allowed, False otherwise.
        """
        if self.maximize:
            return candidate.cost > self.best_sol.cost
        else:
            return candidate.cost < self.best_sol.cost

    def get_neighborhood(self, sol: Solution) -> list:
        """
        Generates the neighborhood of the current solution by considering all possible moves, 
        where a move is represented as a tuple (elem_in, elem_out) with -1 indicating no element 
        is added or removed.
        Args:
            sol: The current solution.
        Returns:
            A list of neighboring solutions.
        """
        neighborhood = []
        for elem in range(self.obj_function.get_domain_size()):
            if elem in sol: # Consider removal of elements in the solution
                neighbor = (-1, elem) # Removal move
                neighborhood.append(neighbor)
            elif elem not in sol: # Consider addition of elements not in the solution
                neighbor = (elem, -1) # Addition move
                neighborhood.append(neighbor)
            for _ in sol.elements: # Consider exchanges
                if _ != elem:
                    neighbor = (elem, _) # Exchange move
                    neighborhood.append(neighbor)

        return neighborhood
    
    def local_search_best(self, neighborhood):
        """
        Performs a best-improvement local search on the neighborhood.
        Args:
            neighborhood: The list of neighboring solutions.
        Returns:
            The best improving solution found, or the least bad move if no improvement is found.
        """
        best_candidate = self.sol.copy()
        move = None
        least_bad = None
        for elem_in, elem_out in neighborhood:
            if elem_in != -1 and elem_out == -1:
                candidate = self.sol.insert(elem_in)
                candidate.cost = self.sol.cost + self.obj_function.evaluate_insertion_cost(elem_in, self.sol)
            elif elem_out != -1 and elem_in == -1:
                candidate = self.sol.remove(elem_out)
                candidate.cost = self.sol.cost + self.obj_function.evaluate_removal_cost(elem_out, self.sol)
            else:
                candidate = self.sol.exchange(elem_in, elem_out)
                candidate.cost = self.sol.cost + self.obj_function.evaluate_exchange_cost(elem_in, elem_out, self.sol)

            # Check if the move is tabu
            if self.is_tabu((elem_in, elem_out)) and not self.aspiration_criteria(candidate):
                continue

            # Update best candidate found
            if (self.maximize and candidate.cost > best_candidate.cost) or (not self.maximize and candidate.cost < best_candidate.cost):
                best_candidate = candidate.copy()
                move = (elem_in, elem_out)
            
            # Keep track of the least bad candidate in case no improving move is found
            if least_bad is None:
                least_bad = candidate.copy()
                move = (elem_in, elem_out)
            elif (self.maximize and candidate.cost > least_bad.cost) or (not self.maximize and candidate.cost < least_bad.cost):
                least_bad = candidate.copy()
                move = (elem_in, elem_out)

        print(f'No improving move found, taking least bad move: {move}, solution cost: {least_bad.cost if least_bad else "N/A":.2f}')

        return best_candidate, move

    def local_search_first(self, neighborhood):
        """
        Performs a first-improvement local search on the neighborhood.
        Args:
            neighborhood: The list of neighboring solutions.
        Returns:
            The first improving solution found, or None if no improvement is found.
        """
        best_candidate = self.sol.copy()
        least_bad = None
        move = None
        for elem_in, elem_out in neighborhood:
            if elem_out == -1:
                candidate = self.sol.insert(elem_in)
                candidate.cost = self.sol.cost + self.obj_function.evaluate_insertion_cost(elem_in, self.sol)
            elif elem_in == -1:
                candidate = self.sol.remove(elem_out)
                candidate.cost = self.sol.cost + self.obj_function.evaluate_removal_cost(elem_out, self.sol)
            else:
                candidate = self.sol.exchange(elem_in, elem_out)
                candidate.cost = self.sol.cost + self.obj_function.evaluate_exchange_cost(elem_in, elem_out, self.sol)

            # Check if the move is tabu
            if self.is_tabu((elem_in, elem_out)) and not self.aspiration_criteria(candidate):
                continue

            # Update best candidate found
            if (self.maximize and candidate.cost > best_candidate.cost) or (not self.maximize and candidate.cost < best_candidate.cost):
                best_candidate = candidate.copy()
                move = (elem_in, elem_out)

                return best_candidate, move
            
            # Keep track of the least bad candidate in case no improving move is found
            if least_bad is None:
                least_bad = candidate.copy()
                move = (elem_in, elem_out)
            elif (self.maximize and candidate.cost > least_bad.cost) or (not self.maximize and candidate.cost < least_bad.cost):
                least_bad = candidate.copy()
                move = (elem_in, elem_out)

        print(f'No improving move found, taking least bad move: {move}, solution cost: {least_bad.cost:.2f}')
        return least_bad, move

    def create_random_sol(self):
        """
        Creates a random solution for diversification.
        Returns:
            A random solution.
        """
        random_sol = self.create_empty_sol()
        domain_size = self.obj_function.get_domain_size()
        # Randomly activate subsets (variables) while respecting constraints
        while not self.obj_function.is_feasible(random_sol):
            for elem in range(domain_size):
                if random.choice([True, False]):
                    random_sol.add(elem)
        self.obj_function.evaluate(random_sol)
        return random_sol

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
        neighborhood = self.get_neighborhood(self.sol)
        best_candidate = None

        # Select the best candidate based on the search type
        if self.search_type == 'best':
            best_candidate, move = self.local_search_best(neighborhood)
        elif self.search_type == 'first':
            best_candidate, move = self.local_search_first(neighborhood)

        # Move to the best candidate (if any) and update the Tabu List
        if self.alt_strategy and self.no_improv % (self.no_improv_iter // 5) == 0:
            # If there's no improving move and we've had no improvement for half the allowed iterations, restart the search
            if self.alt_strategy == 'intensification':
                print("Intensification: Restarting from the best solution found so far.")
                self.sol = self.best_sol.copy()
            elif self.alt_strategy == 'diversification':
                print("Diversification: Restarting from a random solution.")
                self.sol = self.create_random_sol()
        elif best_candidate.cost != self.sol.cost:
            self.sol = best_candidate.copy()
            self.update_TL(move)

            # Update the best solution found so far
            if (self.maximize and self.sol.cost > self.best_sol.cost) or (not self.maximize and self.sol.cost < self.best_sol.cost):
                self.best_sol = self.sol.copy()
        