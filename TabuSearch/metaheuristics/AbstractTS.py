from abc import ABC, abstractmethod
import random
from collections import deque
from TabuSearch.Solution import Solution
from ..problems.Evaluator import Evaluator

# Type variable for generic candidate element

class AbstractTS(ABC):
	"""
	Abstract base class for Tabu Search metaheuristics (minimization problem).
	"""

	# flag that indicates whether the code should print more information on screen
	verbose: bool = True


	def __init__(self, obj_function: Evaluator, 
			  		tenure: int = 0, 
					no_improv_iter: int = 5,
					max_iter: int = 100,
					maximize: bool = True,
					constructive_type: str = 'std'):
		"""
		Constructor for the AbstractTS class.
		Args:
			obj_function: The objective function being minimized.
			tenure: The Tabu tenure parameter.
			no_improv_iter: The number of iterations without improvement to stop the search.
			max_iter: The maximum number of iterations for the search.
			maximize: Whether the objective function is to be maximized (True) or minimized (
			constructive_type: The type of constructive heuristic to use ('std' for standard).
		"""
		# Iniatialize basic parameters of the solver
		self.obj_function = obj_function
		self.no_improv_iter = no_improv_iter # Iterations without improvement
		self.max_iter = max_iter # Maximum number of iterations
		self.maximize = maximize
		self.constructive_type = constructive_type

		# Size of the tabu list, if tenure <= 0, use domain size as tenure otherwise use domain size // tenure (ratio)
		self.tenure = obj_function.get_domain_size() if tenure <= 0 else obj_function.get_domain_size() // tenure
		self.TL_deq: deque = deque()
		self.TL_dict: dict = {}

		# Initialize the solution, Candidate List, and Restricted Candidate List
		self.sol = None
		self.last_sol = None
		self.CL = None
		self.RCL = None
		self.best_sol = None

	@abstractmethod
	def make_CL(self) -> list:
		"""
		Creates the Candidate List, which is a list of candidate elements that can enter a solution.
		Returns:
			The Candidate List.
		"""
		pass

	@abstractmethod
	def make_RCL(self) -> list:
		"""
		Creates the Restricted Candidate List, which is a list of the best candidate elements that can enter a solution.
		Returns:
			The Restricted Candidate List.
		"""
		pass

	@abstractmethod
	def make_TL(self) -> deque:
		"""
		Creates the Tabu List, which is a deque of the Tabu candidate elements.
		Returns:
			The Tabu List.
		"""
		pass

	@abstractmethod
	def update_CL(self) -> None:
		"""
		Updates the Candidate List according to the incumbent solution self.sol.
		Responsible for updating the costs of the candidate solution elements.
		"""
		pass

	@abstractmethod
	def create_empty_sol(self):
		"""
		Creates a new solution which is empty, i.e., does not contain any candidate solution element.
		Returns:
			An empty solution.
		"""
		pass

	@abstractmethod
	def neighborhood_move(self):
		"""
		The TS local search phase is responsible for repeatedly applying a neighborhood operation while the solution is getting improved, until a local optimum is attained. When a local optimum is attained the search continues by exploring moves which can make the current solution worse. Cycling is prevented by not allowing forbidden (tabu) moves that would otherwise backtrack to a previous solution.
		Returns:
			A local optimum solution.
		"""
		pass
	
	# ----------------
	# Concrete methods
	# ----------------
	
	def is_improvement(self, new_cost: float, current_cost: float) -> bool:
		"""
		Auxiliary function to determine if the new cost is an improvement over the current cost.
		"""
		if self.maximize:
			return new_cost > current_cost
		else:
			return new_cost < current_cost

	def add_tabu(self, element) -> None:
		"""
		Adds an element to the Tabu List, maintaining its size within the tenure limit.
		"""
		self.TL.append(element)
		if len(self.TL) > self.tenure:
			self.TL.popleft()

	def constructive_stop_criteria(self) -> bool:
		"""
		A standard stopping criteria for the constructive heuristic is to repeat until the incumbent solution improves by inserting a new candidate element.
		Returns:
			True if the current solution did not improve.
		"""
		if self.maximize:
			return self.sol.cost < self.last_sol.cost
		else:
			return self.sol.cost > self.last_sol.cost

	def constructive_heuristic_std(self) -> Solution:
		"""
		The TS constructive heuristic, which is responsible for building a feasible solution by selecting candidate elements to enter the solution.
		Returns:
			A feasible solution to the problem.
		"""
		
		# Initialize the solution, Candidate List, and Restricted Candidate List
		self.sol = self.create_empty_sol()
		self.CL = self.make_CL()
		self.RCL = self.make_RCL()
		self.last_sol = self.sol.copy()

		# Main loop, which repeats until the stopping criteria is reached.
		while not self.constructive_stop_criteria():
			self.last_sol = self.sol.copy()
			if not self.RCL:
				break
			
			# Select a candidate element from the RCL (here we select randomly, but other strategies can be used)
			selected = random.choice(list(self.RCL))
			self.sol.add(selected)
			self.obj_function.evaluate(self.sol)

			# Update the Candidate List and Restricted Candidate List
			self.update_CL()
			self.RCL = self.make_RCL()
			
		# Return the best solution found during the constructive phase
		if self.last_sol.cost > self.sol.cost:
			self.sol = self.last_sol.copy()
		
		if self.verbose:
			print(f"Constructive Heuristic found solution: {self.sol}")
		
		return self.sol

	def constructive_heuristic_greedy(self) -> Solution:
		"""
		A greedy constructive heuristic that builds a solution by iteratively adding the best candidate element from the
		Candidate List until a feasible solution is formed.
		Returns:
			A feasible solution to the problem.
		"""
		self.sol = self.create_empty_sol()
		self.CL = self.make_CL()
		
		while not self.obj_function.is_feasible(self.sol):
			best_elem = None
			best_cost = None
			for elem in self.CL:
				delta = self.obj_function.evaluate_insertion_cost(elem, self.sol)
				if best_elem is None or (self.maximize and delta > best_cost) or (not self.maximize and delta < best_cost):
					best_elem = elem
					best_cost = delta
			self.sol.add(best_elem)
			self.obj_function.evaluate(self.sol)
			self.update_CL()

		if self.verbose:
			print(f"Greedy Heuristic: found solution {self.sol}")
		return self.sol

	def constructive_heuristic_cost_ratio(self) -> Solution:
		"""
		A cost-ratio constructive heuristic that builds a solution by iteratively adding the candidate element from the
		Candidate List that has the best cost-to-coverage ratio until a feasible solution is formed
		Returns:
			A feasible solution to the problem.
		"""
		self.sol = self.create_empty_sol()
		self.CL = self.make_CL()

		while not self.obj_function.is_feasible(self.sol):
			best_elem = None
			best_ratio = None
			for elem in self.CL:
				cost = self.obj_function.evaluate_insertion_cost(elem, self.sol)
				coverage = len(self.obj_function.sets[elem])
				ratio = cost / coverage if coverage > 0 else float('inf') # Avoid division by zero
				if best_elem is None or (self.maximize and ratio > best_ratio) or (not self.maximize and ratio < best_ratio):
					best_elem = elem
					best_ratio = ratio
			self.sol.add(best_elem)
			self.obj_function.evaluate(self.sol)
			self.update_CL()

		if self.verbose:
			print(f"Cost-Ratio Heuristic: found solution {self.sol}")
		return self.sol
	
	def solve(self) -> Solution:
		"""
		The TS mainframe. It consists of a constructive heuristic followed by a loop, in which each iteration a neighborhood move is performed on the current solution. The best solution is returned as result.
		Returns:
			The best feasible solution obtained throughout all iterations.
		"""
		# Initialize the best solution and Tabu List
		self.best_sol = self.create_empty_sol()
		self.TL = self.make_TL()
		solutions = []

		# Apply the constructive heuristic to find an initial solution
		if self.constructive_type == 'std':
			self.constructive_heuristic_std()
		elif self.constructive_type == 'greedy':
			self.constructive_heuristic_greedy()
		elif self.constructive_type == 'cost_ratio':
			self.constructive_heuristic_cost_ratio()

		# Initialize no improvement counter and iteration counter
		no_improv = 0
		i = 0

		while no_improv < self.no_improv_iter and i < self.max_iter:
			# Apply a neighborhood move to the current solution
			self.neighborhood_move()

			# Update the best solution found so far
			if self.maximize and self.sol.cost > self.best_sol.cost:
				# Assuming Solution has a copy constructor or similar
				self.best_sol = self.sol.copy()
				solutions.append(self.best_sol)
				no_improv = 0
				if self.verbose:
					print(f"(Iter. {i}) best_sol = {self.best_sol}")
			elif not self.maximize and self.sol.cost < self.best_sol.cost:
				# Assuming Solution has a copy constructor or similar
				self.best_sol = self.sol.copy()
				solutions.append(self.best_sol)
				no_improv = 0
				if self.verbose:
					print(f"(Iter. {i}) best_sol = {self.best_sol}")
			else:
				no_improv += 1
				print(f"No improvement in iteration {i}. No improvement count: {no_improv}")
			i += 1

		return self.best_sol

