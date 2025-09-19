from abc import ABC, abstractmethod
import random
from collections import deque
from TabuSearch.Solution import Solution
from problems.Evaluator import Evaluator

# Type variable for generic candidate element

class AbstractTS(ABC):
	"""
	Abstract base class for Tabu Search metaheuristics (minimization problem).
	"""

	# flag that indicates whether the code should print more information on screen
	verbose: bool = True


	def __init__(self, obj_function: Evaluator, tenure: int, iterations: int = 3, maximize: bool = True):
		"""
		Constructor for the AbstractTS class.
		Args:
			obj_function: The objective function being minimized.
			tenure: The Tabu tenure parameter.
			iterations: The number of iterations which the TS will be executed.
		"""
		# Iniatialize basic parameters of the solver
		self.objFunction = obj_function
		self.iterations = iterations # Iterations without improvement
		self.maximize = maximize

		# Size of the tabu list, if tenure <= 0, use domain size as tenure otherwise use domain size // tenure (ratio)
		self.tenure = obj_function.get_domain_size() if tenure <= 0 else obj_function.get_domain_size() // tenure
		self.TL: deque = deque()

		# Initialize the solution, Candidate List, and Restricted Candidate List
		self.sol = None
		self.last_cost: float = float('-inf') if maximize else float('inf')
		self.CL = None
		self.RCL = None
		self.bestSol = None

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
			return self.cost < self.last_cost
		else:
			return self.cost > self.last_cost

	def constructive_heuristic(self) -> Solution:
		"""
		The TS constructive heuristic, which is responsible for building a feasible solution by selecting candidate elements to enter the solution.
		Returns:
			A feasible solution to the problem.
		"""
		
		# Initialize the solution, Candidate List, and Restricted Candidate List
		self.sol = self.create_empty_sol()
		self.CL = self.make_CL()
		self.RCL = self.make_RCL()
		

		# Main loop, which repeats until the stopping criteria is reached.
		while self.constructive_stop_criteria():
			self.last_cost = self.sol.cost
			if not self.RCL:
				break
			
			# Select a candidate element from the RCL (here we select randomly, but other strategies can be used)
			selected = random.choice(list(self.RCL))
			self.sol.add(selected)
			self.objFunction.evaluate(self.sol)
			
			# Update the Candidate List and Restricted Candidate List
			self.update_CL()
			self.RCL = self.make_RCL()

			if self.verbose:
				print(f"Constructive Heuristic: Made removed {selected}, New Solution Cost: {self.sol.cost}")

		return self.sol

	def solve(self) -> Solution:
		"""
		The TS mainframe. It consists of a constructive heuristic followed by a loop, in which each iteration a neighborhood move is performed on the current solution. The best solution is returned as result.
		Returns:
			The best feasible solution obtained throughout all iterations.
		"""
		# Initialize the best solution and Tabu List
		self.bestSol = self.create_empty_sol()
		self.TL = self.make_TL()

		# Apply the constructive heuristic to find an initial solution
		self.constructive_heuristic()
		no_improv = 0
		i = 0

		while no_improv < self.iterations:
			self.neighborhood_move()
			if self.maximize and self.sol.cost > self.bestSol.cost:
				# Assuming Solution has a copy constructor or similar
				self.bestSol = self.sol.copy()
				if self.verbose:
					print(f"(Iter. {i}) BestSol = {self.bestSol}")
			elif not self.maximize and self.sol.cost < self.bestSol.cost:
				# Assuming Solution has a copy constructor or similar
				self.bestSol = self.sol.copy()
				if self.verbose:
					print(f"(Iter. {i}) BestSol = {self.bestSol}")
			else:
				no_improv += 1
			i += 1
		return self.bestSol

