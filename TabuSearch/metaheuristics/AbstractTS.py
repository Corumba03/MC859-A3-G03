from abc import ABC, abstractmethod
from collections import deque

# Type variable for generic candidate element

class AbstractTS(ABC):
	"""
	Abstract base class for Tabu Search metaheuristics (minimization problem).
	"""

	# flag that indicates whether the code should print more information on screen
	verbose: bool = True

	# a random number generator
	import random
	rng = random.Random(0)

	def __init__(self, obj_function, tenure: int, iterations: int):
		"""
		Constructor for the AbstractTS class.
		Args:
			obj_function: The objective function being minimized.
			tenure: The Tabu tenure parameter.
			iterations: The number of iterations which the TS will be executed.
		"""
		self.ObjFunction = obj_function
		self.tenure = tenure
		self.iterations = iterations
		self.bestCost: float = float('inf')
		self.cost: float = float('inf')
		self.bestSol = None
		self.sol = None
		self.CL: list = []
		self.RCL: list = []
		self.TL: deque = deque()

	@abstractmethod
	def makeCL(self) -> list:
		"""
		Creates the Candidate List, which is a list of candidate elements that can enter a solution.
		Returns:
			The Candidate List.
		"""
		pass

	@abstractmethod
	def makeRCL(self) -> list:
		"""
		Creates the Restricted Candidate List, which is a list of the best candidate elements that can enter a solution.
		Returns:
			The Restricted Candidate List.
		"""
		pass

	@abstractmethod
	def makeTL(self) -> deque:
		"""
		Creates the Tabu List, which is a deque of the Tabu candidate elements.
		Returns:
			The Tabu List.
		"""
		pass

	@abstractmethod
	def updateCL(self) -> None:
		"""
		Updates the Candidate List according to the incumbent solution self.sol.
		Responsible for updating the costs of the candidate solution elements.
		"""
		pass

	@abstractmethod
	def createEmptySol(self):
		"""
		Creates a new solution which is empty, i.e., does not contain any candidate solution element.
		Returns:
			An empty solution.
		"""
		pass

	@abstractmethod
	def neighborhoodMove(self):
		"""
		The TS local search phase is responsible for repeatedly applying a neighborhood operation while the solution is getting improved, until a local optimum is attained. When a local optimum is attained the search continues by exploring moves which can make the current solution worse. Cycling is prevented by not allowing forbidden (tabu) moves that would otherwise backtrack to a previous solution.
		Returns:
			A local optimum solution.
		"""
		pass

	def constructiveHeuristic(self):
		"""
		The TS constructive heuristic, which is responsible for building a feasible solution by selecting in a greedy fashion, candidate elements to enter the solution.
		Returns:
			A feasible solution to the problem being minimized.
		"""
		self.CL = self.makeCL()
		self.RCL = self.makeRCL()
		self.sol = self.createEmptySol()
		self.cost = float('inf')

		# Main loop, which repeats until the stopping criteria is reached.
		while not self.constructiveStopCriteria():
			maxCost = float('-inf')
			minCost = float('inf')
			self.cost = self.sol.cost
			self.updateCL()

			# Explore all candidate elements to enter the solution, saving the highest and lowest cost variation achieved by the candidates.
			for c in self.CL:
				deltaCost = self.ObjFunction.evaluateInsertionCost(c, self.sol)
				if deltaCost < minCost:
					minCost = deltaCost
				if deltaCost > maxCost:
					maxCost = deltaCost

			# Among all candidates, insert into the RCL those with the highest performance.
			for c in self.CL:
				deltaCost = self.ObjFunction.evaluateInsertionCost(c, self.sol)
				if deltaCost <= minCost:
					self.RCL.append(c)

			# Choose a candidate randomly from the RCL
			rndIndex = self.rng.randint(0, len(self.RCL) - 1)
			inCand = self.RCL[rndIndex]
			self.CL.remove(inCand)
			self.sol.add(inCand)
			self.ObjFunction.evaluate(self.sol)
			self.RCL.clear()

		return self.sol

	def solve(self):
		"""
		The TS mainframe. It consists of a constructive heuristic followed by a loop, in which each iteration a neighborhood move is performed on the current solution. The best solution is returned as result.
		Returns:
			The best feasible solution obtained throughout all iterations.
		"""
		self.bestSol = self.createEmptySol()
		self.constructiveHeuristic()
		self.TL = self.makeTL()
		for i in range(self.iterations):
			self.neighborhoodMove()
			if self.bestSol.cost > self.sol.cost:
				# Assuming Solution has a copy constructor or similar
				self.bestSol = self.sol.copy() if hasattr(self.sol, 'copy') else type(self.sol)(self.sol)
				if self.verbose:
					print(f"(Iter. {i}) BestSol = {self.bestSol}")
		return self.bestSol

	def constructiveStopCriteria(self) -> bool:
		"""
		A standard stopping criteria for the constructive heuristic is to repeat until the incumbent solution improves by inserting a new candidate element.
		Returns:
			True if the criteria is met.
		"""
		return False if self.cost > self.sol.cost else True
