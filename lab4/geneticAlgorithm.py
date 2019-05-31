import numpy as np 

class GeneticAlgorithm(object): 
	"""
		Implement a simple generationl genetic algorithm as described in the instructions
	"""

	def __init__(	self, chromosomeShape,
					errorFunction,
					elitism = 1,
					populationSize = 25, 
					mutationProbability  = .1, 
					mutationScale = .5,
					numIterations = 10000, 
					errorTreshold = 1e-6
					): 

		self.populationSize = populationSize # size of the population of units
		self.p = mutationProbability # probability of mutation
		self.numIter = numIterations # maximum number of iterations
		self.e = errorTreshold # threshold of error while iterating
		self.f = errorFunction # the error function (reversely proportionl to fitness)
		self.keep = elitism  # number of units to keep for elitism
		self.k = mutationScale # scale of the gaussian noise

		self.i = 0 # iteration counter

		# initialize the population randomly from a gaussian distribution
		# with noise 0.1 and then sort the values and store them internally

		self.population = []
		for _ in range(populationSize):
			chromosome = np.random.randn(chromosomeShape) * 0.1

			fitness = self.calculateFitness(chromosome)
			self.population.append((chromosome, fitness))

		# sort descending according to fitness (larger is better)
		self.population = sorted(self.population, key=lambda t: -t[1])
	
	def step(self):	
		"""
			Run one iteration of the genetic algorithm. In a single iteration,
			you should create a whole new population by first keeping the best
			units as defined by elitism, then iteratively select parents from
			the current population, apply crossover and then mutation.

			The step function should return, as a tuple: 
				
			* boolean value indicating should the iteration stop (True if 
				the learning process is finished, False othwerise)
			* an integer representing the current iteration of the 
				algorithm
			* the weights of the best unit in the current iteration

		"""
		
		self.i += 1
		new_population = []

		# elitism
		new_population += self.bestN(self.keep)

		while len(new_population) < self.populationSize:
			parent1, parent2 = self.selectParents()
			child = self.crossover(parent1, parent2)
			self.mutate(child)
			new_population.append((child, self.calculateFitness(child)))

		self.population = sorted(new_population, key=lambda t: -t[1])

		best_weight = self.best()
		stop_condition = self.i == self.numIter or self.f(best_weight) < self.e

		return stop_condition, self.i, best_weight

	def calculateFitness(self, chromosome):
		"""
			Implement a fitness metric as a function of the error of
			a unit. Remember - fitness is larger as the unit is better!
		"""
		chromosomeError = self.f(chromosome)

		#return -chromosomeError
		return 1./chromosomeError

	def bestN(self, n):		
		"""
			Return the best n units from the population
		"""

		return self.population[:n]

	def best(self):
		"""
			Return the best unit from the population
		"""

		return self.population[0][0]

	def selectParents(self):
		"""
			Select two parents from the population with probability of 
			selection proportional to the fitness of the units in the
			population		
		"""
		fitness_vec = [fitness for chromosome, fitness in self.population]

		# normalization
		#min_value = abs(min(fitness_vec))*1.01
		#fitness_vec = [fit + min_value for fit in fitness_vec]

		fitness_sum = sum(fitness_vec)
		fitness_vec = [fit / fitness_sum for fit in fitness_vec]
		fitness_cumsum = np.cumsum(fitness_vec)


		parents = []
		while len(parents) < 2:
			pick = np.random.random()  # [0, 1)
			parent = np.searchsorted(fitness_cumsum, pick, side='left')
			if parent not in parents: parents.append(parent)

		return self.population[parents[0]][0], self.population[parents[1]][0]


	def crossover(self, p1, p2):
		"""
			Given two parent units p1 and p2, do a simple crossover by 
			averaging their values in order to create a new child unit
		"""

		return (p1 + p2)/2.

	def mutate(self, chromosome):
		"""
			Given a unit, mutate its values by applying gaussian noise
			according to the parameter k
		"""
		length = len(chromosome)

		prob_of_mutation = np.random.random(length)
		index_to_mutate = (prob_of_mutation < self.p).astype(int)
		chromosome += index_to_mutate*np.random.normal(scale=self.k, size=length)
