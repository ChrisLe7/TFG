"""labeling_time_series.py: Algoritmos para el etiquetado de series temporales."""

__author__ = "Christian Luna Escudero"

__email__ = "chris.luna.e@gmail.com"

__contact__ = "https://github.com/ChrisLe7"

__status__ = "Prototype"

__date__ = 29 / 4 / 2023

__copyright__ = "Trabajo Fin de Grado para la Universidad de Córdoba"

__credits__ = "Trabajo Fin de Grado para la Universidad de Córdoba"

import numpy as np
import itertools
from multiprocessing import Pool
import sys
import math

from common_functions import find_motifs, genetic_operators

class Exhaustive_Fixed_Window(object):
	"""docstring for Exhaustive_Fixed_Window"""
	def __init__(self, time_serie: np.array, size_motif: int):
		"""
		Constructor de la clase. Inicializa las variables y comprueba que la configuración asignada sea valida.
		:param time_serie: Serie temporal donde se pretende etiquetar.
		:param size_motif: Tamaño de ventana que tendrán las etiquetas.

		:return: 
		"""
		super(Exhaustive_Fixed_Window, self).__init__()
		self.time_serie = time_serie
		self.size_motif = size_motif
		self.len_time_serie = len(time_serie)
		self.num_of_motifs = 50
		if not(self.__check_size_motif__()):
			raise ValueError("Tamaño de Ventana No Permitido debe permitir etiquetar la serie completa de forma completa y ser >= 10")

	def __check_size_motif__(self):
		"""
		Función encargada de verificar que el tamaño de ventana asignado permite un etiquetado completo de la serie temporal.

		:return: Indica si el tamaño de ventana es valido o no.
		"""

		return (self.size_motif >= 10 and self.len_time_serie % self.size_motif == 0)
		
	def __evaluate_subsequence__(self, time_serie:np.array, motif:list, inicio : int):
		"""
		Función encargada de calcular el NRMSE de la subsecuencia donde se colocará el motif.
		:param time_serie: : Serie temporal original donde se encuentra la subsecuencia a evaluar.
		:param motif: Indica la posición de inicio y tamaño de ventana de la subsecuencia a evaluar.
		:param start_index: Indica el índice donde se encuentra la subsecuencia a evaluar dentro de la serie temporal original.

		:return: Devuelve el NRMSE cometido de la estimación de la subsecuencia en cuestión.
		"""

		inicio_subsecuencia = motif[0]
		size_window_motif = motif[1]
		fin_subsecuencia = inicio_subsecuencia + size_window_motif
		subsecuencia_motif = time_serie[inicio_subsecuencia:fin_subsecuencia]
		subsecuencia_real = time_serie[inicio: inicio + size_window_motif]

		dif_subsecuencia = (subsecuencia_real - subsecuencia_motif)
		max_subsecuencia_real, min_subsecuencia_real = subsecuencia_real.max(), subsecuencia_real.min()
		data_range = max_subsecuencia_real - min_subsecuencia_real
		del subsecuencia_real, subsecuencia_motif

		current_fitness = math.sqrt((dif_subsecuencia**2).mean())
		current_fitness /= data_range
		del dif_subsecuencia
		#print(current_fitness)
		return current_fitness

	def __fitness__(self, pattern):
		"""
		Función encargada de calcular el NRMSE de la estimación propuesta.
		:param pattern: Indica las “etiquetas” utilizadas para la estimación de la serie temporal.

		:return: Devuelve el NRMSE cometido de la estimación obtenida por la unión de las etiquetas.
		"""

		#index_subsequence = [motif[1] for motif in pattern]
		#index_subsequence = list(itertools.accumulate(index_subsequence))
		#total_fitness = sum(map(lambda interval, current_index: evaluar_subsecuencia(time_serie, interval, current_index), pattern, index_subsequence))
		
		total_fitness, current_index = 0, 0
		for motif in pattern:
			total_fitness += self.__evaluate_subsequence__(self.time_serie, motif, current_index)
			current_index += motif[1]

		return total_fitness

	def prepare_motifs(self):
		"""
		Función encargada de buscar los motifs de la serie temporal para el etiquetado.
		
		:return: 
		"""

		#matrix_profile_struct = mp.algorithms.scrimp_plus_plus(self.time_serie,self.size_motif, random_state=0)
		#motifs = mp.discover.motifs(matrix_profile_struct,k=self.num_of_motifs)["motifs"]
		motifs = find_motifs.__struct_motifs__(self.time_serie,self.size_motif)[1]
		self.list_motifs = [[motif["motifs"][0], self.size_motif] for motif in motifs]

	def label_time_serie(self):
		"""
		Función encargada de buscar la mejor combinación de etiquetas que proporcionan el menor error posible de estimación.

		:return: Retorna la mejor combinación de etiquetas que proporcionan un menor NRMSE
		"""
		#combinations = itertools.combinations_with_replacement(self.list_motifs, (self.len_time_serie // self.size_motif))
		combinations = itertools.product(*[self.list_motifs] * (self.len_time_serie // self.size_motif))
		list_pattern = [[pattern, self.__fitness__(pattern)] for pattern in combinations]
		best_pattern = min(list_pattern, key= lambda pattern: pattern[1])
		del list_pattern
		best_pattern[0] = list(best_pattern[0])
		return best_pattern

	def __fitness__parallel__(self, arg):
		"""
		Función encargada de calcular el NRMSE de la estimación propuesta, de forma paralelizada.
		:param arg: Indica tanto la serie temporal como el patrón a evaluar.

		:return: Devuelve el NRMSE cometido de la estimación obtenida por la unión de las etiquetas.
		"""

		total_fitness, current_index = 0, 0
		for motif in arg[0]:
			total_fitness += self.__evaluate_subsequence__(arg[1], motif, current_index)
			current_index += motif[1]

		return [arg[0], total_fitness]

	def label_time_serie_parallel(self, num_threads:int = 2):
		"""
		Función encargada de buscar la mejor combinación de etiquetas que proporcionan el menor error posible de estimación, de forma paralelizada.
		:param num_threads: Entero que especifica el número de hilos que se utilizará para la construcción de la estructura. Por defecto, toma el valor 2.

		:return: Retorna la mejor combinación de etiquetas que proporcionan un menor NRMSE
		"""

		#combinations = itertools.combinations_with_replacement(self.list_motifs, (self.len_time_serie // self.size_motif))
		combinations = itertools.product(*[self.list_motifs] * (self.len_time_serie // self.size_motif))
		vectorTuplas = [[pattern, self.time_serie] for pattern in combinations]

		with Pool(processes = num_threads) as pool:
			list_pattern = pool.map(self.__fitness__parallel__, vectorTuplas)
		
		best_pattern = min(list_pattern, key= lambda pattern: pattern[1])
		del list_pattern, vectorTuplas
		best_pattern[0] = list(best_pattern[0])
		return best_pattern

class Evo_Fixed_Window(object):
	"""docstring for Evo_Fixed_Window"""
	def __init__(self, time_serie: np.array, size_motif: int, size_population: int, type_crossover: int, type_mutation: int, type_repair: int, type_replacement: int):
		"""
		Constructor de la clase. Inicializa las variables y comprueba que la configuración asignada sea valida.
		:param time_serie: Serie temporal donde se pretende etiquetar.
		:param size_motif: Tamaño de ventana que tendrán las etiquetas.
		:param size_population: Tamaño de la población que tendrá el algoritmo evolutivo.
		:param type_crossover:  Indica el tipo de cruce que será utilizado por el algoritmo evolutivo. Si es 1 será utilizado el cruce en un punto, sino el cruce en dos puntos.
		:param type_mutation: Indica el tipo de mutación que será utilizado por el algoritmo evolutivo. Si es 1 será utilizada la mutación de reemplazo de motifs, si es 2 será utilizado el swap de etiquetas y sino será el desplazo de una etiqueta al final de la serie temporal.
		:param type_repair: Indica el tipo de reparación que será utilizado por el algoritmo evolutivo.
		:param type_replacement: Indica el tipo de reemplazo que será utilizado por el algoritmo evolutivo. Si es 1 será utilizado un reemplazo generacional simple, si es 2 un reemplazo generacional con elitismo y sino será utilizado un elitismo puro (quedando en todo momento en la población los mejores individuos).

		:return: 
		"""

		super(Evo_Fixed_Window, self).__init__()
		self.time_serie = time_serie
		self.size_motif = size_motif
		self.len_time_serie = len(time_serie)
		self.num_of_motifs = 50
		self.size_population = size_population if size_population > 0 else 100
		self.cross = genetic_operators.crossover_1pt if type_crossover == 1 else genetic_operators.crossover_2pt
		self.mutate = genetic_operators.mutate if type_mutation == 1 else genetic_operators.mutate_swap if type_replacement == 2 else genetic_operators.mutar_displacement
		self.repair = genetic_operators.repair_motif if type_repair == 1 else genetic_operators.repair_motif
		self.replacement = genetic_operators.generational if type_replacement == 1 else genetic_operators.generational_with_elite if type_replacement == 2 else genetic_operators.full_elite

		if not(self.__check_size_motif__()):
			raise ValueError("Tamaño de Ventana No Permitido debe permitir etiquetar la serie completa de forma completa y ser >= 10")

	def __check_size_motif__(self):
		"""
		Función encargada de verificar que el tamaño de ventana asignado permite un etiquetado completo de la serie temporal.

		:return: Indica si el tamaño de ventana es valido o no.
		"""
		return (self.size_motif >= 10 and self.len_time_serie % self.size_motif == 0)

	def __evaluate_subsequence__(self, time_serie:np.array, motif:list, inicio : int):
		"""
		Función encargada de calcular el NRMSE de la subsecuencia donde se colocará el motif.
		:param time_serie: : Serie temporal original donde se encuentra la subsecuencia a evaluar.
		:param motif: Indica la posición de inicio y tamaño de ventana de la subsecuencia a evaluar.
		:param start_index: Indica el índice donde se encuentra la subsecuencia a evaluar dentro de la serie temporal original.

		:return: Devuelve el NRMSE cometido de la estimación de la subsecuencia en cuestión.
		"""

		inicio_subsecuencia = motif[0]
		size_window_motif = motif[1]
		fin_subsecuencia = inicio_subsecuencia + size_window_motif
		subsecuencia_motif = time_serie[inicio_subsecuencia:fin_subsecuencia]
		subsecuencia_real = time_serie[inicio: inicio + size_window_motif]

		dif_subsecuencia = (subsecuencia_real - subsecuencia_motif)
		max_subsecuencia_real, min_subsecuencia_real = subsecuencia_real.max(), subsecuencia_real.min()
		data_range = max_subsecuencia_real - min_subsecuencia_real
		del subsecuencia_real, subsecuencia_motif

		current_fitness = math.sqrt((dif_subsecuencia**2).mean())
		current_fitness /= data_range
		del dif_subsecuencia
		#print(current_fitness)
		return current_fitness

	def __fitness__(self, pattern):
		"""
		Función encargada de calcular el NRMSE de la estimación propuesta.
		:param pattern: Indica las “etiquetas” utilizadas para la estimación de la serie temporal.

		:return: Devuelve el NRMSE cometido de la estimación obtenida por la unión de las etiquetas.
		"""

		total_fitness, current_index = 0, 0
		for motif in pattern:
			total_fitness += self.__evaluate_subsequence__(self.time_serie, motif, current_index)
			current_index += motif[1]

		return total_fitness

	def prepare_motifs(self):
		"""
		Función encargada de buscar los motifs de la serie temporal para el etiquetado.
		
		:return: 
		"""

		motifs = find_motifs.__struct_motifs__(self.time_serie,self.size_motif)[1]
		self.list_motifs = {self.size_motif: motifs}
	
	def generate_population(self):
		"""
		Función encargada de generar una población inicial para el algoritmo evolutivo.
		
		:return: 
		"""

		population_without_fitness = [genetic_operators.generate_individual(self.list_motifs, self.len_time_serie) for i in range(self.size_population)]
		self.population = [[pattern, self.__fitness__(pattern)] for pattern in population_without_fitness]
		self.population.sort(key = lambda x: x[1])
		del population_without_fitness

	def __fitness__parallel__(self, arg):
		"""
		Función encargada de calcular el NRMSE de la estimación propuesta, de forma paralelizada.
		:param arg: Indica tanto la serie temporal como el patrón a evaluar.

		:return: Devuelve el NRMSE cometido de la estimación obtenida por la unión de las etiquetas.
		"""

		total_fitness, current_index = 0, 0
		for motif in arg[0]:
			total_fitness += self.__evaluate_subsequence__(arg[1], motif, current_index)
			current_index += motif[1]

		return [arg[0], total_fitness]

	def evolve(self, num_epocs:int):
		"""
		Función encargada de evolucionar la población, haciendo uso de los operadores genéticos configurados.
		:param num_epocs: Indica el número de épocas a utilizar para el algoritmo evolutivo.

		:return: Retorna la mejor combinación de etiquetas que proporcionan un menor NRMSE
		"""

		early_stopping = 0
		for epocs in range(num_epocs):
			#start_time_epocs = time.time()
			new_population = []

			for i in range(0, self.size_population, 2):
				fathers = genetic_operators.select_by_tournament(self.population, 3)

				offspring_1 = self.cross(fathers[0][0], fathers[1][0])
				offspring_1 = self.repair(offspring_1, self.list_motifs, self.len_time_serie)
				offspring_1 = self.mutate(offspring_1, self.list_motifs)

				offspring_2 = self.cross(fathers[1][0], fathers[0][0])
				offspring_2 = self.repair(offspring_2, self.list_motifs, self.len_time_serie)
				offspring_2 = self.mutate(offspring_2, self.list_motifs)

				new_population = new_population + [[offspring_1, self.__fitness__(offspring_1)], [offspring_2, self.__fitness__(offspring_2)]]
				
				del fathers, offspring_1, offspring_2

			new_population.sort(key = lambda x: x[1])

			tmp = self.replacement(self.population, new_population)

			del self.population, new_population

			self.population = tmp

			self.population.sort(key = lambda x: x[1])

			#end_time_epocs = time.time()
			# Check Early Stopping

			if epocs == 0:
				best = self.population[0]
			elif best[1] > self.population[0][1]:
				best = self.population[0]
				early_stopping = 0
			else: 
				early_stopping +=1
			
			if early_stopping >= 50:
				break
		return self.population[0]

	def evolve_parallel(self, num_epocs:int, num_threads:int = 2):
		"""
		Función encargada de evolucionar la población, haciendo uso de los operadores genéticos configurados.
		:param num_epocs: Indica el número de épocas a utilizar para el algoritmo evolutivo.
		:param num_threads: Entero que especifica el número de hilos que se utilizará para agilizar la evaluación de la población. Por defecto, toma el valor 2.

		:return: Retorna la mejor combinación de etiquetas que proporcionan un menor NRMSE
		"""

		early_stopping = 0
		for epocs in range(num_epocs):
			#start_time_epocs = time.time()
			new_population = []

			for i in range(0, self.size_population, 2):
				fathers = genetic_operators.select_by_tournament(self.population, 3)

				offspring_1 = self.cross(fathers[0][0], fathers[1][0])
				offspring_1 = self.repair(offspring_1, self.list_motifs, self.len_time_serie)
				offspring_1 = self.mutate(offspring_1, self.list_motifs)

				offspring_2 = self.cross(fathers[1][0], fathers[0][0])
				offspring_2 = self.repair(offspring_2, self.list_motifs, self.len_time_serie)
				offspring_2 = self.mutate(offspring_2, self.list_motifs)

				new_population = new_population + [offspring_1, offspring_2]
				
				del fathers, offspring_1, offspring_2

			vectorTuplas = [[pattern, self.time_serie] for pattern in new_population]

			with Pool(processes = num_threads) as pool:
				list_pattern = pool.map(self.__fitness__parallel__, vectorTuplas)
			
			del new_population, vectorTuplas
			
			new_population = list_pattern

			new_population.sort(key = lambda x: x[1])

			tmp = self.replacement(self.population, new_population)

			del self.population, new_population, list_pattern

			self.population = tmp

			self.population.sort(key = lambda x: x[1])

			#end_time_epocs = time.time()
			# Check Early Stopping

			if epocs == 0:
				best = self.population[0]
			elif best[1] > self.population[0][1]:
				best = self.population[0]
				early_stopping = 0
			else: 
				early_stopping +=1
			
			if early_stopping >= 50:
				break
		return self.population[0]

class Evo_Variable_Window(object):
	"""docstring for Evo_Variable_Window"""
	def __init__(self, time_serie: np.array, min_size_motif: int, size_population: int, type_crossover: int, type_mutation: int, type_repair: int, type_replacement: int):
		"""
		Constructor de la clase. Inicializa las variables y comprueba que la configuración asignada sea valida.
		:param time_serie: Serie temporal donde se pretende etiquetar.
		:param min_size_motif: : Indica el tamaño mínimo de ventana que podrán tener las etiquetas que se utilizarán para el etiquetado.
		:param size_population: Tamaño de la población que tendrá el algoritmo evolutivo.
		:param type_crossover:  Indica el tipo de cruce que será utilizado por el algoritmo evolutivo. Si es 1 será utilizado el cruce en un punto, sino el cruce en dos puntos.
		:param type_mutation: Indica el tipo de mutación que será utilizado por el algoritmo evolutivo. Si es 1 será utilizada la mutación de reemplazo de motifs, si es 2 será utilizado el swap de etiquetas y sino será el desplazo de una etiqueta al final de la serie temporal.
		:param type_repair: Indica el tipo de reparación que será utilizado por el algoritmo evolutivo.
		:param type_replacement: Indica el tipo de reemplazo que será utilizado por el algoritmo evolutivo. Si es 1 será utilizado un reemplazo generacional simple, si es 2 un reemplazo generacional con elitismo y sino será utilizado un elitismo puro (quedando en todo momento en la población los mejores individuos).

		:return: 
		"""

		super(Evo_Variable_Window, self).__init__()
		self.time_serie = time_serie
		self.min_size_motif = min_size_motif
		self.len_time_serie = len(time_serie)
		self.num_of_motifs = 50
		self.size_population = size_population if size_population > 0 else 100
		self.cross = genetic_operators.crossover_1pt if type_crossover == 1 else genetic_operators.crossover_2pt
		self.mutate = genetic_operators.mutate if type_mutation == 1 else genetic_operators.mutate_swap if type_replacement == 2 else genetic_operators.mutar_displacement
		self.repair = genetic_operators.repair_motif if type_repair == 1 else genetic_operators.repair_motif
		self.replacement = genetic_operators.generational if type_replacement == 1 else genetic_operators.generational_with_elite if type_replacement == 2 else genetic_operators.full_elite

		if not(self.__check_size_motif__()):
			raise ValueError("Tamaño de Ventana No Permitido debe de ser >= 10")

	def __check_size_motif__(self):
		"""
		Función encargada de verificar que el tamaño de ventana asignado permite un etiquetado completo de la serie temporal.

		:return: Indica si el tamaño de ventana es valido o no.
		"""

		return (self.min_size_motif >= 10)

	def __evaluate_subsequence__(self, time_serie:np.array, motif:list, inicio : int):
		"""
		Función encargada de calcular el NRMSE de la subsecuencia donde se colocará el motif.
		:param time_serie: : Serie temporal original donde se encuentra la subsecuencia a evaluar.
		:param motif: Indica la posición de inicio y tamaño de ventana de la subsecuencia a evaluar.
		:param start_index: Indica el índice donde se encuentra la subsecuencia a evaluar dentro de la serie temporal original.

		:return: Devuelve el NRMSE cometido de la estimación de la subsecuencia en cuestión.
		"""

		inicio_subsecuencia = motif[0]
		size_window_motif = motif[1]
		fin_subsecuencia = inicio_subsecuencia + size_window_motif
		subsecuencia_motif = time_serie[inicio_subsecuencia:fin_subsecuencia]
		subsecuencia_real = time_serie[inicio: inicio + size_window_motif]

		dif_subsecuencia = (subsecuencia_real - subsecuencia_motif)
		max_subsecuencia_real, min_subsecuencia_real = subsecuencia_real.max(), subsecuencia_real.min()
		data_range = max_subsecuencia_real - min_subsecuencia_real
		del subsecuencia_real, subsecuencia_motif

		current_fitness = math.sqrt((dif_subsecuencia**2).mean())
		current_fitness /= data_range
		del dif_subsecuencia
		#print(current_fitness)
		return current_fitness

	def __fitness__(self, pattern):
		"""
		Función encargada de calcular el NRMSE de la estimación propuesta.
		:param pattern: Indica las “etiquetas” utilizadas para la estimación de la serie temporal.

		:return: Devuelve el NRMSE cometido de la estimación obtenida por la unión de las etiquetas.
		"""

		total_fitness, current_index = 0, 0
		for motif in pattern:
			total_fitness += self.__evaluate_subsequence__(self.time_serie, motif, current_index)
			current_index += motif[1]

		return total_fitness

	def prepare_motifs(self, num_threads:int=2):
		"""
		Función encargada de buscar los motifs de la serie temporal para el etiquetado.
		
		:return: 
		"""
		self.list_motifs = dic_motifs = find_motifs.generate_struct_motifs_parallel(self.time_serie,self.min_size_motif, self.len_time_serie//3, num_threads = num_threads)
	
	def generate_population(self):
		"""
		Función encargada de generar una población inicial para el algoritmo evolutivo.
		
		:return: 
		"""

		population_without_fitness = [genetic_operators.generate_individual(self.list_motifs, self.len_time_serie) for i in range(self.size_population)]
		self.population = [[pattern, self.__fitness__(pattern)] for pattern in population_without_fitness]
		self.population.sort(key = lambda x: x[1])
		del population_without_fitness

	def __fitness__parallel__(self, arg):
		"""
		Función encargada de calcular el NRMSE de la estimación propuesta, de forma paralelizada.
		:param arg: Indica tanto la serie temporal como el patrón a evaluar.

		:return: Devuelve el NRMSE cometido de la estimación obtenida por la unión de las etiquetas.
		"""

		total_fitness, current_index = 0, 0
		for motif in arg[0]:
			total_fitness += self.__evaluate_subsequence__(arg[1], motif, current_index)
			current_index += motif[1]

		return [arg[0], total_fitness]

	def evolve(self, num_epocs:int):
		"""
		Función encargada de evolucionar la población, haciendo uso de los operadores genéticos configurados.
		:param num_epocs: Indica el número de épocas a utilizar para el algoritmo evolutivo.

		:return: Retorna la mejor combinación de etiquetas que proporcionan un menor NRMSE
		"""

		early_stopping = 0
		for epocs in range(num_epocs):
			#start_time_epocs = time.time()
			new_population = []

			for i in range(0, self.size_population, 2):
				fathers = genetic_operators.select_by_tournament(self.population, 3)

				offspring_1 = self.cross(fathers[0][0], fathers[1][0])
				offspring_1 = self.repair(offspring_1, self.list_motifs, self.len_time_serie)
				offspring_1 = self.mutate(offspring_1, self.list_motifs)

				offspring_2 = self.cross(fathers[1][0], fathers[0][0])
				offspring_2 = self.repair(offspring_2, self.list_motifs, self.len_time_serie)
				offspring_2 = self.mutate(offspring_2, self.list_motifs)

				new_population = new_population + [[offspring_1, self.__fitness__(offspring_1)], [offspring_2, self.__fitness__(offspring_2)]]
				
				del fathers, offspring_1, offspring_2

			new_population.sort(key = lambda x: x[1])

			tmp = self.replacement(self.population, new_population)

			del self.population, new_population

			self.population = tmp

			self.population.sort(key = lambda x: x[1])

			#end_time_epocs = time.time()
			# Check Early Stopping

			if epocs == 0:
				best = self.population[0]
			elif best[1] > self.population[0][1]:
				best = self.population[0]
				early_stopping = 0
			else: 
				early_stopping +=1
			
			if early_stopping >= 50:
				break
		return self.population[0]

	def evolve_parallel(self, num_epocs:int, num_threads:int = 6):
		"""
		Función encargada de evolucionar la población, haciendo uso de los operadores genéticos configurados.
		:param num_epocs: Indica el número de épocas a utilizar para el algoritmo evolutivo.
		:param num_threads: Entero que especifica el número de hilos que se utilizará para agilizar la evaluación de la población. Por defecto, toma el valor 2.

		:return: Retorna la mejor combinación de etiquetas que proporcionan un menor NRMSE
		"""

		early_stopping = 0
		for epocs in range(num_epocs):
			#start_time_epocs = time.time()
			new_population = []

			for i in range(0, self.size_population, 2):
				fathers = genetic_operators.select_by_tournament(self.population, 3)

				offspring_1 = self.cross(fathers[0][0], fathers[1][0])
				offspring_1 = self.repair(offspring_1, self.list_motifs, self.len_time_serie)
				offspring_1 = self.mutate(offspring_1, self.list_motifs)

				offspring_2 = self.cross(fathers[1][0], fathers[0][0])
				offspring_2 = self.repair(offspring_2, self.list_motifs, self.len_time_serie)
				offspring_2 = self.mutate(offspring_2, self.list_motifs)

				new_population = new_population + [offspring_1, offspring_2]
				
				del fathers, offspring_1, offspring_2

			vectorTuplas = [[pattern, self.time_serie] for pattern in new_population]

			with Pool(processes = num_threads) as pool:
				list_pattern = pool.map(self.__fitness__parallel__, vectorTuplas)
			
			del new_population, vectorTuplas
			
			new_population = list_pattern

			new_population.sort(key = lambda x: x[1])

			tmp = self.replacement(self.population, new_population)

			del self.population, new_population, list_pattern

			self.population = tmp

			self.population.sort(key = lambda x: x[1])

			#end_time_epocs = time.time()
			# Check Early Stopping

			if epocs == 0:
				best = self.population[0]
			elif best[1] > self.population[0][1]:
				best = self.population[0]
				early_stopping = 0
			else: 
				early_stopping +=1
			
			if early_stopping >= 50:
				break
		return self.population[0]

if __name__ == '__main__':
	import pandas as pd
	import time
	import random
	def main_exhaustive_algorithm():
		name_file = "../datasets/own_synthetic_datasets/synthetic_Seed_1.csv"
		df = pd.read_csv(name_file)
		time_serie = df[df.columns[1]].to_numpy()
		date = df[df.columns[0]].to_numpy()
		size_motif = 250
		a = time.time()
		model = Exhaustive_Fixed_Window(time_serie, size_motif)
		model.prepare_motifs()
		label_ts = model.label_time_serie()
		b = time.time()
		print(f"Tiempo requerido {b-a} segundos")
		print(f"Fitness Best pattern {label_ts[1]}, Pattern {label_ts[0]}")

	def main_evo_fixed_algorithm():
		name_file = "../datasets/own_synthetic_datasets/synthetic_Seed_1.csv"
		df = pd.read_csv(name_file)
		time_serie = df[df.columns[1]].to_numpy()
		date = df[df.columns[0]].to_numpy()
		size_motif = 250
		a = time.time()
		model = Evo_Fixed_Window(time_serie, size_motif, 100, 1, 2, 1, 1)
		model.prepare_motifs()
		model.generate_population()
		label_ts = model.evolve(100)
		b = time.time()
		print(f"Tiempo requerido {b-a} segundos")
		print(f"Fitness Best pattern {label_ts[1]}, Pattern {label_ts[0]}")

	def main_evo_variable_algorithm():
		name_file = "../datasets/own_synthetic_datasets/synthetic_Seed_1.csv"
		df = pd.read_csv(name_file)
		time_serie = df[df.columns[1]].to_numpy()
		date = df[df.columns[0]].to_numpy()
		size_motif = 10
		a = time.time()
		model = Evo_Variable_Window(time_serie, size_motif, 100, 1, 2, 1, 1)
		model.prepare_motifs()
		model.generate_population()
		label_ts = model.evolve(100)
		b = time.time()
		print(f"Tiempo requerido {b-a} segundos")
		print(f"Fitness Best pattern {label_ts[1]}, Pattern {label_ts[0]}")

	main_exhaustive_algorithm()
	#main_evo_fixed_algorithm()
	#main_evo_variable_algorithm()